"""Quantization mechanics shared by the QAT driver.

Everything here answers one question: *which* parts of a model get quantized,
and how. The answer matters for the paper's Sec. 3.5 claims, because the
published quantization-aware-training run was reported by an audit of the
pre-release driver to have attached its qconfig to the whole wrapper -- putting
``FakeQuantize`` observers on the torchlibrosa STFT / log-mel front end, whose
output spans roughly -100 to +40 dB, and on ``bn0`` -- and to have skipped conv-bn
fusion entirely, on an x86 (``fbgemm``) backend although the deployment target is a
Cortex-A53. That driver is not shipped here and was not re-run, so this account is
not independently confirmed; see ``docs/errata.md`` E1 for what the shipped log does
establish.

The scope enforced here is the backbone and nothing else:
``spectrogram_extractor``, ``logmel_extractor``, ``spec_augmenter`` and ``bn0``
stay in FP32 in both modes.

Paper artifact served: :mod:`mirasol.training.qat`, i.e. the QAT sentence of
Sec. 3.5 and the abstract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.ao.quantization as tq
import torch.nn as nn

__all__ = [
    "PreparationInfo",
    "QAT_BACKENDS",
    "QAT_MODES",
    "QuantizableBackbone",
    "capture_backbone_input",
    "convert_backbone",
    "fuse_conv_bn_relu",
    "prepare_backbone_for_qat",
]

#: Quantization backends. ``qnnpack`` is the ARM backend and the default because
#: the deployment target of the paper is a Cortex-A53; ``fbgemm``/``x86`` are the
#: x86 backends and only reproduce the (mismatched) published configuration.
QAT_BACKENDS: tuple[str, ...] = ("qnnpack", "fbgemm", "x86")

#: Quantization frontends. ``fx`` (graph mode) is the default because it is the
#: only one of the two that yields a *runnable* INT8 model here: it fuses conv-bn
#: automatically and leaves ops with no quantized kernel (SiLU, Hardswish,
#: LayerNorm) in floating point, inserting the quant/dequant pairs itself.
#: ``eager`` reproduces the approach of the published run - manual fusion, manual
#: stubs - and is retained because it is what the paper's QAT numbers came from.
QAT_MODES: tuple[str, ...] = ("fx", "eager")


class QuantizableBackbone(nn.Module):
    """Quantization boundary around a model's backbone.

    Only what sits between the stubs is quantized. The mel front end
    (``Spectrogram`` / ``LogmelFilterBank``) and ``bn0`` remain outside and stay
    in FP32, which is the fix for observers having been placed on a tensor
    spanning roughly -100 to +40 dB.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.quant = tq.QuantStub()
        self.backbone = backbone
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize, run the backbone, dequantize."""
        return self.dequant(self.backbone(self.quant(x)))


def fuse_conv_bn_relu(module: nn.Module) -> list[list[str]]:
    """Fuse every ``Conv2d(+BatchNorm2d)(+ReLU)`` group found in ``module``.

    Folding batch norm into the preceding convolution before observers are
    inserted is what keeps QAT statistics meaningful; the published run skipped
    fusion entirely. Only ``nn.ReLU`` is fusable, so EfficientNet-B0 (SiLU) and
    MobileNetV3 (Hardswish) fuse conv-bn pairs and leave their activations alone.

    Args:
        module: the module to fuse in place. It must be in training mode, which
            is what ``fuse_modules_qat`` requires to emit ``ConvBn2d``.

    Returns:
        The list of fused module-name groups, empty for architectures with no
        conv-bn pairs (pure transformers).
    """
    if not module.training:
        raise RuntimeError(
            "fuse_conv_bn_relu requires the module in training mode so batch norm "
            "can be folded for QAT; call module.train() first"
        )
    groups = _conv_bn_relu_groups(module)
    if groups:
        tq.fuse_modules_qat(module, groups, inplace=True)
    return groups


def _conv_bn_relu_groups(module: nn.Module) -> list[list[str]]:
    """Collect dotted names of fusable conv-bn(-relu) runs.

    Types are compared exactly rather than with ``isinstance``: torch's fuser
    method table is keyed on exact types, and timm's ``BatchNormAct2d`` is a
    ``BatchNorm2d`` subclass that carries its own activation, so treating it as a
    plain batch norm would either raise "did not find fuser method" or silently
    drop the activation. Architectures built from it (the timm backbones) simply
    report zero fused groups.
    """
    groups: list[list[str]] = []
    for parent_name, parent in module.named_modules():
        children = list(parent.named_children())
        index = 0
        while index < len(children) - 1:
            name_a, child_a = children[index]
            name_b, child_b = children[index + 1]
            if not (type(child_a) is nn.Conv2d and type(child_b) is nn.BatchNorm2d):
                index += 1
                continue
            names = [name_a, name_b]
            step = 2
            if index + 2 < len(children):
                name_c, child_c = children[index + 2]
                if type(child_c) is nn.ReLU:
                    names.append(name_c)
                    step = 3
            prefix = f"{parent_name}." if parent_name else ""
            groups.append([prefix + name for name in names])
            index += step
    return groups


@dataclass(frozen=True)
class PreparationInfo:
    """What was done to a model before fine-tuning began."""

    qmode: str
    qbackend: str
    fused_groups: tuple[tuple[str, ...], ...]

    @property
    def n_fused(self) -> int:
        """Number of explicitly fused conv-bn(-relu) groups (0 in FX mode)."""
        return len(self.fused_groups)

    def describe(self) -> str:
        """One-line human-readable summary."""
        if self.qmode == "fx":
            return (
                f"FX graph mode, backend {self.qbackend}: conv-bn folded by the "
                "quantizer, unsupported ops left in FP32"
            )
        if self.n_fused:
            return (
                f"eager mode, backend {self.qbackend}: {self.n_fused} conv-bn(-relu) "
                "group(s) fused before prepare_qat"
            )
        return (
            f"eager mode, backend {self.qbackend}: no plain Conv2d+BatchNorm2d pairs "
            "to fuse (this backbone uses timm's BatchNormAct2d, which has no fuser "
            "method)"
        )


def capture_backbone_input(sed_model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """Return one example of the tensor the backbone actually receives.

    FX graph mode needs a representative input to trace with. Rather than
    recomputing the mel front end here -- and risking a shape that disagrees with
    what training produced -- this records the real tensor with a forward pre-hook
    on the backbone during one FP32 forward pass.

    Args:
        sed_model: a model with ``.backbone``, on the same device as ``waveform``.
        waveform: a real batch of waveforms, e.g. ``batch['waveform']``.

    Returns:
        The first element of the captured tensor, on the CPU, with the batch
        dimension kept.

    Raises:
        RuntimeError: if the backbone was not called during the forward pass.
    """
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Record the positional input the backbone was called with."""
        captured.append(args[0].detach())

    handle = sed_model.backbone.register_forward_pre_hook(hook)
    was_training = sed_model.training
    try:
        sed_model.eval()
        with torch.no_grad():
            sed_model(waveform, None, True)
    finally:
        handle.remove()
        sed_model.train(was_training)
    if not captured:
        raise RuntimeError(
            "the backbone was never called during the calibration forward pass, so "
            "no example input could be captured for FX tracing"
        )
    return captured[0][:1].detach().cpu()


def prepare_backbone_for_qat(
    sed_model: nn.Module,
    qbackend: str,
    qmode: str = "fx",
    example_input: torch.Tensor | None = None,
) -> PreparationInfo:
    """Insert fake-quantization observers into the backbone of ``sed_model``.

    The scope is the backbone and nothing else, so ``spectrogram_extractor``,
    ``logmel_extractor``, ``spec_augmenter`` and ``bn0`` stay in FP32. The
    published run attached the qconfig to the whole wrapper, which placed
    observers on the log-mel output.

    Args:
        sed_model: a model with a ``backbone`` attribute (every architecture here
            except HTS-AT, whose Swin blocks are not an eager-mode target and are
            not symbolically traceable either).
        qbackend: one of :data:`QAT_BACKENDS`.
        qmode: ``'fx'`` (graph mode, default) or ``'eager'``.
        example_input: representative backbone input, required for ``'fx'``; see
            :func:`capture_backbone_input`.

    Returns:
        A :class:`PreparationInfo` describing what was done.

    Raises:
        AttributeError: if ``sed_model`` has no ``backbone``.
        ValueError: if ``qmode`` is unknown, or ``'fx'`` is used without an
            ``example_input``.
    """
    if not hasattr(sed_model, "backbone"):
        raise AttributeError(
            f"{type(sed_model).__name__} has no .backbone attribute, so the "
            "quantization boundary cannot be scoped to exclude the mel front end. "
            "QAT here supports the timm and torchvision backbones, not HTS-AT."
        )
    torch.backends.quantized.engine = qbackend
    sed_model.train()

    if qmode == "fx":
        if example_input is None:
            raise ValueError(
                "qmode='fx' needs example_input; call capture_backbone_input first"
            )
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_qat_fx

        mapping = get_default_qat_qconfig_mapping(qbackend)
        try:
            sed_model.backbone = prepare_qat_fx(
                sed_model.backbone, mapping, (example_input,)
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"FX graph-mode preparation failed ({type(exc).__name__}: {exc}). "
                "This backbone is not symbolically traceable, which is the case for "
                "the windowed-attention transformers (TinyViT, FastViT). No QAT "
                "measurement is possible for it in fx mode; --qmode eager will "
                "prepare it but its converted form cannot run either."
            ) from exc
        return PreparationInfo(qmode=qmode, qbackend=qbackend, fused_groups=())

    if qmode != "eager":
        raise ValueError(f"unknown qmode {qmode!r}; choose from {', '.join(QAT_MODES)}")

    wrapped = QuantizableBackbone(sed_model.backbone)
    fused = fuse_conv_bn_relu(wrapped.backbone)
    # qconfig=None on the whole model first, then on the wrapper only, so
    # propagate_qconfig_ cannot reach the torchlibrosa front end or bn0.
    sed_model.qconfig = None  # type: ignore[assignment]
    wrapped.qconfig = tq.get_default_qat_qconfig(qbackend)  # type: ignore[assignment]
    sed_model.backbone = wrapped
    tq.prepare_qat(wrapped, inplace=True)
    return PreparationInfo(
        qmode=qmode,
        qbackend=qbackend,
        fused_groups=tuple(tuple(group) for group in fused),
    )


def convert_backbone(backbone: nn.Module, qmode: str) -> nn.Module:
    """Convert a prepared backbone from fake quantization to real INT8 ops.

    Args:
        backbone: the module returned by a previous
            :func:`prepare_backbone_for_qat` call, in eval mode and on CPU.
        qmode: the mode it was prepared with.

    Returns:
        The converted module. Exceptions are *not* caught here: the caller must
        surface them, because a conversion failure means no INT8 measurement
        exists and the preceding fake-quant score is not a substitute for one.
    """
    if qmode == "fx":
        from torch.ao.quantization.quantize_fx import convert_fx

        return convert_fx(backbone)
    if qmode != "eager":
        raise ValueError(f"unknown qmode {qmode!r}; choose from {', '.join(QAT_MODES)}")
    return tq.convert(backbone, inplace=False)
