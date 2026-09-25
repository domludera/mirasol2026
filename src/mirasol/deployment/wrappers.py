# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
#
# Scope of the derivation: HTSATExportWrapper.forward below is a re-derivation of
# HTS_AT_Swin_Transformer.reshape_wav2img (upstream model/htsat.py, and
# mirasol/models/htsat.py here), whose spatial reshaping in turn follows
# Microsoft's Swin-Transformer. The rest of this module is new work.
"""ONNX export wrappers: log-mel spectrogram in, class logits out.

Serves Table 4 and Sec. 3.5: these are the modules that get traced to the ONNX
graphs published as GitHub Release assets.

Each wrapper reproduces the part of the training forward pass that must stay
inside the graph -- the ``bn0`` input normalisation and the time/frequency
transpose into vision layout -- and leaves everything upstream of it out, so the
graph's input is exactly the tensor ``mirasol.signal.mel`` (and its vendored
copy ``deployment/edge/mel.py``) produces.  Spectrogram extraction and
SpecAugment are deliberately outside the graph: the device computes the
spectrogram in numpy, and augmentation is training-only.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

MODEL_TYPES: Final[tuple[str, ...]] = (
    "efficientnet_b0",
    "fastvit",
    "htsat",
    "mobilenetv3",
    "resnet18",
    "tinyvit",
)

_TIMM_MODEL_TYPES: Final[tuple[str, ...]] = (
    "fastvit",
    "tinyvit",
    "mobilenetv3",
    "resnet18",
)


class TimmExportWrapper(nn.Module):
    """Export wrapper for the timm backbones.

    Covers FastViT, TinyViT, MobileNetV3-S and ResNet-18.

    Input:  ``(batch, 1, time_steps, mel_bins)`` log-mel spectrogram.
    Output: ``(batch, num_classes)`` unnormalised logits.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.bn0 = model.bn0
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)  # (B, mel_bins, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, mel_bins)
        x = x.transpose(2, 3)  # (B, 1, mel_bins, T) - vision layout
        return self.backbone(x)


class EfficientNetExportWrapper(TimmExportWrapper):
    """Export wrapper for the torchvision EfficientNet-B0 builder.

    Structurally identical to :class:`TimmExportWrapper`; kept as a distinct
    name because EfficientNet-B0 deliberately uses torchvision IMAGENET1K_V1
    weights rather than timm's, and every released ``efficientnet_b0/*.onnx``
    derives from that builder.
    """


class HTSATExportWrapper(nn.Module):
    """Export wrapper for HTS-AT (Swin Transformer).

    Flattens the branching training forward pass into the single path
    ``mel-spec -> reshape_wav2img -> forward_features -> clipwise_output``.

    The reshaping is a re-derivation of
    ``HTS_AT_Swin_Transformer.reshape_wav2img`` (MIT, (c) 2022 Knut(Ke) Chen;
    spatial reshaping after Microsoft Swin-Transformer, MIT).  Upstream asserts
    ``T <= spec_size * freq_ratio`` (1024); this pipeline feeds 2001 frames, so
    the assert is bypassed and the bicubic interpolation runs unconditionally.

    IMPORTANT, and recorded in ``docs/errata.md``: because of that interpolation
    this graph is a *latency-benchmark* graph, not an accuracy-equivalent one.
    The evaluated PyTorch path averages three native-resolution 1024-frame
    crops; this graph interpolates 2001 -> 1024 and runs one forward pass.
    Measured patient Challenge Score on fold 0: 0.618 for the graph vs 0.871
    for the PyTorch model.

    Input:  ``(batch, 1, time_steps, mel_bins)``.
    Output: ``(batch, num_classes)``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)  # (B, mel_bins, T, 1)
        x = self.model.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, mel_bins)

        target_t = int(self.model.spec_size * self.model.freq_ratio)  # 256*4 = 1024
        target_f = self.model.spec_size // self.model.freq_ratio  # 256/4 = 64
        x = torch.nn.functional.interpolate(
            x, (target_t, target_f), mode="bicubic", align_corners=True
        )

        # reshape_wav2img: (B, 1, T, F) -> (B, 1, spec_size, spec_size)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(
            x.shape[0], x.shape[1], x.shape[2],
            self.model.freq_ratio, x.shape[3] // self.model.freq_ratio,
        )
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])

        return self.model.forward_features(x)["clipwise_output"]


def get_export_wrapper(model_type: str, sed_model: nn.Module) -> nn.Module:
    """Return the export wrapper for ``model_type``.

    Raises:
        ValueError: unknown architecture.
    """
    if model_type == "htsat":
        return HTSATExportWrapper(sed_model)
    if model_type == "efficientnet_b0":
        return EfficientNetExportWrapper(sed_model)
    if model_type in _TIMM_MODEL_TYPES:
        return TimmExportWrapper(sed_model)
    raise ValueError(
        f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}"
    )
