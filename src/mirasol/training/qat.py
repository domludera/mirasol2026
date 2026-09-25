"""Quantization-aware training for the murmur classifiers (library half).

Paper artifact served: the sentence in Sec. 3.5 / the abstract that
quantization-aware training "does not recover" the post-training INT8 loss.
``experiments/run_qat.py`` is the thin CLI over this module.

Read ``docs/errata.md`` before quoting a QAT number. The published figure was
produced by an earlier version of this code in which

1. ``torch.ao.quantization.convert`` raised ``cannot pickle 'module' object``
   and a bare ``except`` substituted the **last fake-quant evaluation** as the
   reported "QAT INT8 Challenge Score" -- so the published value is not a
   measurement of a quantized model;
and, as reported by an audit of the pre-release driver -- which this repository does
not ship and did not re-run, so items 2 to 4 are **not independently confirmed** --

2. the qconfig was attached to the whole wrapper, which would put ``FakeQuantize``
   observers on the torchlibrosa STFT / log-mel front end whose output spans
   roughly -100 to +40 dB, and on ``bn0``;
3. conv-bn fusion was skipped entirely; and
4. the backend was ``fbgemm`` (x86) although the deployment target is a
   Cortex-A53 (ARM, ``qnnpack``).

Item 1 is provable from the shipped ``results/logs/qat_results.log``; see
``docs/errata.md`` E1 for the evidence and for what it does not establish.

All four are addressed here, and the first in the strongest available
sense: a ``convert`` failure, or a failure of the post-convert evaluation, is
re-raised. There is no code path in this module by which a fake-quant score can
be reported as an INT8 score.

Design notes:

* the quantization mechanics live in :mod:`mirasol.training.quantization`;
  quantization is scoped to the backbone in both modes, so
  ``spectrogram_extractor``, ``logmel_extractor``, ``spec_augmenter`` and ``bn0``
  stay FP32;
* ``--qmode fx`` (the default) uses graph-mode quantization: it folds conv-bn
  itself and leaves operators with no quantized kernel in floating point.
  ``--qmode eager`` reproduces the approach of the published run -- an explicit
  ``QuantStub``/``DeQuantStub`` boundary plus manual conv-bn(-relu) fusion;
* PTQ reference scores are read from a measured artifact, never hardcoded;
* the FP32 checkpoint is chosen by the caller, so this module contains no
  checkpoint globbing and cannot pick up an unrelated run.

Measured on this repository's models (torch 2.10, x86 host, ``--qbackend
qnnpack``): ``fx`` converts and runs for EfficientNet-B0 and MobileNetV3-S;
``eager`` converts but then cannot run either of them, because ``aten::silu`` and
``aten::native_batch_norm`` have no QuantizedCPU kernel (timm's
``BatchNormAct2d`` cannot be fused). TinyViT and FastViT are not symbolically
traceable, and HTS-AT has no ``backbone`` attribute to scope to. Each of those is
a hard error naming the cause -- never a silently substituted score.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mirasol.metrics import (
    CLASS_NAMES,
    compute_metrics,
    patient_id_from_filename,
    pool_by_patient,
)
from mirasol.training.lightning_module import SEDWrapper
from mirasol.training.predictions import DEFAULT_PATIENT_RANGE
from mirasol.training.quantization import (
    QAT_BACKENDS,
    QAT_MODES,
    capture_backbone_input,
    convert_backbone,
    prepare_backbone_for_qat,
)

__all__ = [
    "QATResult",
    "QATSettings",
    "QAT_BACKENDS",
    "QAT_MODES",
    "add_qat_arguments",
    "evaluate_patient_level",
    "load_ptq_reference",
    "run_qat",
    "settings_from_args",
]

#: Fine-tuning defaults, unchanged from the run that produced the published log.
DEFAULT_QAT_EPOCHS = 10
DEFAULT_QAT_LR = 1e-5
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4


@dataclass(frozen=True)
class QATSettings:
    """Immutable settings for one quantization-aware-training run.

    Attributes:
        model_type: architecture key understood by ``mirasol.models.factory``.
        checkpoint: FP32 Lightning checkpoint to start from. Selecting it is the
            caller's job (``experiments/run_experiment.py`` owns the one shared
            checkpoint selector), so this module can never silently pick up a
            checkpoint from a different experiment.
        fold: held-out fold, mirrored into ``config.esc_fold``.
        epochs: fine-tuning epochs after the observers are inserted.
        learning_rate: AdamW learning rate for fine-tuning.
        batch_size: batch size for both fine-tuning and evaluation.
        num_workers: dataloader workers.
        qbackend: one of :data:`QAT_BACKENDS`.
        qmode: one of :data:`QAT_MODES`.
        eval_every: evaluate the fake-quantized model every N epochs (0 = never).
        ptq_reference: optional JSON of measured PTQ patient-level scores.
        output: optional JSON path for the result record.
        device: ``'cuda'``, ``'cpu'`` or ``None`` for "cuda if available".
        patient_range: inclusive bounds on the number of evaluated patients, or
            ``None`` to skip the check.
    """

    model_type: str
    checkpoint: Path
    fold: int = 0
    epochs: int = DEFAULT_QAT_EPOCHS
    learning_rate: float = DEFAULT_QAT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    qbackend: str = "qnnpack"
    qmode: str = "fx"
    eval_every: int = 2
    ptq_reference: Path | None = None
    output: Path | None = None
    device: str | None = None
    patient_range: tuple[int, int] | None = DEFAULT_PATIENT_RANGE

    def __post_init__(self) -> None:
        if self.qbackend not in QAT_BACKENDS:
            raise ValueError(
                f"unknown --qbackend {self.qbackend!r}; choose from "
                f"{', '.join(QAT_BACKENDS)}"
            )
        if self.qmode not in QAT_MODES:
            raise ValueError(
                f"unknown --qmode {self.qmode!r}; choose from {', '.join(QAT_MODES)}"
            )
        if self.epochs < 0:
            raise ValueError(f"--epochs must be >= 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"--batch-size must be > 0, got {self.batch_size}")

    def resolved_device(self) -> torch.device:
        """Device used for fine-tuning (conversion and INT8 eval are CPU-only)."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class QATResult:
    """Outcome of one QAT run, as written to ``--output``."""

    model_type: str
    fold: int
    checkpoint: str
    qbackend: str
    qmode: str
    epochs: int
    learning_rate: float
    fused_groups: int
    fp32: dict[str, float]
    fake_quant: dict[str, float]
    int8: dict[str, float]
    ptq_reference: float | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "model_type": self.model_type,
            "fold": self.fold,
            "checkpoint": self.checkpoint,
            "qbackend": self.qbackend,
            "qmode": self.qmode,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "fused_conv_bn_groups": self.fused_groups,
            "fp32": self.fp32,
            "fake_quant_final_epoch": self.fake_quant,
            "int8_converted": self.int8,
            "ptq_reference_challenge_score": self.ptq_reference,
            "environment": self.environment,
        }


def evaluate_patient_level(
    model: SEDWrapper,
    loader: DataLoader,
    device: torch.device,
    patient_range: tuple[int, int] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, float]:
    """Score a model at patient level: mean-pool segment logits, then argmax.

    Pooling and scoring both go through :mod:`mirasol.metrics`, so a QAT number
    is produced by exactly the rule that produced Table 1. Both Challenge Score
    variants are reported, because the project score in the paper and the official
    PhysioNet 2022 weighted accuracy differ substantially.

    Args:
        model: wrapper whose ``sed_model`` is called in inference mode.
        loader: dataloader over the held-out fold.
        device: device to run on.
        patient_range: inclusive bounds on the pooled patient count, or ``None``.
        class_names: label order; pass ``BINARY_CLASS_NAMES`` for a two-class run,
            in which case the two Challenge Scores are reported as ``nan``
            because neither is defined outside the three-class configuration.

    Raises:
        ValueError: if the number of pooled patients falls outside
            ``patient_range``.
    """
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            output_dict = model.sed_model(waveform, None, True)
            logits.append(output_dict["clipwise_output"].detach().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            names.extend(str(name) for name in batch["audio_name"])

    segment_logits = np.concatenate(logits, axis=0)
    segment_targets = np.concatenate(targets, axis=0)
    pooled = pool_by_patient(
        segment_logits,
        [patient_id_from_filename(name) for name in names],
        segment_targets,
    )
    if patient_range is not None:
        low, high = patient_range
        if not low <= len(pooled) <= high:
            raise ValueError(
                f"pooled {len(pooled)} patients, outside the expected range "
                f"{low}-{high}; the evaluation fold is not the one this comparison "
                "assumes"
            )
    metric_set = compute_metrics(
        pooled.y_true, pooled.y_pred, level="patient", class_names=class_names
    )
    return {
        "challenge_score": metric_set.project_challenge_score,
        "official_challenge_score": metric_set.official_physionet_weighted_accuracy,
        "accuracy": metric_set.accuracy,
        "macro_f1": metric_set.macro_f1,
        "n_patients": float(len(pooled)),
        "n_segments": float(segment_logits.shape[0]),
    }


def load_ptq_reference(path: Path, model_type: str) -> float:
    """Read the measured PTQ patient-level Challenge Score for ``model_type``.

    The published script hardcoded ``{'tinyvit': 0.784, 'efficientnet_b0': 0.784}``
    -- which is, as it happens, simply the score of an all-Absent classifier --
    so the QAT-versus-PTQ delta was compared against a literal. This reads it
    from a measured artifact instead.

    Expected JSON: an object keyed by architecture whose values are either the
    score itself or an object with a ``challenge_score`` key, e.g.::

        {"efficientnet_b0": {"challenge_score": 0.7841, "format": "int8_static"}}

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError: if ``model_type`` is absent.
        TypeError: if the entry is neither a number nor an object with a score.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"PTQ reference {resolved} not found. Produce it from the measured "
            "post-training-quantization run (analysis/patient_level_int8.py), or "
            "omit --ptq-reference to skip the PTQ comparison."
        )
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or model_type not in payload:
        available = ", ".join(sorted(payload)) if isinstance(payload, dict) else "none"
        raise KeyError(
            f"{resolved} has no entry for {model_type!r}; available keys: {available}"
        )
    entry = payload[model_type]
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for key in ("challenge_score", "Challenge Score", "patient_challenge_score"):
            if key in entry:
                return float(entry[key])
        raise TypeError(
            f"{resolved}[{model_type!r}] has no challenge_score key; keys: "
            f"{', '.join(sorted(entry))}"
        )
    raise TypeError(
        f"{resolved}[{model_type!r}] must be a number or an object with a "
        f"challenge_score key, got {type(entry).__name__}"
    )


def run_qat(config: Any, settings: QATSettings) -> QATResult:
    """Fine-tune one architecture with fake quantization, then convert and score.

    Args:
        config: resolved configuration. ``model_type``, ``esc_fold`` and
            ``classes_num`` must already agree with ``settings``; the caller owns
            the configuration and this function never rewrites it.
        settings: see :class:`QATSettings`.

    Returns:
        A :class:`QATResult` holding the FP32 baseline, the final fake-quant
        evaluation and the converted-INT8 evaluation, clearly separated.

    Raises:
        ValueError: if ``config`` disagrees with ``settings``.
        RuntimeError: if ``convert`` or the INT8 evaluation fails. This is
            deliberate: a fake-quant number must never be reportable as INT8.
    """
    _check_config_matches(config, settings)
    device = settings.resolved_device()
    # ("Present", "Unknown", "Absent"), or the two-class names for Table S7.
    class_names = config.class_names
    train_loader, eval_loader, eval_dataset = _build_loaders(config, settings)

    print(
        f"QAT {settings.model_type} fold {settings.fold}: {settings.qmode} mode, "
        f"backend {settings.qbackend}, {settings.epochs} epochs, "
        f"lr {settings.learning_rate}, device {device}",
        flush=True,
    )

    model = _load_fp32_model(config, settings, device)
    fp32_metrics = evaluate_patient_level(
        model, eval_loader, device, settings.patient_range, class_names
    )
    _print_metrics("FP32 baseline", fp32_metrics)

    example_input = None
    if settings.qmode == "fx":
        calibration_batch = next(iter(eval_loader))
        example_input = capture_backbone_input(
            model.sed_model, calibration_batch["waveform"].to(device)
        ).to(device)
    preparation = prepare_backbone_for_qat(
        model.sed_model, settings.qbackend, settings.qmode, example_input
    )
    print(f"  {preparation.describe()}", flush=True)
    model = model.to(device)

    fake_quant_metrics = _finetune(
        model, train_loader, eval_loader, device, config, settings, class_names
    )
    _print_metrics("fake-quantized (NOT an INT8 measurement)", fake_quant_metrics)

    int8_metrics = _convert_and_evaluate(
        model, eval_dataset, settings, class_names
    )
    _print_metrics("converted INT8", int8_metrics)

    ptq_reference: float | None = None
    if settings.ptq_reference is not None:
        ptq_reference = load_ptq_reference(settings.ptq_reference, settings.model_type)

    result = QATResult(
        model_type=settings.model_type,
        fold=settings.fold,
        checkpoint=str(settings.checkpoint),
        qbackend=settings.qbackend,
        qmode=settings.qmode,
        epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        fused_groups=preparation.n_fused,
        fp32=fp32_metrics,
        fake_quant=fake_quant_metrics,
        int8=int8_metrics,
        ptq_reference=ptq_reference,
        environment={
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "quantized_engine": torch.backends.quantized.engine,
        },
    )
    _print_summary(result)
    _write_result(result, config, settings)
    return result


# --------------------------------------------------------------------------- #
# Command-line surface (used by experiments/run_qat.py)
# --------------------------------------------------------------------------- #
def add_qat_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the QAT flags to ``parser`` and return it."""
    parser.add_argument(
        "--model-type",
        required=True,
        help="architecture to fine-tune, e.g. efficientnet_b0",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="FP32 Lightning checkpoint to start from",
    )
    parser.add_argument("--fold", type=int, default=0, help="held-out fold (default 0)")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_QAT_EPOCHS, help="fine-tuning epochs"
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_QAT_LR, dest="learning_rate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--qbackend",
        choices=QAT_BACKENDS,
        default="qnnpack",
        help=(
            "quantization backend; qnnpack (default) is the ARM backend matching "
            "the Cortex-A53 deployment target, fbgemm/x86 are x86 backends"
        ),
    )
    parser.add_argument(
        "--qmode",
        choices=QAT_MODES,
        default="fx",
        help=(
            "fx (default) uses graph-mode quantization, which folds conv-bn and "
            "leaves ops without a quantized kernel in FP32; eager reproduces the "
            "approach of the published run"
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=2,
        help="evaluate the fake-quantized model every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--ptq-reference",
        type=Path,
        default=None,
        help="JSON of measured PTQ patient-level Challenge Scores keyed by architecture",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result path (default runs/<exp_name>/qat_<model>_fold<n>.json)",
    )
    parser.add_argument(
        "--device", default=None, choices=("cpu", "cuda"), help="fine-tuning device"
    )
    parser.add_argument(
        "--expect-patients",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=list(DEFAULT_PATIENT_RANGE),
        help="inclusive bounds on the number of patients in the evaluated fold",
    )
    parser.add_argument(
        "--no-expect-patients",
        action="store_true",
        help="skip the patient-count check (custom or reduced extracts)",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> QATSettings:
    """Build immutable :class:`QATSettings` from parsed arguments."""
    patient_range: tuple[int, int] | None = None
    if not args.no_expect_patients:
        low, high = args.expect_patients
        patient_range = (int(low), int(high))
    return QATSettings(
        model_type=args.model_type,
        checkpoint=Path(args.checkpoint),
        fold=args.fold,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        qbackend=args.qbackend,
        qmode=args.qmode,
        eval_every=args.eval_every,
        ptq_reference=args.ptq_reference,
        output=args.output,
        device=args.device,
        patient_range=patient_range,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _check_config_matches(config: Any, settings: QATSettings) -> None:
    if config.model_type != settings.model_type:
        raise ValueError(
            f"config.model_type is {config.model_type!r} but --model-type is "
            f"{settings.model_type!r}; pass --set model_type={settings.model_type}"
        )
    if int(config.esc_fold) != settings.fold:
        raise ValueError(
            f"config.esc_fold is {config.esc_fold} but --fold is {settings.fold}; "
            f"pass --set esc_fold={settings.fold}"
        )


def _build_loaders(
    config: Any, settings: QATSettings
) -> tuple[DataLoader, DataLoader, Any]:
    """Build the fold's train and eval loaders from the prepared ``.npy``."""
    from mirasol.cli.train import load_fold_dataset  # local: CLI owns path resolution
    from mirasol.data.dataset import ESC_Dataset

    full_dataset = load_fold_dataset(config)
    train_dataset = ESC_Dataset(dataset=full_dataset, config=config, eval_mode=False)
    eval_dataset = ESC_Dataset(dataset=full_dataset, config=config, eval_mode=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    return train_loader, eval_loader, eval_dataset


def _load_fp32_model(
    config: Any, settings: QATSettings, device: torch.device
) -> SEDWrapper:
    from mirasol.models.factory import create_model

    if not settings.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint {settings.checkpoint} not found")
    sed_model = create_model(config)
    model = SEDWrapper(sed_model=sed_model, config=config, dataset=None)
    # weights_only=False is explicit: a Lightning checkpoint carries pickled
    # hyperparameters, so the default in torch >= 2.6 would reject it.
    ckpt = torch.load(settings.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device)


def _finetune(
    model: SEDWrapper,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    config: Any,
    settings: QATSettings,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Fine-tune with observers inserted; return the last fake-quant evaluation."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=0.01
    )
    weight = None
    if getattr(config, "loss_weight", None) is not None:
        weight = torch.FloatTensor(config.loss_weight).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    metrics: dict[str, float] | None = None
    for epoch in range(settings.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            waveform = batch["waveform"].to(device)
            target = batch["target"].to(device).long()
            pred = model.sed_model(waveform)["clipwise_output"]
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        average_loss = epoch_loss / max(n_batches, 1)
        is_last = epoch + 1 == settings.epochs
        due = settings.eval_every > 0 and (epoch + 1) % settings.eval_every == 0
        if due or is_last:
            metrics = evaluate_patient_level(
                model, eval_loader, device, settings.patient_range, class_names
            )
            print(
                f"  epoch {epoch + 1}/{settings.epochs}: loss={average_loss:.4f} "
                f"score={metrics['challenge_score']:.4f} "
                f"acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f}",
                flush=True,
            )
        else:
            print(
                f"  epoch {epoch + 1}/{settings.epochs}: loss={average_loss:.4f}",
                flush=True,
            )
    if metrics is None:
        # --epochs 0: observers inserted but no fine-tuning, still report the
        # fake-quant score so the INT8 number has something to be compared with.
        metrics = evaluate_patient_level(
            model, eval_loader, device, settings.patient_range, class_names
        )
    return metrics


def _convert_and_evaluate(
    model: SEDWrapper,
    eval_dataset: Any,
    settings: QATSettings,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Convert the fake-quantized backbone to INT8 and score it on CPU.

    Any failure is re-raised. The published run swallowed exactly this exception
    and printed the preceding fake-quant score under the heading "QAT INT8
    Challenge Score"; there is no fallback here by design.
    """
    model.eval()
    model.sed_model = model.sed_model.cpu()
    model = model.cpu()
    torch.backends.quantized.engine = settings.qbackend
    try:
        model.sed_model.backbone = convert_backbone(
            model.sed_model.backbone, settings.qmode
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(
            f"conversion to INT8 failed for {settings.model_type} in "
            f"{settings.qmode} mode ({type(exc).__name__}: {exc}). No INT8 "
            "measurement exists for this run: the preceding fake-quant score is NOT "
            "an INT8 score and must not be reported as one. See docs/errata.md."
        ) from exc

    cpu_loader = DataLoader(
        eval_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    try:
        return evaluate_patient_level(
            model, cpu_loader, torch.device("cpu"), settings.patient_range, class_names
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(
            f"the converted INT8 {settings.model_type} model failed to evaluate in "
            f"{settings.qmode} mode ({type(exc).__name__}: {exc}). No INT8 score "
            "exists for this run, and the fake-quant score is not a substitute. In "
            "eager mode this is expected for activations with no quantized kernel "
            "(SiLU, Hardswish) and for timm's unfused BatchNormAct2d; use "
            "--qmode fx, which leaves those ops in FP32."
        ) from exc


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"  [{label}] challenge_score={metrics['challenge_score']:.4f} "
        f"official={metrics['official_challenge_score']:.4f} "
        f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"n_patients={int(metrics['n_patients'])}",
        flush=True,
    )


def _print_summary(result: QATResult) -> None:
    fp32 = result.fp32["challenge_score"]
    int8 = result.int8["challenge_score"]
    print(f"\n  SUMMARY {result.model_type} fold {result.fold}", flush=True)
    print(f"  FP32 Challenge Score:            {fp32:.4f}", flush=True)
    if result.ptq_reference is not None:
        print(
            f"  PTQ INT8 Challenge Score:        {result.ptq_reference:.4f} "
            f"(delta vs FP32 {result.ptq_reference - fp32:+.4f})",
            flush=True,
        )
    print(
        f"  QAT INT8 Challenge Score:        {int8:.4f} (delta vs FP32 {int8 - fp32:+.4f})",
        flush=True,
    )
    if result.ptq_reference is not None:
        print(
            f"  QAT recovery over PTQ:           {int8 - result.ptq_reference:+.4f}",
            flush=True,
        )
    print(
        "  fake-quant score is reported separately and is not an INT8 measurement",
        flush=True,
    )


def _write_result(result: QATResult, config: Any, settings: QATSettings) -> Path:
    output = settings.output
    if output is None:
        output = config.run_dir / f"qat_{settings.model_type}_fold{settings.fold}.json"
    output = Path(output)
    # Refuses to create anything under version control.
    config.paths.ensure(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, sort_keys=True)
    print(f"  wrote {output}", flush=True)
    return output
