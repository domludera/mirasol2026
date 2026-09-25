"""Shared plumbing for the scripts in ``analysis/`` that regenerate paper artifacts.

Serves every artifact produced from this directory -- Table 1, Table 2, Table 4,
Tables S1-S8, Table S10, and Figures 1 and S1 of "A Low-Cost AI Stethoscope for
Resource-Constrained Heart Murmur Screening" (Ludera & Kersten-Oertel, MIRASOL @
MICCAI 2026).

Four things live here so that no analysis script redefines them:

* the six architectures in the paper's Table 1 / Table 2 column order, together
  with the parameter counts printed in those column headers;
* path resolution: every default comes from :mod:`mirasol.paths`, every default
  is overridable by a command-line flag, and a missing input raises one
  actionable error instead of quietly producing a partial table;
* the patient-level decision rule used for every published number -- mean-pool a
  patient's *segment logits*, then argmax. Segment logits are unnormalised
  (``loss_type='weighted_clip_ce'``), so pooling logits and pooling softmax
  probabilities are different estimators; the paper pools logits;
* cohort assertions. Five folds must be present and must partition 941 patients.
  A missing fold file is an error, never a dropped row.

The two Challenge Score variants are imported from :mod:`mirasol.metrics` and are
never reimplemented here. ``challenge_score`` is the weighted-credit variant used
throughout the published tables; ``official_challenge_score`` is the PhysioNet
2022 murmur weighted accuracy. See ``docs/errata.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mirasol.metrics import challenge_score, official_challenge_score
from mirasol.paths import CIRCOR_AUDIO, CIRCOR_META, PREDS_DIR, REPO_ROOT, RESULTS_DIR

# --------------------------------------------------------------------------
# The cohort and the architectures
# --------------------------------------------------------------------------
FOLDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

#: Patients with an out-of-fold prediction. 942 are in the CirCor public
#: partition; patient 50321 is lost during dataset preparation because all four
#: of its recordings use the duplicate-location naming the loader does not match.
#: See ``docs/splits.md``.
N_PATIENTS: Final[int] = 941

#: Per-fold patient counts are 189/188/188/188/188.
MIN_FOLD_PATIENTS: Final[int] = 187
MAX_FOLD_PATIENTS: Final[int] = 189

#: Three-class target convention produced by ``mirasol.data.prepare``.
PRESENT: Final[int] = 0
UNKNOWN: Final[int] = 1
ABSENT: Final[int] = 2
CLASS_NAMES: Final[tuple[str, str, str]] = ("Present", "Unknown", "Absent")

#: Display name -> prediction-dump prefix, in the paper's Table 1 column order
#: (ascending trainable-parameter count).
MODELS: Final[Mapping[str, str]] = {
    "MobileNetV3-S": "mobilenetv3_3class",
    "FastViT": "fastvit_3class",
    "EfficientNet-B0": "efficientnet_b0_3class",
    "TinyViT": "tinyvit_3class",
    "ResNet-18": "resnet18_3class",
    "HTS-AT": "htsat_3class",
}

#: Display name -> architecture key used by ``mirasol.models.factory`` and by the
#: ``exports/<arch>/`` and ``results/benchmarks/`` artifacts.
ARCH_KEYS: Final[Mapping[str, str]] = {
    "MobileNetV3-S": "mobilenetv3",
    "FastViT": "fastvit",
    "EfficientNet-B0": "efficientnet_b0",
    "TinyViT": "tinyvit",
    "ResNet-18": "resnet18",
    "HTS-AT": "htsat",
}

#: Trainable-parameter counts as printed in the Table 1 column headers, in
#: millions. Regenerate with ``python -m mirasol.cli.count_params``.
PARAMS: Final[Mapping[str, str]] = {
    "MobileNetV3-S": "1.52",
    "FastViT": "3.26",
    "EfficientNet-B0": "4.01",
    "TinyViT": "5.07",
    "ResNet-18": "11.2",
    "HTS-AT": "27.6",
}

#: Systolic murmur grades that clinical guidance treats as unlikely to be
#: innocent (>= 3/6 intensity); used for endpoint E3 of Table 2.
HIGH_GRADE_SYSTOLIC: Final[frozenset[str]] = frozenset(
    {"III/VI", "IV/VI", "V/VI", "VI/VI"}
)

#: The four architectures that were also run in the two-class configuration
#: (Table S7).
BINARY_MODELS: Final[Mapping[str, str]] = {
    "EfficientNet-B0": "efficientnet_b0",
    "FastViT": "fastvit",
    "TinyViT": "tinyvit",
    "HTS-AT": "htsat",
}

#: Table 1 row order. All seven rows are printed by ``table1_fivefold.py``.
METRIC_ORDER: Final[tuple[str, ...]] = (
    "Challenge Score",
    "Accuracy",
    "Macro F1",
    "Precision (Present)",
    "Recall (Present)",
    "F1 (Present)",
    "F1 (Unknown)",
)

#: Reported alongside every Table 1 Challenge Score (VERIFIED_FINDINGS section 1).
OFFICIAL_METRIC: Final[str] = "Challenge Score (official PhysioNet)"

PHYSIONET_URL: Final[str] = "https://physionet.org/content/circor-heart-sound/"

_PREDS_HINT: Final[str] = (
    "results/predictions/ ships with the repository; pass --preds-dir to point at "
    "your own run directory, or set the environment variable used by mirasol.paths."
)
_META_HINT: Final[str] = (
    "training_data.csv is a 114 KB download from " + PHYSIONET_URL + " (the audio "
    "is not needed for any table); place it as data/circor/training_data.csv or "
    "pass --circor-meta."
)
_AUDIO_HINT: Final[str] = (
    "the recordings are the ~560 MB training_data/ directory from " + PHYSIONET_URL
    + "; place it as data/circor/training_data/ or pass --audio-dir."
)


# --------------------------------------------------------------------------
# Command-line surface, shared by every script in this directory
# --------------------------------------------------------------------------
def add_preds_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--preds-dir``, defaulting to the committed prediction dumps."""
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"directory of *.data.csv prediction dumps (default: {_rel(PREDS_DIR)})",
    )


def add_circor_meta_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--circor-meta``, the only CirCor file any table needs."""
    parser.add_argument(
        "--circor-meta",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "CirCor DigiScope training_data.csv, not redistributed here "
            f"(default: {_rel(CIRCOR_META)})"
        ),
    )


def add_audio_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--audio-dir`` for the one figure panel that needs a waveform."""
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "CirCor DigiScope training_data/ recordings, not redistributed here "
            f"(default: {_rel(CIRCOR_AUDIO)})"
        ),
    )


def add_out_argument(parser: argparse.ArgumentParser, default_subdir: str) -> None:
    """Add ``--out``, always under a repo-relative default, always created."""
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"output directory (default: {default_subdir})",
    )
    parser.set_defaults(_out_default=default_subdir)


def _rel(path: Path) -> str:
    """Render a path relative to the repository root, never as an absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def resolve_preds_dir(cli: Path | None) -> Path:
    """Resolve and validate the prediction-dump directory."""
    return _require_dir(cli or PREDS_DIR, "prediction dumps", _PREDS_HINT)


def resolve_circor_meta(cli: Path | None) -> Path:
    """Resolve and validate the CirCor metadata CSV."""
    return _require_file(cli or CIRCOR_META, "CirCor patient metadata", _META_HINT)


def resolve_audio_dir(cli: Path | None) -> Path:
    """Resolve and validate the CirCor recording directory."""
    return _require_dir(cli or CIRCOR_AUDIO, "CirCor recordings", _AUDIO_HINT)


def resolve_results_dir(cli: Path | None, subdir: str = "") -> Path:
    """Resolve and validate a directory under the committed ``results/`` tree."""
    base = cli or (RESULTS_DIR / subdir if subdir else RESULTS_DIR)
    return _require_dir(base, "committed results", "results/ ships with the repository.")


def resolve_out_dir(args: argparse.Namespace) -> Path:
    """Resolve ``--out`` against the repository root and create it."""
    out = args.out or (REPO_ROOT / args._out_default)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _require_dir(path: Path, what: str, hint: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"error: {what} not found at '{_rel(path)}'.\n  {hint}")
    return path


def _require_file(path: Path, what: str, hint: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"error: {what} not found at '{_rel(path)}'.\n  {hint}")
    return path


# --------------------------------------------------------------------------
# Segment predictions -> patient-level predictions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PatientPredictions:
    """Patient-level predictions produced by the paper's aggregation rule.

    ``logits`` holds each patient's mean-pooled segment logits, so
    ``predictions`` is the deployed argmax decision and ``probabilities`` is the
    softmax of the pooled logits used for the threshold sweeps of Table S6 and
    Fig. S1(a).
    """

    patients: tuple[str, ...]
    logits: np.ndarray
    targets: np.ndarray
    n_segments: int

    def __len__(self) -> int:
        return len(self.patients)

    @property
    def n_classes(self) -> int:
        return int(self.logits.shape[1])

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.logits, axis=1)

    @property
    def probabilities(self) -> np.ndarray:
        return softmax(self.logits)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def read_segment_predictions(path: Path) -> pd.DataFrame:
    """Read one ``*.data.csv`` dump of segment logits.

    Schema (no header row): ``logit_0, ..., logit_{C-1}, target, filename``.
    The number of classes is inferred from the column count so that the
    three-class and two-class dumps read through the same function.
    """
    frame = pd.read_csv(path, header=None)
    n_classes = frame.shape[1] - 2
    if n_classes < 2:
        raise SystemExit(
            f"error: '{path.name}' has {frame.shape[1]} columns; expected "
            "logit_0..logit_{C-1},target,filename with C >= 2."
        )
    frame.columns = [f"logit_{i}" for i in range(n_classes)] + ["target", "filename"]
    frame["patient"] = frame["filename"].astype(str).str.split("_").str[0]
    return frame


def pool_by_patient(frame: pd.DataFrame) -> PatientPredictions:
    """Mean-pool segment logits per patient, then argmax.

    This is the single aggregation rule behind every patient-level number in the
    paper, and the one place in ``analysis/`` that implements it. ``target`` is
    constant within a patient, so ``first`` is exact.

    ``mirasol.deployment.validate`` applies the same rule to live model output via
    :mod:`mirasol.metrics`; ``tests/test_metrics.py`` should assert that the two
    agree, since a divergence would silently change every published table.
    """
    logit_columns = [c for c in frame.columns if c.startswith("logit_")]
    aggregation = {c: (c, "mean") for c in logit_columns}
    aggregation["target"] = ("target", "first")
    pooled = frame.groupby("patient", as_index=False).agg(**aggregation)
    pooled = pooled.sort_values("patient", kind="mergesort").reset_index(drop=True)
    return PatientPredictions(
        patients=tuple(pooled["patient"].astype(str)),
        logits=pooled[logit_columns].to_numpy(dtype=float),
        targets=pooled["target"].to_numpy(dtype=int),
        n_segments=int(len(frame)),
    )


def load_patient_level(path: Path) -> PatientPredictions:
    """Read a prediction dump and aggregate it to the patient level."""
    return pool_by_patient(read_segment_predictions(path))


def fold_path(preds_dir: Path, prefix: str, fold: int) -> Path:
    """Path of one held-out fold's prediction dump."""
    return preds_dir / f"{prefix}_fold{fold}.data.csv"


def require_all_folds(preds_dir: Path, prefixes: Mapping[str, str]) -> None:
    """Exit non-zero listing every missing fold file, rather than dropping a model.

    ``aggregate_5fold_all.py`` warned and silently omitted an incomplete model,
    which is how two of Table 1's architectures came to be produced by a second
    script. Refusing to print a partial table is the fix.
    """
    missing = [
        _rel(fold_path(preds_dir, prefix, fold))
        for prefix in prefixes.values()
        for fold in FOLDS
        if not fold_path(preds_dir, prefix, fold).is_file()
    ]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(
            f"error: {len(missing)} prediction dump(s) missing, refusing to print a "
            f"partial table:\n  {listing}\n  {_PREDS_HINT}"
        )


def assert_fold_cohort(label: str, fold_sizes: Sequence[int]) -> None:
    """Assert five folds that partition the 941-patient analysis cohort."""
    if len(fold_sizes) != len(FOLDS):
        raise SystemExit(
            f"error: {label} has {len(fold_sizes)} folds, expected {len(FOLDS)}."
        )
    for fold, size in zip(FOLDS, fold_sizes, strict=True):
        if not MIN_FOLD_PATIENTS <= size <= MAX_FOLD_PATIENTS:
            raise SystemExit(
                f"error: {label} fold {fold} has {size} patients, expected "
                f"{MIN_FOLD_PATIENTS}-{MAX_FOLD_PATIENTS}."
            )
    total = int(sum(fold_sizes))
    if total != N_PATIENTS:
        raise SystemExit(
            f"error: {label} covers {total} patients across five folds, expected "
            f"{N_PATIENTS}. Per-fold counts: {list(fold_sizes)}."
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def three_class_metrics(patient: PatientPredictions) -> dict[str, float]:
    """Every metric Table 1 reports, plus the official Challenge Score.

    Both Challenge Score variants come from :mod:`mirasol.metrics`; the paper
    prints the first, and ``docs/errata.md`` explains why both are shown.
    """
    if patient.n_classes != 3:
        raise ValueError(f"expected three-class predictions, got {patient.n_classes}")
    labels = [PRESENT, UNKNOWN, ABSENT]
    y_true, y_pred = patient.targets, patient.predictions
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    return {
        "Challenge Score": challenge_score(y_true, y_pred),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Precision (Present)": float(per_class_precision[PRESENT]),
        "Recall (Present)": float(per_class_recall[PRESENT]),
        "F1 (Present)": float(per_class_f1[PRESENT]),
        "F1 (Unknown)": float(per_class_f1[UNKNOWN]),
        OFFICIAL_METRIC: official_challenge_score(y_true, y_pred),
    }


def two_class_metrics(patient: PatientPredictions) -> dict[str, float]:
    """Table S7 metrics: murmur (class 0) versus Absent (class 1)."""
    if patient.n_classes != 2:
        raise ValueError(f"expected two-class predictions, got {patient.n_classes}")
    y_true, y_pred = patient.targets, patient.predictions
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Specificity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def confusion_counts(patient: PatientPredictions) -> np.ndarray:
    """Row-true / column-predicted confusion matrix over the class indices."""
    size = patient.n_classes
    matrix = np.zeros((size, size), dtype=int)
    for true, predicted in zip(patient.targets, patient.predictions, strict=True):
        matrix[int(true), int(predicted)] += 1
    return matrix


# --------------------------------------------------------------------------
# ASCII and LaTeX formatting
# --------------------------------------------------------------------------
def d3(value: float) -> str:
    """Three decimals with the leading zero dropped, matching the paper."""
    if not np.isfinite(value):
        return "---"
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text


def pct(value: float) -> str:
    """One-decimal percentage, as printed in the Table 2 referral column."""
    return "---" if not np.isfinite(value) else f"{100 * value:.1f}"


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and population s.d. (``ddof=0``), the convention used in the paper."""
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)), float(np.std(array))


def ascii_mean_sd(values: Sequence[float]) -> str:
    """``0.886 +/- 0.008``; ASCII only, so a minimal console never fails."""
    mean, sd = mean_sd(values)
    return f"{mean:.3f} +/- {sd:.3f}"


def latex_mean_sd(values: Sequence[float]) -> str:
    """``.886$\\pm$.008``, leading zeros dropped as the paper does."""
    mean, sd = mean_sd(values)
    return f"{d3(mean)}$\\pm${d3(sd)}"


def print_header(title: str, width: int = 88) -> None:
    """ASCII section banner."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_latex_comment(title: str, width: int = 74) -> None:
    """Banner for a block meant to be pasted into the manuscript."""
    print("\n" + "%" * width)
    print(f"% {title}")
    print("%" * width)
