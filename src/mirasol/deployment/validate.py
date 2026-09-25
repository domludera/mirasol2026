"""Validate a quantized ONNX graph against its FP32 PyTorch baseline.

Paper artifact: Sec. 3.5.  This is the script behind the FP32-vs-INT8 and
FP32-vs-FP16 agreement figures (the 100.0% / 99.9% FP16 agreements, and the
static-INT8 collapse) recorded in ``results/logs/int8_quantization.log`` and
``results/logs/static_fp16.log``.

Reads the same held-out fold twice -- once through the PyTorch model, once
through the ONNX graph, on one shared set of pre-computed log-mel inputs -- and
reports accuracy, macro F1 and Challenge Score at BOTH levels:

* **segment level**: one row per 30 s segment.  This is what the original
  script printed, unlabelled, which invites confusion with the paper's numbers.
* **patient level** (``--patient-level``): segment logits are mean-pooled per
  patient before the argmax, which is the aggregation every patient-level
  number in the paper uses.  The paper's 0.784 / 0.784 / 0.311 static-INT8
  figures in Sec. 3.5(3) are patient-level.

Challenge Score is reported twice, because the two definitions disagree
substantially (see ``docs/errata.md``):

* ``challenge_score`` -- the partial-credit, diagonal-normalised variant the
  paper's tables were computed with;
* ``official_challenge_score`` -- the official PhysioNet 2022 murmur weighted
  accuracy, ``sum_i w_i TP_i / sum_i w_i N_i`` with ``w = [5, 3, 1]``.

Usage
-----
    python -m mirasol.deployment.validate --model-type efficientnet_b0 \\
        --checkpoint runs/effnetb0_3class_fold0/.../l-epoch=142-acc=0.855.ckpt \\
        --onnx-model exports/efficientnet_b0/model_int8_static.onnx \\
        --patient-level
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

from mirasol.deployment.calibration import (
    CLASS_NAMES,
    Segment,
    load_fold_segments,
)
from mirasol.deployment.export import DEFAULT_DATASET, load_sed_model
from mirasol.deployment.wrappers import MODEL_TYPES, get_export_wrapper

try:
    from mirasol.metrics import challenge_score, official_challenge_score
except ImportError as error:  # pragma: no cover - packaging contract
    raise ImportError(
        "mirasol.metrics must provide challenge_score(y_true, y_pred) (the "
        "paper's partial-credit variant) and official_challenge_score("
        "y_true, y_pred) (PhysioNet 2022 weighted accuracy, w = [5, 3, 1])."
    ) from error

# F1 deltas at or below this are treated as quantization noise; above the second
# threshold the graph is reported as unusable.
F1_PASS_DELTA: Final[float] = 0.02
F1_WARN_DELTA: Final[float] = 0.05

# Do not shout "constant classifier" on a handful of items: with the cohort's
# ~72% Absent prior, an all-Absent answer over a dozen patients is unremarkable.
# The export-time guard uses a class-stratified hold-out precisely so that it
# does not need this caveat.
MIN_ITEMS_FOR_CONSTANT_WARNING: Final[int] = 20


@dataclass(frozen=True)
class Predictions:
    """Logits and argmax predictions for one backend over one segment list."""

    backend: str
    logits: np.ndarray  # (n_segments, n_classes) float64
    mean_latency_ms: float

    @property
    def labels(self) -> np.ndarray:
        return np.argmax(self.logits, axis=1)


@dataclass(frozen=True)
class ScoreSet:
    """The four numbers reported at one aggregation level."""

    level: str
    n_items: int
    accuracy: float
    macro_f1: float
    challenge_score: float
    official_challenge_score: float

    def as_mapping(self) -> Mapping[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "challenge_score": self.challenge_score,
            "official_challenge_score": self.official_challenge_score,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def run_pytorch(
    model_type: str, checkpoint: Path, segments: Sequence[Segment], classes_num: int
) -> Predictions:
    """Run the FP32 PyTorch baseline through the same export wrapper as ONNX."""
    sed_model = load_sed_model(model_type, checkpoint, classes_num)
    export_model = get_export_wrapper(model_type, sed_model).eval()

    logits: list[np.ndarray] = []
    latencies: list[float] = []
    with torch.no_grad():
        for index, segment in enumerate(segments, start=1):
            tensor = torch.from_numpy(segment.mel)
            started = time.perf_counter()
            output = export_model(tensor)
            latencies.append((time.perf_counter() - started) * 1000.0)
            logits.append(output.cpu().numpy().reshape(-1).astype(np.float64))
            if index % 100 == 0:
                print(f"    PyTorch {index}/{len(segments)}")
    return Predictions(
        backend="PyTorch FP32",
        logits=np.stack(logits),
        mean_latency_ms=float(np.mean(latencies)),
    )


def run_onnx(onnx_path: Path, segments: Sequence[Segment]) -> Predictions:
    """Run the ONNX graph, casting the input to the dtype the graph declares.

    Raises:
        FileNotFoundError: the graph is missing.
        ValueError: the graph's output width does not match ``CLASS_NAMES``.

    A session that fails to load raises out of here.  The original script
    caught it and returned empty prediction lists, which downstream code then
    reported as an aborted-but-successful run.
    """
    import onnxruntime as ort

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX graph not found: {onnx_path}")
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    dtype = np.float16 if input_meta.type == "tensor(float16)" else np.float32

    logits: list[np.ndarray] = []
    latencies: list[float] = []
    for index, segment in enumerate(segments, start=1):
        batch = np.ascontiguousarray(segment.mel, dtype=dtype)
        started = time.perf_counter()
        outputs = session.run(None, {input_meta.name: batch})
        latencies.append((time.perf_counter() - started) * 1000.0)
        row = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if row.size != len(CLASS_NAMES):
            raise ValueError(
                f"{onnx_path} emits {row.size} logits, expected "
                f"{len(CLASS_NAMES)} for {CLASS_NAMES}"
            )
        logits.append(row)
        if index % 100 == 0:
            print(f"    ONNX {index}/{len(segments)}")
    return Predictions(
        backend=f"ONNX ({onnx_path.name})",
        logits=np.stack(logits),
        mean_latency_ms=float(np.mean(latencies)),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation and scoring
# ═════════════════════════════════════════════════════════════════════════════

def patient_of(segment_name: str) -> str:
    """CirCor patient identifier of a segment.

    Segment names are ``{patient_id}_{location}.wav`` or
    ``{patient_id}_{location}-{window}.wav``; the paper's patient-level
    aggregation splits on the first underscore, so this does too.
    """
    return segment_name.split("_", 1)[0]


def pool_by_patient(
    segments: Sequence[Segment], logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool segment logits per patient, then return pooled logits + targets.

    Pooling happens on unnormalised logits, before any softmax, matching
    ``mirasol.metrics`` and the aggregation behind Table 1 and Table 2.

    Raises:
        ValueError: a patient's segments disagree about the target, which would
            mean the prepared dataset is inconsistent.
    """
    order: list[str] = []
    grouped: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        patient = patient_of(segment.name)
        if patient not in grouped:
            grouped[patient] = []
            order.append(patient)
        grouped[patient].append(index)

    pooled = np.empty((len(order), logits.shape[1]), dtype=np.float64)
    targets = np.empty(len(order), dtype=np.int64)
    for row, patient in enumerate(order):
        indices = grouped[patient]
        patient_targets = {segments[index].target for index in indices}
        if len(patient_targets) != 1:
            raise ValueError(
                f"patient {patient} has conflicting targets {sorted(patient_targets)} "
                "across its segments; the prepared dataset is inconsistent"
            )
        pooled[row] = logits[indices].mean(axis=0)
        targets[row] = patient_targets.pop()
    return pooled, targets


def score(level: str, y_true: np.ndarray, y_pred: np.ndarray) -> ScoreSet:
    """Accuracy, macro F1 and both Challenge Score definitions."""
    from sklearn.metrics import accuracy_score, f1_score

    return ScoreSet(
        level=level,
        n_items=int(y_true.size),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        challenge_score=float(challenge_score(y_true, y_pred)),
        official_challenge_score=float(official_challenge_score(y_true, y_pred)),
    )


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, ...]:
    """How often each class was predicted -- the constant-classifier tell."""
    return tuple(
        int(np.count_nonzero(y_pred == index)) for index in range(len(CLASS_NAMES))
    )


# ═════════════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════════════

def print_comparison(
    baseline: ScoreSet, candidate: ScoreSet, backends: tuple[str, str]
) -> None:
    """Side-by-side table for one aggregation level."""
    print(f"\n-- {baseline.level} ({baseline.n_items} items) " + "-" * 24)
    print(f"{'Metric':<28} {backends[0]:>20.20} {backends[1]:>20.20} {'Delta':>10}")
    print("-" * 82)
    reference = baseline.as_mapping()
    measured = candidate.as_mapping()
    for name in reference:
        delta = measured[name] - reference[name]
        print(
            f"{name:<28} {reference[name]:>20.4f} {measured[name]:>20.4f} "
            f"{delta:>+10.4f}"
        )


def print_prediction_spread(
    label: str, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """Predicted-class histogram, so a constant classifier is unmissable."""
    counts = confusion_counts(y_true, y_pred)
    spread = " / ".join(
        f"{count} {name}"
        for count, name in zip(counts, CLASS_NAMES, strict=True)
    )
    print(f"  {label:<22.22} {spread}")
    if max(counts) == y_pred.size and y_pred.size >= MIN_ITEMS_FOR_CONSTANT_WARNING:
        print(
            f"  WARNING: {label} is a CONSTANT classifier "
            f"({CLASS_NAMES[int(np.argmax(counts))]} for all {y_pred.size} items). "
            "It must not be deployed as a screener."
        )


def verdict(baseline_f1: float, candidate_f1: float) -> str:
    """One-line pass/warn/fail on the macro-F1 delta."""
    delta = abs(candidate_f1 - baseline_f1)
    if delta <= F1_PASS_DELTA:
        return f"PASS - macro-F1 delta {delta:.4f} within {F1_PASS_DELTA:.2f}."
    if delta <= F1_WARN_DELTA:
        return (
            f"REVIEW - macro-F1 delta {delta:.4f} is between {F1_PASS_DELTA:.2f} "
            f"and {F1_WARN_DELTA:.2f}."
        )
    return (
        f"FAIL - macro-F1 delta {delta:.4f} exceeds {F1_WARN_DELTA:.2f}. Re-export "
        "with per-channel INT8 weights, or use FP32."
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mirasol.deployment.validate",
        description=(
            "Compare a quantized ONNX graph against its FP32 PyTorch baseline "
            "on a held-out fold, at segment and patient level."
        ),
    )
    parser.add_argument("--model-type", required=True, choices=MODEL_TYPES)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx-model", required=True, type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="prepared .npy holding the folds",
    )
    parser.add_argument("--fold", type=int, default=0, help="held-out fold to evaluate")
    parser.add_argument("--classes-num", type=int, default=3, choices=(2, 3))
    parser.add_argument(
        "--max-samples", type=int,
        help="evaluate only the first N segments (quick check; patient-level "
             "numbers are then not comparable to the paper's)",
    )
    parser.add_argument(
        "--patient-level", action="store_true",
        help="also mean-pool segment logits per patient, which is the "
             "aggregation the paper's patient-level numbers use",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    segments = load_fold_segments(args.dataset, args.fold, args.max_samples)
    targets = np.asarray([segment.target for segment in segments], dtype=np.int64)
    truth = " / ".join(
        f"{int(np.count_nonzero(targets == index))} {name}"
        for index, name in enumerate(CLASS_NAMES)
    )
    print(f"Fold {args.fold}: {len(segments)} segments, truth {truth}")
    print(f"Input shape: {segments[0].mel.shape}")

    print("\n-- PyTorch FP32 --")
    baseline = run_pytorch(
        args.model_type, args.checkpoint, segments, args.classes_num
    )
    print(f"  mean latency {baseline.mean_latency_ms:.1f} ms/segment")

    print(f"\n-- {args.onnx_model.name} --")
    candidate = run_onnx(args.onnx_model, segments)
    print(f"  mean latency {candidate.mean_latency_ms:.1f} ms/segment")

    backends = (baseline.backend, candidate.backend)
    print(f"\n{'=' * 74}")
    print(f"VALIDATION REPORT - {args.model_type} vs {args.onnx_model.name}")
    print(f"{'=' * 74}")

    baseline_segment = score("segment-level", targets, baseline.labels)
    candidate_segment = score("segment-level", targets, candidate.labels)
    print_comparison(baseline_segment, candidate_segment, backends)
    print("\n  predicted-class spread, segment level:")
    print_prediction_spread(backends[0], targets, baseline.labels)
    print_prediction_spread(backends[1], targets, candidate.labels)

    if args.patient_level:
        baseline_pooled, patient_targets = pool_by_patient(segments, baseline.logits)
        candidate_pooled, _ = pool_by_patient(segments, candidate.logits)
        baseline_patient = np.argmax(baseline_pooled, axis=1)
        candidate_patient = np.argmax(candidate_pooled, axis=1)
        print_comparison(
            score("patient-level", patient_targets, baseline_patient),
            score("patient-level", patient_targets, candidate_patient),
            backends,
        )
        print("\n  predicted-class spread, patient level:")
        print_prediction_spread(backends[0], patient_targets, baseline_patient)
        print_prediction_spread(backends[1], patient_targets, candidate_patient)

    agreement = float(np.mean(baseline.labels == candidate.labels))
    matches = int(round(agreement * baseline.labels.size))
    print(
        f"\nSegment-level prediction agreement: {agreement:.1%} "
        f"({matches}/{baseline.labels.size})"
    )
    print(
        f"Latency: PyTorch {baseline.mean_latency_ms:.1f} ms vs ONNX "
        f"{candidate.mean_latency_ms:.1f} ms per segment (this host, not the "
        "Orange Pi; Table 4 is the on-device record)"
    )
    print(f"\n{verdict(baseline_segment.macro_f1, candidate_segment.macro_f1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
