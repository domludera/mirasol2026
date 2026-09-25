"""Calibration data and the constant-classifier guard for INT8 ONNX export.

Serves Sec. 3.5(3) of "A Low-Cost AI Stethoscope for Resource-Constrained Heart
Murmur Screening".

Two jobs, both of which the original export script did badly enough to matter:

1. **Calibration input.**  Static INT8 quantization needs representative
   activations.  These come from real 30 s segments drawn from the four
   *training* folds (never the held-out fold), turned into log-mel
   spectrograms by the one front end the models were trained with.

2. **The constant-classifier guard.**  Four of the five static-INT8 graphs
   published with the paper are constant classifiers -- EfficientNet-B0,
   TinyViT and MobileNetV3-S answer "Absent" for every input, FastViT answers
   "Unknown" -- yet the export script "smoke-tested" them on an all-zeros
   tensor, which a constant classifier passes trivially.  A screener that never
   refers anyone is a safety problem, so exporting a graph whose argmax is
   constant over a held-out batch of real segments is treated as a hard
   failure here, not a warning.

Deliberately kept separate from :mod:`mirasol.deployment.export` so that the
guard can be unit-tested and re-run against already-exported graphs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

# The published export used 50 calibration segments drawn with
# ``np.random.RandomState(42)``; both are kept as defaults so the shipped
# graphs remain reproducible.
DEFAULT_CALIBRATION_SEGMENTS: Final[int] = 50
CALIBRATION_SEED: Final[int] = 42

# 60 hold-out segments, class-stratified. 60 is the size the reported FP32-vs-
# INT8 prediction distributions were measured on, and stratification is what
# gives the guard power: on an unstratified draw from a ~78%-Absent pool a
# perfectly healthy model can answer "Absent" 20 times in a row, so a small
# unstratified hold-out would produce false alarms.
DEFAULT_HOLDOUT_SEGMENTS: Final[int] = 60
MIN_HOLDOUT_SEGMENTS: Final[int] = 20

# The 30 s clip is concatenated with itself before the STFT, exactly as the
# training/eval dataset does.
CLIP_REPEATS: Final[int] = 2

CLASS_NAMES: Final[tuple[str, str, str]] = ("Present", "Unknown", "Absent")


@dataclass(frozen=True)
class Segment:
    """One prepared 30 s segment: its name, integer target and mel input."""

    name: str
    target: int
    mel: np.ndarray  # (1, 1, time_steps, mel_bins) float32


@dataclass(frozen=True)
class CalibrationSet:
    """Calibration segments plus a disjoint hold-out used by the guard."""

    dataset_path: Path
    eval_fold: int
    seed: int
    calibration: tuple[Segment, ...]
    holdout: tuple[Segment, ...]

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self.calibration[0].mel.shape)

    def describe(self) -> str:
        distribution = Counter(segment.target for segment in self.holdout)
        counts = " / ".join(
            f"{distribution.get(index, 0)} {name}"
            for index, name in enumerate(CLASS_NAMES)
        )
        return (
            f"{len(self.calibration)} calibration + {len(self.holdout)} hold-out "
            f"segments from {self.dataset_path.name} "
            f"(training folds, eval fold {self.eval_fold} excluded, "
            f"seed {self.seed}); hold-out truth {counts}; "
            f"input shape {self.input_shape}"
        )


@dataclass(frozen=True)
class ConstantPredictionReport:
    """Outcome of running a graph over the hold-out segments."""

    counts: tuple[int, ...]
    n_segments: int

    @property
    def is_constant(self) -> bool:
        return self.n_segments > 0 and max(self.counts) == self.n_segments

    def describe(self) -> str:
        spread = " / ".join(
            f"{count} {name}"
            for count, name in zip(self.counts, CLASS_NAMES, strict=True)
        )
        return f"{spread} over {self.n_segments} held-out segments"


def _int_target(target: object) -> int:
    """Prepared datasets store the target as an int or as a one-hot vector."""
    if isinstance(target, (int, float, np.integer, np.floating)):
        return int(target)
    return int(np.argmax(np.asarray(target)))


def read_prepared_folds(dataset_path: Path) -> np.ndarray:
    """Load the prepared ``.npy`` fold array, with an actionable error.

    Raises:
        FileNotFoundError: the dataset has not been built.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"prepared dataset not found: {dataset_path}. Build it with "
            "'python -m mirasol.data.prepare --num-classes 3 ...' after "
            "downloading CirCor DigiScope from "
            "https://physionet.org/content/circor-heart-sound/"
        )
    return np.load(dataset_path, allow_pickle=True)


def segment_from_sample(sample: Mapping[str, object], fallback_name: str) -> Segment:
    """Turn one prepared sample into a :class:`Segment` model input.

    The 30 s waveform is concatenated with itself, exactly as the training and
    evaluation dataset does, then passed through the one log-mel front end the
    models were trained with.
    """
    from mirasol.signal.mel import waveform_to_mel_spectrogram

    waveform = np.asarray(sample["waveform"], dtype=np.float32)
    tiled = np.concatenate([waveform] * CLIP_REPEATS)
    mel = waveform_to_mel_spectrogram(tiled)[np.newaxis, :, :, :]
    return Segment(
        name=str(sample.get("name", fallback_name)),
        target=_int_target(sample["target"]),
        mel=np.ascontiguousarray(mel, dtype=np.float32),
    )


def load_fold_segments(
    dataset_path: Path, fold: int, limit: int | None = None
) -> tuple[Segment, ...]:
    """All segments of one fold, in stored order -- the evaluation set.

    Raises:
        ValueError: ``fold`` is out of range.
    """
    folds = read_prepared_folds(dataset_path)
    if not 0 <= fold < len(folds):
        raise ValueError(
            f"fold {fold} is outside 0..{len(folds) - 1} for {dataset_path}"
        )
    samples = list(folds[fold])
    if limit is not None:
        samples = samples[:limit]
    return tuple(
        segment_from_sample(sample, f"{dataset_path.stem}_fold{fold}_{index}")
        for index, sample in enumerate(samples)
    )


def load_calibration_set(
    dataset_path: Path,
    eval_fold: int,
    n_calibration: int = DEFAULT_CALIBRATION_SEGMENTS,
    n_holdout: int = DEFAULT_HOLDOUT_SEGMENTS,
    seed: int = CALIBRATION_SEED,
) -> CalibrationSet:
    """Draw calibration and hold-out segments from the prepared dataset.

    Both sets come from the folds that are *not* ``eval_fold``, i.e. from the
    data the model was trained on, so the guard never leaks the held-out fold
    into the export decision.

    Calibration segments are the first ``n_calibration`` entries of
    ``RandomState(seed).permutation(n)``, which is exactly what the published
    export's ``RandomState(42).choice(n, 50, replace=False)`` produced, so the
    shipped graphs stay reproducible.

    The hold-out set is drawn from the remainder of the same permutation and is
    **class-stratified**: classes are filled round-robin so the guard is not
    fooled by the ~78% Absent prior.  It is disjoint from the calibration set by
    construction.

    Args:
        dataset_path: prepared ``.npy`` written by ``mirasol.data.prepare``.
        eval_fold: the held-out fold index, excluded from both sets.
        n_calibration: calibration segments (50 in the published export).
        n_holdout: hold-out segments for the constant-classifier guard.
        seed: RNG seed for the draw.

    Raises:
        FileNotFoundError: the dataset is missing.
        ValueError: ``eval_fold`` is out of range, the counts are negative, or
            the dataset has too few training segments.
    """
    if n_calibration < 0 or n_holdout < 0:
        raise ValueError(
            f"n_calibration ({n_calibration}) and n_holdout ({n_holdout}) "
            "must not be negative"
        )
    folds = read_prepared_folds(dataset_path)
    if not 0 <= eval_fold < len(folds):
        raise ValueError(
            f"eval_fold {eval_fold} is outside 0..{len(folds) - 1} for {dataset_path}"
        )

    train_samples: list[dict] = []
    for index in range(len(folds)):
        if index != eval_fold:
            train_samples.extend(folds[index])

    wanted = n_calibration + n_holdout
    if len(train_samples) < wanted:
        raise ValueError(
            f"{dataset_path} has only {len(train_samples)} training segments; "
            f"{wanted} are needed ({n_calibration} calibration + {n_holdout} "
            "hold-out)"
        )

    permutation = np.random.RandomState(seed).permutation(len(train_samples))
    calibration_positions = [int(position) for position in permutation[:n_calibration]]
    holdout_positions = _stratified_positions(
        train_samples, [int(position) for position in permutation[n_calibration:]],
        n_holdout,
    )

    def to_segments(positions: Sequence[int]) -> tuple[Segment, ...]:
        return tuple(
            segment_from_sample(train_samples[position], f"segment_{position}")
            for position in positions
        )

    return CalibrationSet(
        dataset_path=dataset_path,
        eval_fold=eval_fold,
        seed=seed,
        calibration=to_segments(calibration_positions),
        holdout=to_segments(holdout_positions),
    )


def _stratified_positions(
    samples: Sequence[Mapping[str, object]],
    candidates: Sequence[int],
    n_wanted: int,
) -> tuple[int, ...]:
    """Pick ``n_wanted`` candidate positions, round-robin over target classes.

    ``candidates`` is already in random order, so taking the first entries of
    each per-class bucket keeps the draw random while balancing the classes.  If
    a class has too few candidates the remaining slots go to the other classes,
    so a two-class dataset or a rare "Unknown" class still yields ``n_wanted``
    segments.
    """
    buckets: dict[int, list[int]] = {}
    for position in candidates:
        buckets.setdefault(_int_target(samples[position]["target"]), []).append(position)

    chosen: list[int] = []
    classes = sorted(buckets)
    cursors = dict.fromkeys(classes, 0)
    while len(chosen) < n_wanted:
        progressed = False
        for target in classes:
            if len(chosen) >= n_wanted:
                break
            cursor = cursors[target]
            if cursor < len(buckets[target]):
                chosen.append(buckets[target][cursor])
                cursors[target] = cursor + 1
                progressed = True
        if not progressed:
            break
    return tuple(chosen)


def build_ort_reader(
    segments: Sequence[Segment], input_name: str
) -> object:
    """Wrap ``segments`` in an ONNX Runtime ``CalibrationDataReader``.

    Imported lazily because ``onnxruntime.quantization`` is only installed with
    the ``[export]`` extra.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class _SegmentReader(CalibrationDataReader):
        """Feeds pre-computed mel spectrograms to ``quantize_static``."""

        def __init__(self) -> None:
            self._segments = tuple(segments)
            self._index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self._index >= len(self._segments):
                return None
            batch = {input_name: self._segments[self._index].mel}
            self._index += 1
            return batch

        def rewind(self) -> None:
            self._index = 0

    return _SegmentReader()


def predict_segments(model_path: Path, segments: Sequence[Segment]) -> np.ndarray:
    """Argmax prediction of an ONNX graph over ``segments``.

    Raises:
        ValueError: the graph's output width is not ``len(CLASS_NAMES)``.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    dtype = np.float16 if session.get_inputs()[0].type == "tensor(float16)" else np.float32

    predictions: list[int] = []
    for segment in segments:
        outputs = session.run(
            None, {input_name: np.ascontiguousarray(segment.mel, dtype=dtype)}
        )
        logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if logits.size != len(CLASS_NAMES):
            raise ValueError(
                f"{model_path} emits {logits.size} logits, expected "
                f"{len(CLASS_NAMES)} for {CLASS_NAMES}"
            )
        predictions.append(int(np.argmax(logits)))
    return np.asarray(predictions, dtype=np.int64)


def constant_prediction_report(
    model_path: Path, segments: Sequence[Segment]
) -> ConstantPredictionReport:
    """Prediction spread of a graph over held-out real segments."""
    predictions = predict_segments(model_path, segments)
    counts = tuple(
        int(np.count_nonzero(predictions == index)) for index in range(len(CLASS_NAMES))
    )
    return ConstantPredictionReport(counts=counts, n_segments=int(predictions.size))


class DegenerateModelError(RuntimeError):
    """A quantized graph predicts one class for every held-out segment."""


def assert_not_constant(
    model_path: Path,
    segments: Sequence[Segment],
    allow_degenerate: bool = False,
    reference: ConstantPredictionReport | None = None,
) -> ConstantPredictionReport:
    """Fail the export if the quantized graph is a constant classifier.

    Args:
        model_path: the quantized ``.onnx`` to test.
        segments: class-stratified held-out real segments; at least
            ``MIN_HOLDOUT_SEGMENTS`` are required for the check to mean
            anything.
        allow_degenerate: keep the file and only warn.  This flag exists so the
            four degenerate graphs the paper reports can be regenerated as
            reproduction artifacts; it must never be used for a deployable
            build.
        reference: the FP32 graph's spread over the SAME segments.  When it is
            also constant, the collapse is not attributable to quantization and
            the message says so -- an all-Absent FP32 model is a different (and
            worse) problem.

    Returns:
        The measured prediction spread.

    Raises:
        ValueError: fewer than ``MIN_HOLDOUT_SEGMENTS`` segments were supplied.
        DegenerateModelError: the argmax was constant and ``allow_degenerate``
            is False.
    """
    if len(segments) < MIN_HOLDOUT_SEGMENTS:
        raise ValueError(
            f"the constant-classifier guard needs at least "
            f"{MIN_HOLDOUT_SEGMENTS} hold-out segments, got {len(segments)}"
        )
    report = constant_prediction_report(model_path, segments)
    if not report.is_constant:
        return report

    constant_class = CLASS_NAMES[int(np.argmax(report.counts))]
    message = (
        f"{model_path} is a CONSTANT classifier: it answers {constant_class} for "
        f"all {report.n_segments} class-stratified held-out segments "
        f"({report.describe()}). A screener that always answers the same class "
        "never refers anyone."
    )
    if reference is not None and reference.is_constant:
        message += (
            f" NOTE: the FP32 reference is also constant on this hold-out "
            f"({reference.describe()}), so the collapse is not caused by "
            "quantization -- check the checkpoint."
        )
    else:
        message += (
            " Four of the five static-INT8 graphs published with the paper fail "
            "this check. Re-export with per-channel weights (the default), or "
            "pass --allow-degenerate to keep it as a reproduction artifact only."
        )
    if allow_degenerate:
        print(f"  WARNING: {message}")
        return report
    raise DegenerateModelError(message)
