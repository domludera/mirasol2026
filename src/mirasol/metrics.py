"""The one implementation of every metric reported in the paper.

This module replaces four divergent copies of the same code in the original
source tree and is imported by ``analysis/table1_fivefold.py`` (Table 1),
``analysis/table2_clinical.py`` and ``analysis/supp_tables.py`` (Table 2,
Tables S1-S8), ``analysis/table_s10_ablation.py`` (Table S10) and
``analysis/patient_level_int8.py`` (Sec. 3.5).

Two different "Challenge Scores"
--------------------------------
The paper's Table 1 caption describes the PhysioNet 2022 class-weighted
formulation, but the code that produced the published values computes something
else. Both are provided here, under names that cannot be mistaken for one
another, and every table script reports them side by side.

:func:`project_challenge_score` - **reproduces the published numbers.** A
partial-credit score with a diagonal-weight normaliser::

    W = [[5, 3, 0],
         [1, 3, 1],
         [0, 1, 5]]                       rows = true class, cols = prediction

    score = sum(confusion * W) / sum(diag(row_totals) * W)

Off-diagonal entries award partial credit, so confusing *Present* with
*Unknown* still earns 3 of a possible 5. The denominator is the score a perfect
classifier would obtain, i.e. ``5*N_Present + 3*N_Unknown + 5*N_Absent``.

:func:`official_physionet_weighted_accuracy` - the PhysioNet 2022 murmur
weighted accuracy, with class weights ``w = (5, 3, 1)`` for
Present/Unknown/Absent::

    score = sum_i w_i * TP_i / sum_i w_i * N_i          N_i = true count of i

which is a weighted average of per-class recalls and awards no partial credit.

The two are not monotonically related. On the shipped out-of-fold predictions
the project form spans 0.875-0.896 across the six architectures while the
official form spans 0.684-0.741, and the ranking changes (ResNet-18 is best
under the first, HTS-AT under the second). The paper's central claim - that
diagnostic performance is largely independent of architecture - survives both,
because the extreme values still overlap within one standard deviation. See
``docs/errata.md``. Note also that under the official metric these scores sit
*below*, not above, the PhysioNet hidden-test entries quoted in Table 3, which
are in any case not comparable across evaluation protocols.

Patient-level protocol
----------------------
Segment-level **logits** are mean-pooled per patient and then argmax-ed
(Sec. 2.4 of the paper, restated in the supplement). Pooling logits rather than
softmax probabilities is load-bearing: the two disagree, and part of the
reported FP32-to-INT8 drop in Sec. 3.5 came from silently switching to the
latter. :func:`pool_by_patient` is therefore the only pooling used anywhere.

Two short aliases, ``challenge_score`` and ``official_challenge_score``, are
provided for the two functions above because that is how the analysis scripts
refer to them. The long names remain the documentation anchors.

Dispersion
----------
Cross-fold summaries use the population standard deviation (``ddof=0``), which
is what produced every ``mean +- s.d.`` in the paper; see :func:`mean_std`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np

__all__ = [
    "ABSENT",
    "CLASS_NAMES",
    "Interval",
    "MetricSet",
    "NUM_CLASSES",
    "OFFICIAL_CLASS_WEIGHTS",
    "PRESENT",
    "PROJECT_WEIGHT_MATRIX",
    "PatientPredictions",
    "PerClassMetrics",
    "PredictionTable",
    "UNKNOWN",
    "Z_95",
    "abnormality_score",
    "accuracy",
    "challenge_score",
    "compute_metrics",
    "confusion_matrix",
    "evaluate_prediction_file",
    "load_predictions",
    "macro_f1",
    "mean_std",
    "official_challenge_score",
    "official_physionet_weighted_accuracy",
    "patient_id_from_filename",
    "per_class_metrics",
    "pool_by_patient",
    "project_challenge_score",
    "softmax",
    "wilson_interval",
]

#: Label order produced by dataset preparation and assumed by every exported
#: graph. Fixed by ``mirasol.data.prepare``; do not reorder.
CLASS_NAMES: Final[tuple[str, str, str]] = ("Present", "Unknown", "Absent")
NUM_CLASSES: Final[int] = 3
PRESENT: Final[int] = 0
UNKNOWN: Final[int] = 1
ABSENT: Final[int] = 2

#: Two-class configuration of Table S7: Present|Unknown vs Absent.
BINARY_CLASS_NAMES: Final[tuple[str, str]] = ("Present or Unknown", "Absent")

_PROJECT_WEIGHTS = np.array([[5.0, 3.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]])
_PROJECT_WEIGHTS.flags.writeable = False
#: Partial-credit weights of :func:`project_challenge_score`; read-only.
PROJECT_WEIGHT_MATRIX: Final[np.ndarray] = _PROJECT_WEIGHTS

_OFFICIAL_WEIGHTS = np.array([5.0, 3.0, 1.0])
_OFFICIAL_WEIGHTS.flags.writeable = False
#: PhysioNet 2022 murmur class weights for Present/Unknown/Absent; read-only.
OFFICIAL_CLASS_WEIGHTS: Final[np.ndarray] = _OFFICIAL_WEIGHTS

#: Two-sided normal quantile for a 95 % interval.
Z_95: Final[float] = 1.959963985


# ---------------------------------------------------------------------------
# Confusion matrix and the scalar scores
# ---------------------------------------------------------------------------


def _as_labels(values: Iterable[int] | np.ndarray, num_classes: int) -> np.ndarray:
    """Validate and return an integer label vector."""
    labels = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if labels.ndim != 1:
        raise ValueError(f"expected a 1-D label vector, got shape {labels.shape}")
    labels = labels.astype(np.int64, copy=False)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(
            f"labels must lie in [0, {num_classes}); got range "
            f"[{labels.min()}, {labels.max()}]. Mixing a 2-class dump with the "
            "3-class label order is the usual cause."
        )
    return labels


def confusion_matrix(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Return the ``(num_classes, num_classes)`` counts, rows true, cols predicted.

    Unlike the original implementations this raises on an out-of-range label
    instead of silently discarding the sample, which is how a 2-class dump could
    previously be scored against 3-class weights and yield a plausible number.
    """
    true = _as_labels(y_true, num_classes)
    pred = _as_labels(y_pred, num_classes)
    if true.size != pred.size:
        raise ValueError(
            f"y_true has {true.size} entries but y_pred has {pred.size}"
        )
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(matrix, (true, pred), 1.0)
    return matrix


def project_challenge_score(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
) -> float:
    """Partial-credit Challenge Score as implemented for the paper.

    **This is the value printed in Table 1, Table S7 and Table S10.** It is not
    the official PhysioNet formulation - see the module docstring and
    :func:`official_physionet_weighted_accuracy`.

    Returns 0.0 for an empty input, matching the original behaviour.
    """
    matrix = confusion_matrix(y_true, y_pred, NUM_CLASSES)
    support = matrix.sum(axis=1)
    best_possible = float((np.diag(support) * PROJECT_WEIGHT_MATRIX).sum())
    if best_possible <= 0.0:
        return 0.0
    return float((matrix * PROJECT_WEIGHT_MATRIX).sum() / best_possible)


def official_physionet_weighted_accuracy(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
) -> float:
    """PhysioNet 2022 murmur weighted accuracy, weights ``(5, 3, 1)``.

    ``sum_i w_i * TP_i / sum_i w_i * N_i`` where ``N_i`` is the number of
    patients whose *true* class is ``i``. No partial credit for near misses.
    """
    matrix = confusion_matrix(y_true, y_pred, NUM_CLASSES)
    support = matrix.sum(axis=1)
    denominator = float((OFFICIAL_CLASS_WEIGHTS * support).sum())
    if denominator <= 0.0:
        return 0.0
    numerator = float((OFFICIAL_CLASS_WEIGHTS * np.diag(matrix)).sum())
    return numerator / denominator


#: Short name for :func:`project_challenge_score`, used across the analysis
#: scripts. It is the paper's partial-credit form - the one that reproduces the
#: published tables - and is *not* the official PhysioNet formulation.
challenge_score = project_challenge_score

#: Short name for :func:`official_physionet_weighted_accuracy`.
official_challenge_score = official_physionet_weighted_accuracy


def accuracy(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    """Plain accuracy."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else 0.0


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision, recall and F1 for one class, with its support."""

    label: int
    name: str
    support: int
    precision: float
    recall: float
    f1: float


def per_class_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> tuple[PerClassMetrics, ...]:
    """Per-class precision/recall/F1, computed from the confusion matrix.

    A zero denominator yields 0.0 (``zero_division=0`` in scikit-learn terms).
    Every class in *class_names* is reported even if it never appears, so the
    macro average is over a fixed set of labels and cannot change silently when
    a degenerate model stops predicting a class.
    """
    num_classes = len(class_names)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)

    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    results = []
    for index, name in enumerate(class_names):
        precision = _ratio(true_positive[index], predicted[index])
        recall = _ratio(true_positive[index], support[index])
        f1 = _ratio(2.0 * precision * recall, precision + recall)
        results.append(
            PerClassMetrics(
                label=index,
                name=name,
                support=int(support[index]),
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return tuple(results)


def macro_f1(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> float:
    """Unweighted mean of the per-class F1 scores."""
    per_class = per_class_metrics(y_true, y_pred, class_names)
    return float(np.mean([entry.f1 for entry in per_class]))


# ---------------------------------------------------------------------------
# Interval estimates
# ---------------------------------------------------------------------------


class Interval(NamedTuple):
    """A closed confidence interval.

    A :class:`~typing.NamedTuple` so that ``low, high = wilson_interval(...)``
    and ``interval[0]`` both work - the tuple convention the clinical-endpoint
    tables were written against - while staying immutable.
    """

    low: float
    high: float

    def __str__(self) -> str:
        return f"[{self.low:.3f}, {self.high:.3f}]"


def wilson_interval(successes: int, total: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because several strata in Tables
    S5-S6 are small (n=46 high-grade murmurs, n=5 diastolic murmurs), where the
    normal interval leaves the unit range. Returns ``nan`` bounds for ``total=0``
    rather than raising, so an empty stratum can still be tabulated.
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(
            f"require 0 <= successes <= total, got successes={successes}, total={total}"
        )
    if total == 0:
        return Interval(float("nan"), float("nan"))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = (z / denominator) * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4 * total * total)
    )
    return Interval(max(0.0, centre - float(half_width)), min(1.0, centre + float(half_width)))


def mean_std(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Mean and **population** standard deviation (``ddof=0``).

    ``ddof=0`` is what produced every ``mean +- s.d.`` in the paper. Using the
    sample standard deviation would inflate the five-fold spreads by about 12 %
    and no longer match the published tables.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarise an empty sequence")
    return float(array.mean()), float(array.std(ddof=0))


# ---------------------------------------------------------------------------
# Metric bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSet:
    """Every scalar the paper reports for one evaluation, computed once."""

    n: int
    level: str
    confusion: np.ndarray
    accuracy: float
    macro_f1: float
    project_challenge_score: float
    official_physionet_weighted_accuracy: float
    per_class: tuple[PerClassMetrics, ...]

    def klass(self, label: int) -> PerClassMetrics:
        """Per-class metrics for *label*."""
        return self.per_class[label]

    def as_dict(self) -> Mapping[str, float | int | str]:
        """Flat, JSON-friendly view keyed exactly as the paper's table rows."""
        flat: dict[str, float | int | str] = {
            "n": self.n,
            "level": self.level,
            "Accuracy": self.accuracy,
            "Macro F1": self.macro_f1,
            "Challenge Score (project)": self.project_challenge_score,
            "Challenge Score (official PhysioNet)": (
                self.official_physionet_weighted_accuracy
            ),
        }
        for entry in self.per_class:
            flat[f"Precision ({entry.name})"] = entry.precision
            flat[f"Recall ({entry.name})"] = entry.recall
            flat[f"F1 ({entry.name})"] = entry.f1
            flat[f"Support ({entry.name})"] = entry.support
        return flat


def compute_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    *,
    level: str = "patient",
    class_names: Sequence[str] = CLASS_NAMES,
) -> MetricSet:
    """Compute the full metric bundle.

    Args:
        y_true: reference labels.
        y_pred: predicted labels, already pooled to the level being reported.
        level: ``"patient"`` or ``"segment"``, carried through to output so a
            segment-level number can never be printed under a patient-level
            heading (the confusion that made ``validate_quantized.py``'s
            Challenge Score row misleading).
        class_names: label order; pass :data:`BINARY_CLASS_NAMES` for Table S7.

    The two Challenge Score variants are only defined for the three-class
    configuration; for any other ``class_names`` they are reported as ``nan``.
    """
    if level not in {"patient", "segment"}:
        raise ValueError(f"level must be 'patient' or 'segment', got {level!r}")
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    matrix.flags.writeable = False
    three_class = tuple(class_names) == CLASS_NAMES
    return MetricSet(
        n=int(matrix.sum()),
        level=level,
        confusion=matrix,
        accuracy=accuracy(y_true, y_pred, len(class_names)),
        macro_f1=macro_f1(y_true, y_pred, class_names),
        project_challenge_score=(
            project_challenge_score(y_true, y_pred) if three_class else float("nan")
        ),
        official_physionet_weighted_accuracy=(
            official_physionet_weighted_accuracy(y_true, y_pred)
            if three_class
            else float("nan")
        ),
        per_class=per_class_metrics(y_true, y_pred, class_names),
    )


# ---------------------------------------------------------------------------
# Patient-level pooling
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = np.asarray(logits, dtype=np.float64) - np.max(
        logits, axis=axis, keepdims=True
    )
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def abnormality_score(pooled_logits: np.ndarray) -> np.ndarray:
    """Continuous screening score used for Table S6 and Fig. 3(a).

    Defined in the supplement as the softmax of the **pooled** logits summed
    over *Present* and *Unknown* - i.e. the probability that the patient is not
    confidently normal, which is what a referral threshold sweeps.
    """
    probabilities = softmax(np.atleast_2d(pooled_logits), axis=-1)
    if probabilities.shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"abnormality_score needs {NUM_CLASSES} logits per patient, got "
            f"{probabilities.shape[-1]}"
        )
    return probabilities[..., PRESENT] + probabilities[..., UNKNOWN]


def patient_id_from_filename(filename: str) -> str:
    """Extract the CirCor patient identifier from a segment filename.

    ``"85026_AV.wav"`` and ``"85026_AV-3.wav"`` both yield ``"85026"``.
    """
    identifier = str(filename).split("_")[0].strip()
    if not identifier:
        raise ValueError(
            f"cannot read a patient id from {filename!r}; expected "
            "'<patient>_<location>[-<segment>].wav'"
        )
    return identifier


@dataclass(frozen=True)
class PatientPredictions:
    """One prediction per patient, produced by :func:`pool_by_patient`."""

    patient_ids: tuple[str, ...]
    pooled_logits: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    segments_per_patient: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.patient_ids)

    @property
    def abnormality_scores(self) -> np.ndarray:
        """Continuous screening score per patient, in :attr:`patient_ids` order."""
        return abnormality_score(self.pooled_logits)


def pool_by_patient(
    logits: np.ndarray,
    patient_ids: Sequence[str],
    targets: Iterable[int] | np.ndarray,
) -> PatientPredictions:
    """Mean-pool segment logits per patient, then argmax - the paper's rule.

    Patients are returned in ascending order of the identifier *as a string*,
    which is the order the original pandas ``groupby`` produced, so per-patient
    dumps stay comparable with the shipped artifacts.

    Raises if a patient carries two different reference labels, which would mean
    the prediction dump had been appended to twice with inconsistent targets.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits (segments, classes), got {logits.shape}")
    labels = _as_labels(targets, logits.shape[1])
    if len(patient_ids) != logits.shape[0] or labels.size != logits.shape[0]:
        raise ValueError(
            f"length mismatch: {logits.shape[0]} logit rows, {len(patient_ids)} "
            f"patient ids, {labels.size} targets"
        )

    grouped: dict[str, list[int]] = defaultdict(list)
    for row, patient in enumerate(patient_ids):
        grouped[patient].append(row)

    ordered = sorted(grouped)
    pooled = np.empty((len(ordered), logits.shape[1]), dtype=np.float64)
    y_true = np.empty(len(ordered), dtype=np.int64)
    counts: list[int] = []
    for index, patient in enumerate(ordered):
        rows = grouped[patient]
        pooled[index] = logits[rows].mean(axis=0)
        patient_labels = set(labels[rows].tolist())
        if len(patient_labels) != 1:
            raise ValueError(
                f"patient {patient} carries inconsistent reference labels "
                f"{sorted(patient_labels)}; the prediction file is corrupt "
                "(duplicated rows from an append-mode write?)"
            )
        y_true[index] = patient_labels.pop()
        counts.append(len(rows))

    pooled.flags.writeable = False
    y_true.flags.writeable = False
    y_pred = np.argmax(pooled, axis=1).astype(np.int64)
    y_pred.flags.writeable = False
    return PatientPredictions(
        patient_ids=tuple(ordered),
        pooled_logits=pooled,
        y_true=y_true,
        y_pred=y_pred,
        segments_per_patient=tuple(counts),
    )


# ---------------------------------------------------------------------------
# Reading the shipped prediction dumps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionTable:
    """A parsed ``results/predictions/<exp>.data.csv``.

    Schema (header-less, as written during evaluation): one row per evaluated
    segment, ``<logit_0> ... <logit_{K-1}>, <target>, <filename>``. The logits
    are the network's raw outputs, not probabilities.
    """

    path: Path
    logits: np.ndarray
    targets: np.ndarray
    filenames: tuple[str, ...]
    patient_ids: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return int(self.logits.shape[1])

    @property
    def num_segments(self) -> int:
        return int(self.logits.shape[0])

    @property
    def num_patients(self) -> int:
        return len(set(self.patient_ids))

    @property
    def class_names(self) -> tuple[str, ...]:
        if self.num_classes == NUM_CLASSES:
            return CLASS_NAMES
        if self.num_classes == 2:
            return BINARY_CLASS_NAMES
        raise ValueError(f"no class names defined for {self.num_classes} classes")

    def pool(self) -> PatientPredictions:
        """Patient-level predictions under the paper's protocol."""
        return pool_by_patient(self.logits, self.patient_ids, self.targets)


def load_predictions(path: Path | str) -> PredictionTable:
    """Parse a prediction dump, inferring the class count from the row width.

    Raises on a ragged file or a duplicated ``(filename)`` row - the latter is
    exactly what the original append-mode writer produced when a fold was
    evaluated twice, and it silently doubled every patient's segment count.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise ValueError(f"{path} is empty")

    width = len(rows[0])
    if width < 3:
        raise ValueError(
            f"{path}: expected at least 3 columns (logits, target, filename), got {width}"
        )
    ragged = [index for index, row in enumerate(rows) if len(row) != width]
    if ragged:
        raise ValueError(
            f"{path}: rows {ragged[:5]} have a different column count than row 0 "
            f"({width}); the file is not a single evaluation run"
        )

    num_classes = width - 2
    logits = np.array([[float(value) for value in row[:num_classes]] for row in rows])
    targets = np.array([int(float(row[num_classes])) for row in rows], dtype=np.int64)
    filenames = tuple(row[num_classes + 1] for row in rows)

    duplicates = len(filenames) - len(set(filenames))
    if duplicates:
        raise ValueError(
            f"{path}: {duplicates} duplicated segment filename(s). A prediction "
            "dump must contain each evaluated segment exactly once; re-run the "
            "evaluation with a fresh output file."
        )

    logits.flags.writeable = False
    targets.flags.writeable = False
    return PredictionTable(
        path=path,
        logits=logits,
        targets=targets,
        filenames=filenames,
        patient_ids=tuple(patient_id_from_filename(name) for name in filenames),
    )


def evaluate_prediction_file(
    path: Path | str, *, level: str = "patient"
) -> tuple[MetricSet, PredictionTable]:
    """Load a prediction dump and score it, returning the table as well.

    ``level="patient"`` applies mean-pooling of logits then argmax;
    ``level="segment"`` scores the raw segments. The returned :class:`MetricSet`
    records which was used.
    """
    table = load_predictions(path)
    if level == "patient":
        pooled = table.pool()
        return (
            compute_metrics(
                pooled.y_true,
                pooled.y_pred,
                level="patient",
                class_names=table.class_names,
            ),
            table,
        )
    if level == "segment":
        return (
            compute_metrics(
                table.targets,
                np.argmax(table.logits, axis=1),
                level="segment",
                class_names=table.class_names,
            ),
            table,
        )
    raise ValueError(f"level must be 'patient' or 'segment', got {level!r}")
