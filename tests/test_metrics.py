"""Tests for every metric behind a published number.

Covered:

* ``mirasol.metrics.project_challenge_score`` - the partial-credit score with a
  diagonal-weight normaliser that produced Table 1, Table S7 and Table S10;
* ``mirasol.metrics.official_physionet_weighted_accuracy`` - the PhysioNet 2022
  murmur weighted accuracy (``w = 5, 3, 1``) reported alongside it
  (``docs/errata.md``);
* ``mirasol.metrics.pool_by_patient`` - patient-level mean-logit pooling, the
  decision rule behind Table 1, Table 2 and Tables S1-S8;
* ``mirasol.metrics.wilson_interval`` and ``analysis.clinical_endpoint.wilson_ci``
  against the intervals printed in Tables S1-S3 and S5;
* ``analysis.clinical_endpoint.auroc`` against scikit-learn, ties included;
* ``analysis.clinical_endpoint.screening_metrics`` against a hand-built
  confusion matrix;
* ``mirasol.metrics.load_predictions``, which must reject the duplicated rows
  the original append-mode writer produced.

Expected values are computed by hand in the test body or transcribed from the
camera-ready supplement, never taken from the code under test.  Nothing here
needs CirCor audio, a GPU or an ONNX file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import (
    ABSENT,
    PRESENT,
    UNKNOWN,
    prediction_csv,
    reference_patient_pool,
)

from analysis.clinical_endpoint import auroc, screening_metrics, wilson_ci
from mirasol.metrics import (
    BINARY_CLASS_NAMES,
    CLASS_NAMES,
    abnormality_score,
    accuracy,
    compute_metrics,
    confusion_matrix,
    evaluate_prediction_file,
    load_predictions,
    macro_f1,
    mean_std,
    official_physionet_weighted_accuracy,
    patient_id_from_filename,
    per_class_metrics,
    pool_by_patient,
    project_challenge_score,
    softmax,
    wilson_interval,
)

# A hand-built patient-level confusion matrix, rows = truth, cols = prediction.
#   support        = (6, 6, 25)   predicted = (7, 6, 24)   TP = (4, 2, 20)
HAND_CONFUSION = np.array([[4, 1, 1], [1, 2, 3], [2, 3, 20]], dtype=float)


def expand_confusion(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand a confusion matrix into the label vectors that produce it."""
    truth: list[int] = []
    predicted: list[int] = []
    for true_class in range(matrix.shape[0]):
        for pred_class in range(matrix.shape[1]):
            count = int(matrix[true_class, pred_class])
            truth.extend([true_class] * count)
            predicted.extend([pred_class] * count)
    return np.asarray(truth), np.asarray(predicted)


@pytest.fixture(scope="module")
def hand_labels() -> tuple[np.ndarray, np.ndarray]:
    return expand_confusion(HAND_CONFUSION)


# --------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------
def test_confusion_matrix_round_trips_the_hand_built_matrix(
    hand_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    truth, predicted = hand_labels
    assert np.array_equal(confusion_matrix(truth, predicted), HAND_CONFUSION)


def test_confusion_matrix_rejects_an_out_of_range_label() -> None:
    """The original implementations discarded such rows and scored the rest.

    A 2-class dump scored against the 3-class weights then produced a plausible
    number instead of an error, which is how the two label conventions could be
    mixed unnoticed.
    """
    with pytest.raises(ValueError):
        confusion_matrix([0, 1, 2, 3], [0, 1, 2, 2])
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0, -1])


def test_confusion_matrix_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        confusion_matrix([0, 1, 2], [0, 1])


# --------------------------------------------------------------------------
# The two Challenge Score variants
# --------------------------------------------------------------------------
def test_project_challenge_score_on_a_hand_computed_matrix(
    hand_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Arithmetic written out, so the normaliser cannot drift unnoticed."""
    truth, predicted = hand_labels
    # numerator:   4*5 + 1*3 + 1*0 + 1*1 + 2*3 + 3*1 + 2*0 + 3*1 + 20*5 = 136
    # denominator: 5*6 (Present) + 3*6 (Unknown) + 5*25 (Absent)        = 173
    assert project_challenge_score(truth, predicted) == pytest.approx(136.0 / 173.0)


def test_official_weighted_accuracy_on_the_same_matrix(
    hand_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Weighted mean of per-class recalls: no credit for a near miss."""
    truth, predicted = hand_labels
    # numerator:   5*4 + 3*2 + 1*20 = 46
    # denominator: 5*6 + 3*6 + 1*25 = 73
    assert official_physionet_weighted_accuracy(truth, predicted) == pytest.approx(
        46.0 / 73.0
    )


def test_the_variants_differ_exactly_by_partial_credit() -> None:
    """Present mistaken for Unknown earns 3 of 5 under the paper's metric."""
    assert project_challenge_score([PRESENT], [UNKNOWN]) == pytest.approx(0.6)
    assert official_physionet_weighted_accuracy([PRESENT], [UNKNOWN]) == 0.0
    # Present mistaken for Absent earns nothing under either.
    assert project_challenge_score([PRESENT], [ABSENT]) == 0.0


def test_both_variants_are_one_for_a_perfect_classifier() -> None:
    truth = [PRESENT] * 7 + [UNKNOWN] * 3 + [ABSENT] * 11
    assert project_challenge_score(truth, truth) == pytest.approx(1.0)
    assert official_physionet_weighted_accuracy(truth, truth) == pytest.approx(1.0)


def test_empty_input_scores_zero_rather_than_dividing_by_zero() -> None:
    assert project_challenge_score([], []) == 0.0
    assert official_physionet_weighted_accuracy([], []) == 0.0


def test_all_absent_classifier_reproduces_the_published_int8_score(
    preds_dir,
) -> None:
    """0.784 is the score of a constant Absent classifier, not of a model.

    Four of the five static-INT8 graphs are degenerate; two predict Absent for every segment and
    still score 0.784 under the paper's metric (Sec. 3.5,
    ``docs/quantization.md``).  Fold 0's patient-level label distribution is the
    whole explanation, and the official metric does not hide the collapse.
    """
    del preds_dir  # the fixture only guards the directory's presence
    pooled = reference_patient_pool(prediction_csv("efficientnet_b0_3class", 0))
    truth = pooled["target"].to_numpy(dtype=int)
    counts = tuple(int(np.sum(truth == label)) for label in (PRESENT, UNKNOWN, ABSENT))
    assert counts == (34, 14, 141)
    assert truth.size == 189

    constant = np.full_like(truth, ABSENT)
    assert project_challenge_score(truth, constant) == pytest.approx(0.784, abs=5e-4)
    assert official_physionet_weighted_accuracy(truth, constant) == pytest.approx(
        0.399, abs=5e-4
    )


# --------------------------------------------------------------------------
# Accuracy, per-class metrics, dispersion
# --------------------------------------------------------------------------
def test_accuracy_and_per_class_metrics_against_the_hand_built_matrix(
    hand_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    truth, predicted = hand_labels
    assert accuracy(truth, predicted) == pytest.approx(26.0 / 37.0)

    per_class = per_class_metrics(truth, predicted)
    assert tuple(entry.name for entry in per_class) == CLASS_NAMES
    assert tuple(entry.support for entry in per_class) == (6, 6, 25)
    assert per_class[PRESENT].precision == pytest.approx(4 / 7)
    assert per_class[PRESENT].recall == pytest.approx(4 / 6)
    # F1 = 2*TP / (predicted + support)
    assert per_class[PRESENT].f1 == pytest.approx(8 / 13)
    assert per_class[UNKNOWN].f1 == pytest.approx(4 / 12)
    assert per_class[ABSENT].f1 == pytest.approx(40 / 49)
    assert macro_f1(truth, predicted) == pytest.approx(
        (8 / 13 + 4 / 12 + 40 / 49) / 3
    )


def test_a_class_that_is_never_predicted_still_appears() -> None:
    """A degenerate all-Absent model must not shrink the macro average's base."""
    truth = [PRESENT, UNKNOWN, ABSENT]
    per_class = per_class_metrics(truth, [ABSENT, ABSENT, ABSENT])
    assert len(per_class) == 3
    assert per_class[PRESENT].precision == 0.0
    assert per_class[PRESENT].f1 == 0.0
    assert macro_f1(truth, [ABSENT, ABSENT, ABSENT]) == pytest.approx(
        (0.0 + 0.0 + 2 * 1 / (3 + 1)) / 3
    )


def test_mean_std_uses_the_population_standard_deviation() -> None:
    """ddof=1 would inflate every published five-fold s.d. by about 12%."""
    mean, sd = mean_std([0.10, 0.20])
    assert mean == pytest.approx(0.15)
    assert sd == pytest.approx(0.05)  # ddof=1 would give 0.0707
    with pytest.raises(ValueError):
        mean_std([])


def test_compute_metrics_records_the_level_and_guards_the_binary_case() -> None:
    truth = [PRESENT] * 3 + [UNKNOWN] + [ABSENT] * 6
    metrics = compute_metrics(truth, truth, level="segment")
    assert metrics.level == "segment"
    assert metrics.n == 10
    assert metrics.as_dict()["Challenge Score (project)"] == pytest.approx(1.0)

    binary = compute_metrics([0, 1, 1], [0, 1, 0], class_names=BINARY_CLASS_NAMES)
    assert math.isnan(binary.project_challenge_score)
    assert math.isnan(binary.official_physionet_weighted_accuracy)
    with pytest.raises(ValueError):
        compute_metrics(truth, truth, level="recording")


# --------------------------------------------------------------------------
# Patient-level mean-logit pooling
# --------------------------------------------------------------------------
def test_pooling_is_mean_of_logits_not_of_softmax_or_of_votes() -> None:
    """Pin the exact rule stated in the supplement's reproducibility notes.

    Patient ``A`` is built so the three plausible rules disagree: mean-logit
    gives Present, mean-softmax and segment majority vote both give Absent.
    HTS-AT's published Challenge Score (.875) is only reproducible under
    mean-logit pooling; the alternatives give .869 and .846.
    """
    logits = np.array(
        [
            [10.0, 0.0, 0.0],
            [-1.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        ]
    )
    patients = ["A", "A", "A", "B"]
    targets = [PRESENT, PRESENT, PRESENT, UNKNOWN]

    # The fixture really is discriminating.
    assert int(np.argmax(logits[:3].mean(axis=0))) == PRESENT
    assert int(np.argmax(softmax(logits[:3]).mean(axis=0))) == ABSENT
    assert [int(np.argmax(row)) for row in logits[:3]].count(ABSENT) == 2

    pooled = pool_by_patient(logits, patients, targets)
    assert pooled.patient_ids == ("A", "B")
    assert pooled.segments_per_patient == (3, 1)
    assert pooled.y_pred.tolist() == [PRESENT, UNKNOWN]
    assert pooled.y_true.tolist() == [PRESENT, UNKNOWN]
    assert pooled.pooled_logits[0].tolist() == pytest.approx([8 / 3, 0.0, 4 / 3])


def test_pooling_orders_patients_as_strings() -> None:
    """"1001" < "999": the published dumps encode string ordering."""
    logits = np.zeros((3, 3))
    pooled = pool_by_patient(logits, ["999", "1001", "50043"], [ABSENT] * 3)
    assert pooled.patient_ids == ("1001", "50043", "999")


def test_pooling_rejects_inconsistent_targets_for_one_patient() -> None:
    """Two different labels for one patient means the dump was appended to."""
    with pytest.raises(ValueError, match="inconsistent"):
        pool_by_patient(np.zeros((2, 3)), ["A", "A"], [PRESENT, ABSENT])


def test_pooling_matches_the_reference_rule_on_a_released_fold(preds_dir) -> None:
    """Package pooling and the pandas reference agree on all 189 patients."""
    del preds_dir
    path = prediction_csv("tinyvit_3class", 0)
    reference = reference_patient_pool(path)
    table = load_predictions(path)
    pooled = table.pool()

    assert list(pooled.patient_ids) == reference["patient"].tolist()
    assert pooled.y_pred.tolist() == reference["pred"].tolist()
    assert pooled.y_true.tolist() == reference["target"].tolist()
    assert len(pooled) == 189
    assert sum(pooled.segments_per_patient) == 739


def test_abnormality_score_is_one_minus_the_absent_probability() -> None:
    """The screening score of Table S6 and Fig. S1(a)."""
    pooled = np.array([[2.0, -1.0, 0.5], [-3.0, -2.0, 4.0]])
    probabilities = softmax(pooled)
    assert abnormality_score(pooled) == pytest.approx(1.0 - probabilities[:, ABSENT])


def test_patient_id_from_filename_handles_the_shipped_naming() -> None:
    assert patient_id_from_filename("85026_AV.wav") == "85026"
    assert patient_id_from_filename("85026_AV-3.wav") == "85026"
    with pytest.raises(ValueError):
        patient_id_from_filename("_AV.wav")


# --------------------------------------------------------------------------
# Reading the shipped dumps
# --------------------------------------------------------------------------
def test_load_predictions_rejects_a_duplicated_segment_row(tmp_path) -> None:
    """The append-mode writer doubled every row of a re-evaluated fold.

    Table 1, Table 2 and Tables S1-S8 all aggregate these files, so a silent
    duplicate would change published numbers without changing any code.
    """
    path = tmp_path / "duplicated.data.csv"
    path.write_text(
        "1.0,0.0,-1.0,0.0,85026_AV.wav\n"
        "0.5,0.0,-1.0,0.0,85026_MV.wav\n"
        "1.0,0.0,-1.0,0.0,85026_AV.wav\n"
    )
    with pytest.raises(ValueError, match="duplicated"):
        load_predictions(path)


def test_load_predictions_rejects_a_ragged_file(tmp_path) -> None:
    path = tmp_path / "ragged.data.csv"
    path.write_text("1.0,0.0,-1.0,0.0,85026_AV.wav\n0.5,0.0,0.0,49669_MV.wav\n")
    with pytest.raises(ValueError):
        load_predictions(path)


def test_load_predictions_infers_the_two_class_schema(preds_dir) -> None:
    """Table S7's dumps carry two logits, not three."""
    del preds_dir
    table = load_predictions(prediction_csv("tinyvit_binary", 0))
    assert table.num_classes == 2
    assert table.class_names == BINARY_CLASS_NAMES
    assert table.num_segments == 739
    assert table.num_patients == 189


def test_segment_and_patient_level_scoring_differ(preds_dir) -> None:
    """Pooling is what makes the reported numbers patient-level."""
    del preds_dir
    path = prediction_csv("htsat_3class", 0)
    patient, table = evaluate_prediction_file(path, level="patient")
    segment, _ = evaluate_prediction_file(path, level="segment")
    assert patient.level == "patient"
    assert segment.level == "segment"
    assert patient.n == 189
    assert segment.n == table.num_segments == 739
    assert patient.project_challenge_score != segment.project_challenge_score


# --------------------------------------------------------------------------
# Wilson score intervals
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("successes", "total", "low", "high", "source"),
    [
        (46, 46, 0.923, 1.000, "S5 TinyViT, systolic grade >=III/VI"),
        (45, 46, 0.887, 0.996, "S5 EfficientNet-B0, grade >=III/VI"),
        (44, 46, 0.855, 0.988, "S5 MobileNetV3-S, grade >=III/VI"),
        (2, 5, 0.118, 0.769, "S5 MobileNetV3-S, any diastolic murmur"),
        (4, 5, 0.376, 0.964, "S5 TinyViT, any diastolic murmur"),
        (101, 150, 0.595, 0.743, "S5 murmur present & status abnormal"),
        (7, 29, 0.122, 0.421, "S5 murmur present & status normal"),
        (455, 455, 0.992, 1.000, "S1 refer-all sensitivity, 455 positives"),
        (0, 486, 0.000, 0.008, "S1 refer-all specificity, 486 negatives"),
        (455, 941, 0.452, 0.515, "S1 refer-all PPV, 941 screened"),
        (150, 150, 0.975, 1.000, "S2 refer-all sensitivity, 150 positives"),
        (50, 50, 0.929, 1.000, "S3 refer-all sensitivity, 50 positives"),
        (0, 891, 0.000, 0.004, "S3 refer-all specificity, 891 negatives"),
    ],
)
def test_wilson_interval_matches_the_published_intervals(
    successes: int, total: int, low: float, high: float, source: str
) -> None:
    """Wilson 95% intervals to three decimals, against the camera-ready."""
    interval = wilson_interval(successes, total)
    assert round(interval.low, 3) == low, source
    assert round(interval.high, 3) == high, source


@pytest.mark.parametrize(("successes", "total"), [(46, 46), (2, 5), (101, 150), (0, 486)])
def test_the_two_wilson_implementations_agree(successes: int, total: int) -> None:
    """``analysis`` keeps a tuple-returning copy; it must not drift."""
    interval = wilson_interval(successes, total)
    assert wilson_ci(successes, total) == pytest.approx((interval.low, interval.high))


def test_wilson_interval_brackets_the_point_estimate() -> None:
    for total in (1, 5, 29, 46, 150, 941):
        for successes in sorted({0, 1, total // 2, total}):
            interval = wilson_interval(successes, total)
            proportion = successes / total
            assert 0.0 <= interval.low <= proportion + 1e-9
            assert proportion - 1e-9 <= interval.high <= 1.0


def test_wilson_interval_of_an_empty_stratum_is_undefined() -> None:
    interval = wilson_interval(0, 0)
    assert math.isnan(interval.low) and math.isnan(interval.high)
    assert all(math.isnan(bound) for bound in wilson_ci(0, 0))


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


# --------------------------------------------------------------------------
# AUROC
# --------------------------------------------------------------------------
def test_auroc_matches_sklearn_with_many_ties() -> None:
    """Ties are the normal case for a pooled three-class softmax sum."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    scores = rng.integers(0, 5, size=400).astype(float)
    positive = rng.random(400) < 0.3
    assert auroc(scores, positive) == pytest.approx(
        sklearn_metrics.roc_auc_score(positive, scores)
    )


def test_auroc_is_one_for_a_perfectly_separating_score() -> None:
    assert auroc(
        np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])
    ) == pytest.approx(1.0)


def test_auroc_of_a_single_class_is_undefined() -> None:
    assert math.isnan(auroc(np.array([0.1, 0.2, 0.3]), np.array([True, True, True])))


# --------------------------------------------------------------------------
# Screening operating characteristics
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hand_screening() -> tuple[np.ndarray, np.ndarray]:
    """3 TP, 2 FP, 4 FN, 11 TN, laid out explicitly."""
    referred = np.array([True] * 5 + [False] * 15)
    positive = np.array([True] * 3 + [False] * 2 + [True] * 4 + [False] * 11)
    return referred, positive


def test_screening_metrics_against_a_hand_built_confusion_matrix(
    hand_screening: tuple[np.ndarray, np.ndarray],
) -> None:
    metrics = screening_metrics(*hand_screening)
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (3, 2, 4, 11)
    assert metrics.n == 20
    assert metrics.sensitivity == pytest.approx(3 / 7)
    assert metrics.specificity == pytest.approx(11 / 13)
    assert metrics.ppv == pytest.approx(3 / 5)
    assert metrics.npv == pytest.approx(11 / 15)
    assert metrics.referral_rate == pytest.approx(5 / 20)
    assert metrics.nnr == pytest.approx(5 / 3)


def test_screening_intervals_come_from_the_matching_denominators(
    hand_screening: tuple[np.ndarray, np.ndarray],
) -> None:
    """Sensitivity is over positives, specificity over negatives (Tables S1-S3)."""
    metrics = screening_metrics(*hand_screening)
    assert metrics.sensitivity_ci == pytest.approx(wilson_ci(3, 7))
    assert metrics.specificity_ci == pytest.approx(wilson_ci(11, 13))
    assert metrics.ppv_ci == pytest.approx(wilson_ci(3, 5))
    assert metrics.npv_ci == pytest.approx(wilson_ci(11, 15))


def test_screening_metrics_per_thousand_scaling(
    hand_screening: tuple[np.ndarray, np.ndarray],
) -> None:
    per_thousand = screening_metrics(*hand_screening).per_thousand()
    assert per_thousand["referred"] == pytest.approx(250.0)
    assert per_thousand["true_positive_referrals"] == pytest.approx(150.0)
    assert per_thousand["false_positive_referrals"] == pytest.approx(100.0)
    assert per_thousand["missed_positives"] == pytest.approx(200.0)


def test_screening_metrics_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        screening_metrics(np.array([True, False]), np.array([True]))
