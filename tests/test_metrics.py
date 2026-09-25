import numpy as np
import pytest

from src.metrics import CHALLENGE_WEIGHTS, challenge_score


def test_perfect_predictions_score_one():
    y = [0, 1, 2, 2, 0]
    assert challenge_score(y, y) == pytest.approx(1.0)


def test_worst_case_present_absent_swap_scores_zero():
    # Present <-> Absent confusions carry zero weight
    assert challenge_score([0, 2], [2, 0]) == pytest.approx(0.0)


def test_matches_hand_computed_value():
    y_true = [0, 0, 1, 2]
    y_pred = [0, 1, 1, 1]
    # score = w[0,0] + w[0,1] + w[1,1] + w[2,1] = 5 + 3 + 3 + 1 = 12
    # max   = w[0,0]*2 + w[1,1] + w[2,2]     = 10 + 3 + 5    = 18
    assert challenge_score(y_true, y_pred) == pytest.approx(12 / 18)


def test_out_of_range_labels_are_ignored():
    assert challenge_score([0, 5, -1], [0, 0, 0]) == pytest.approx(1.0)


def test_empty_input_scores_zero():
    assert challenge_score([], []) == 0


def test_accepts_numpy_float_labels():
    y = np.array([0.0, 1.0, 2.0])
    assert challenge_score(y, y) == pytest.approx(1.0)


def test_weights_are_physionet_2022():
    np.testing.assert_array_equal(CHALLENGE_WEIGHTS, [[5, 3, 0], [1, 3, 1], [0, 1, 5]])
