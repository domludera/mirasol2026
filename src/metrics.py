"""Shared evaluation metrics for the 3-class (Present / Unknown / Absent) task."""
import numpy as np

# PhysioNet 2022 murmur scoring weights: rows = true class, cols = predicted class
CHALLENGE_WEIGHTS = np.array([
    [5, 3, 0],
    [1, 3, 1],
    [0, 1, 5],
])


def challenge_score(y_true, y_pred, weights=CHALLENGE_WEIGHTS):
    """Compute the normalised PhysioNet 2022 weighted challenge score.

    Pairs with a label outside [0, n_classes) are ignored.
    Returns 0 when there are no valid pairs.
    """
    n = weights.shape[0]
    conf = np.zeros((n, n))
    max_conf = np.zeros((n, n))
    for t, p in zip(y_true, y_pred):
        t, p = int(t), int(p)
        if 0 <= t < n and 0 <= p < n:
            conf[t, p] += 1
            max_conf[t, t] += 1
    score = np.sum(conf * weights)
    max_score = np.sum(max_conf * weights)
    return score / max_score if max_score > 0 else 0
