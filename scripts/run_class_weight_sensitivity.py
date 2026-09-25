"""
Class weight sensitivity analysis on TinyViT (3-class, fold 0 only).
Tests weight configurations against the [4, 10, 1] baseline
(baseline results come from run_5fold.py: tinyvit_3class_fold0).

Usage:
    python scripts/run_class_weight_sensitivity.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import TASKS, predictions_exist, train_and_test

FOLD = 0

WEIGHT_CONFIGS = [
    ([2.0, 5.0, 1.0], "cw_2_5_1"),
    ([6.0, 15.0, 1.0], "cw_6_15_1"),
]


def main():
    t0 = time.time()
    for weights, suffix in WEIGHT_CONFIGS:
        exp_name = f"tinyvit_{suffix}_fold{FOLD}"
        if predictions_exist(exp_name):
            print(f"  SKIP: {exp_name}")
            continue

        print(f"\n{'=' * 50}\n{exp_name} weights={weights}\n{'=' * 50}")
        overrides = {**TASKS["3class"], "loss_weight": weights}
        status = train_and_test(exp_name, model_type="tinyvit", esc_fold=FOLD, **overrides)
        print(f"  {exp_name}: {status}")

    print(f"\nTotal: {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
