"""Write segment-level test predictions to outputs/preds_targets/<exp_name>.data.csv.

Each row: logit_0, ..., logit_{C-1}, target, filename. Rows are appended batch by batch,
so call reset_file() once before a test run.
"""
import csv
import os

import numpy as np

OUTPUT_DIR = "outputs/preds_targets"


def output_path(config) -> str:
    return os.path.join(OUTPUT_DIR, f"{config.exp_name}.data.csv")


def reset_file(config) -> None:
    """Remove predictions left over from a previous run of the same experiment."""
    path = output_path(config)
    if os.path.exists(path):
        os.remove(path)


def save_to_file(preds, targets, names, config) -> None:
    preds_arr = preds.cpu().numpy()
    targets_arr = targets.cpu().numpy()
    if len(preds_arr) != len(targets_arr) or len(preds_arr) != len(names):
        raise ValueError(f"Length mismatch: {len(preds_arr)} preds, "
                         f"{len(targets_arr)} targets, {len(names)} names")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path(config), "a", newline="") as file:
        writer = csv.writer(file)
        for pred, target, name in zip(preds_arr, targets_arr, names):
            writer.writerow(list(np.append(pred, target)) + [name])
