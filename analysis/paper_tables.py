"""
Recompute the paper's result tables from the released segment-level predictions.

  Table 2  patient-level 5-fold CV for all six models (mean +/- s.d. over folds)
  Table 5  preprocessing ablation (TinyViT, fixed held-out fold)
  Sec 3.5  class-weight sensitivity (TinyViT, fold 0)

Each CSV in predictions/ has one row per segment: logit_0, logit_1, logit_2, target, filename.
Segment logits are mean-pooled per patient, then argmax'd (Sec. 2.5). Missing CSVs
(e.g. while a --preds_dir run is incomplete) are reported and skipped.

Usage:
    python analysis/paper_tables.py [--preds_dir predictions]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import challenge_score

# (experiment prefix, display name, params) in the paper's column order
MODELS = [
    ("mobilenetv3", "MobileNetV3", "1.52M"),
    ("fastvit", "FastViT", "3.26M"),
    ("efficientnet_b0", "EffNet-B0", "4.01M"),
    ("tinyvit", "TinyViT", "5.07M"),
    ("resnet18", "ResNet-18", "11.2M"),
    ("htsat", "HTS-AT", "27.6M"),
]
FOLDS = range(5)

ABLATIONS = [
    ("tinyvit_3class", "Baseline (30s / BP / Multi)"),
    ("ablation_single_location", "Location: single-location only"),
    ("ablation_window_10s", "Window: 10 s"),
    ("ablation_window_15s", "Window: 15 s"),
    ("ablation_no_bandpass", "Filter: none"),
    ("ablation_wide_bandpass", "Filter: 25-1000 Hz"),
]

CLASS_WEIGHTS = [
    ("tinyvit_cw_2_5_1_fold0", "[2, 5, 1]"),
    ("tinyvit_3class_fold0", "[4, 10, 1] (selected)"),
    ("tinyvit_cw_6_15_1_fold0", "[6, 15, 1]"),
]

TABLE2_METRICS = ["Challenge Score", "Accuracy", "Macro F1", "Precision (Pres.)",
                  "Recall (Pres.)", "F1 (Present)", "F1 (Unknown)"]
SINGLE_RUN_METRICS = ["Accuracy", "Macro F1", "Challenge Score", "F1 (Unknown)"]


def patient_metrics(csv_path: str) -> dict:
    """Mean-pool segment logits per patient, argmax, and score."""
    logit_cols = ["logit_0", "logit_1", "logit_2"]
    df = pd.read_csv(csv_path, header=None, names=[*logit_cols, "target", "filename"])
    df["patient"] = df["filename"].str.split("_").str[0]
    grouped = df.groupby("patient").agg(
        **{c: (c, "mean") for c in logit_cols}, target=("target", "first"))

    y_pred = grouped[logit_cols].to_numpy().argmax(axis=1)
    y_true = grouped["target"].astype(int).to_numpy()

    labels = [0, 1, 2]
    f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {
        "Challenge Score": challenge_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (Pres.)": prec[0],
        "Recall (Pres.)": rec[0],
        "F1 (Present)": f1[0],
        "F1 (Unknown)": f1[1],
    }


def csv_path(preds_dir: str, exp_name: str) -> str:
    return os.path.join(preds_dir, f"{exp_name}.data.csv")


def metrics_or_none(preds_dir: str, exp_name: str) -> dict | None:
    path = csv_path(preds_dir, exp_name)
    if not os.path.exists(path):
        print(f"  missing: {path}", file=sys.stderr)
        return None
    return patient_metrics(path)


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = "  ".join
    print(f"\n{title}\n{'=' * len(title)}")
    print(line(h.ljust(w) for h, w in zip(header, widths)))
    print(line("-" * w for w in widths))
    for r in rows:
        print(line(c.ljust(w) for c, w in zip(r, widths)))


def table2(preds_dir: str) -> None:
    header, cells = ["Metric"], []
    for prefix, name, params in MODELS:
        folds = [metrics_or_none(preds_dir, f"{prefix}_3class_fold{k}") for k in FOLDS]
        folds = [f for f in folds if f is not None]
        if not folds:
            continue
        n = "" if len(folds) == len(FOLDS) else f", {len(folds)}/{len(FOLDS)} folds"
        header.append(f"{name} ({params}{n})")
        # s.d. over folds (population, ddof=0)
        cells.append({m: f"{np.mean([f[m] for f in folds]):.3f}+/-{np.std([f[m] for f in folds]):.3f}"
                      for m in TABLE2_METRICS})

    rows = [[m] + [c[m] for c in cells] for m in TABLE2_METRICS]
    print_table("Table 2: patient-level 5-fold CV (mean +/- s.d.)", header, rows)


def single_run_table(title: str, first_col: str, runs: list[tuple[str, str]],
                     preds_dir: str) -> None:
    rows = []
    for exp_name, label in runs:
        m = metrics_or_none(preds_dir, exp_name)
        if m is None:
            continue
        rows.append([label] + [f"{m[k]:.3f}" for k in SINGLE_RUN_METRICS])
    print_table(title, [first_col] + SINGLE_RUN_METRICS, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preds_dir", default="predictions",
                        help="Directory of <exp>.data.csv files (training writes to outputs/preds_targets)")
    args = parser.parse_args()

    table2(args.preds_dir)
    single_run_table("Table 5: preprocessing ablation (TinyViT, fixed held-out fold)",
                     "Condition", ABLATIONS, args.preds_dir)
    single_run_table("Sec. 3.5: class-weight sensitivity (TinyViT, fold 0)",
                     "Weights [P, U, A]", CLASS_WEIGHTS, args.preds_dir)


if __name__ == "__main__":
    main()
