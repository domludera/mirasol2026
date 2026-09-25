"""Regenerate Table 1: patient-level five-fold cross-validation, all six architectures.

Paper artifact
--------------
Table 1 of "A Low-Cost AI Stethoscope for Resource-Constrained Heart Murmur
Screening" (Ludera & Kersten-Oertel, MIRASOL @ MICCAI 2026), and the Sec. 3.1
running text that quotes its Challenge Score range (0.875-0.896) and the
Table 3 Group C rows.

This is the merge of two earlier aggregation scripts, neither of which could
produce the published table on its own: one covered four architectures, the
other the remaining two, and between them two of Table 1's seven rows were never
printed -- ``F1 (Unknown)`` was computed and dropped, and ``Precision (Present)``
was not computed at all. This script emits all six architectures in Table 1
column order and all seven metric rows.

Two Challenge Score variants are reported side by side:

* ``Challenge Score`` -- the weighted-credit variant used in the published
  tables: ``sum(confusion * W) / sum(diag(confusion_total) * W)`` with
  ``W = [[5,3,0],[1,3,1],[0,1,5]]``.
* ``Challenge Score (official PhysioNet)`` -- the PhysioNet 2022 murmur weighted
  accuracy, ``sum_i w_i TP_i / sum_i w_i N_i`` with ``w = [5,3,1]``.

They rank the architectures differently (the official metric puts HTS-AT first
rather than ResNet-18) while leaving the paper's architecture-independence claim
intact, since the best and worst official scores still overlap within one
standard deviation. ``docs/errata.md`` states this, and Table 3 already warns
that scores are not comparable across evaluation protocols.

Both definitions live in :mod:`mirasol.metrics` and are not reimplemented here.

Usage
-----
    python -m analysis.table1_fivefold
    python -m analysis.table1_fivefold --preds-dir results/predictions --out build/table1
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from analysis._common import (
    FOLDS,
    METRIC_ORDER,
    MODELS,
    OFFICIAL_METRIC,
    PARAMS,
    add_out_argument,
    add_preds_dir_argument,
    ascii_mean_sd,
    assert_fold_cohort,
    confusion_counts,
    d3,
    fold_path,
    latex_mean_sd,
    load_patient_level,
    mean_sd,
    print_header,
    print_latex_comment,
    require_all_folds,
    resolve_out_dir,
    resolve_preds_dir,
    three_class_metrics,
)

#: Every row emitted, in Table 1 order, with the official variant appended.
ROWS: Final[tuple[str, ...]] = METRIC_ORDER + (OFFICIAL_METRIC,)

#: Rows whose Table 1 label is abbreviated to fit the column width.
LATEX_LABELS: Final[Mapping[str, str]] = {
    "Precision (Present)": "Precision (Pres.)",
    "Recall (Present)": "Recall (Pres.)",
    OFFICIAL_METRIC: "Challenge Score (official)",
}

#: Column headers as abbreviated in the printed table.
LATEX_HEADERS: Final[Mapping[str, str]] = {
    "MobileNetV3-S": "MobileNetV3",
    "EfficientNet-B0": "EffNet-B0",
}

#: Rows where the published table's bold marker sits on the second-best cell.
#: This script bolds the actual maximum, so those two markers move to HTS-AT.
BOLD_DIVERGENCE: Final[tuple[str, ...]] = ("F1 (Present)", "F1 (Unknown)")


def collect(preds_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Per-fold metrics for every architecture, after validating the cohort.

    Raises ``SystemExit`` if any of the thirty fold dumps is missing or if a
    model does not partition exactly 941 patients over five folds.
    """
    require_all_folds(preds_dir, MODELS)
    results: dict[str, list[dict[str, float]]] = {}
    for name, prefix in MODELS.items():
        per_fold: list[dict[str, float]] = []
        fold_sizes: list[int] = []
        for fold in FOLDS:
            patient = load_patient_level(fold_path(preds_dir, prefix, fold))
            per_fold.append(three_class_metrics(patient))
            fold_sizes.append(len(patient))
        assert_fold_cohort(name, fold_sizes)
        results[name] = per_fold
    return results


def print_per_fold(results: Mapping[str, Sequence[Mapping[str, float]]]) -> None:
    """Per-fold Challenge Score / accuracy / macro F1, one line per fold."""
    print_header("Per-fold patient-level results (3-class: Present/Unknown/Absent)")
    for name, per_fold in results.items():
        for fold, metrics in zip(FOLDS, per_fold, strict=True):
            print(
                f"  {name:<16} fold {fold}: "
                f"score={metrics['Challenge Score']:.4f}  "
                f"official={metrics[OFFICIAL_METRIC]:.4f}  "
                f"acc={metrics['Accuracy']:.4f}  "
                f"macroF1={metrics['Macro F1']:.4f}"
            )


def print_summary(results: Mapping[str, Sequence[Mapping[str, float]]]) -> None:
    """ASCII mean +/- s.d. table, one row per metric, one column per architecture."""
    print()
    print_header("Table 1: mean +/- s.d. over five folds")
    label_width = max(len(row) for row in ROWS) + 1
    header = " " * label_width + "".join(f"{name:>18}" for name in results)
    print(header)
    print(" " * label_width + "".join(f"{'(' + PARAMS[n] + 'M)':>18}" for n in results))
    print("-" * len(header))
    for row in ROWS:
        line = f"{row:<{label_width}}"
        for per_fold in results.values():
            line += f"{ascii_mean_sd([m[row] for m in per_fold]):>18}"
        print(line)


def latex_table(results: Mapping[str, Sequence[Mapping[str, float]]]) -> str:
    """The Table 1 tabular body, bolding the best cell in each row."""
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\textbf{Metric} & "
        + " & ".join(
            rf"\textbf{{{LATEX_HEADERS.get(name, name)}}}" for name in results
        )
        + r" \\",
        " & "
        + " & ".join(rf"\textbf{{({PARAMS[name]}M)}}" for name in results)
        + r" \\",
        r"\midrule",
    ]
    for row in ROWS:
        means = {name: mean_sd([m[row] for m in fm])[0] for name, fm in results.items()}
        best = max(means, key=means.__getitem__)
        cells = []
        for name, per_fold in results.items():
            values = [m[row] for m in per_fold]
            mean, sd = mean_sd(values)
            cell = (
                rf"\textbf{{{d3(mean)}}}$\pm${d3(sd)}"
                if name == best
                else latex_mean_sd(values)
            )
            cells.append(cell)
        lines.append(f"{LATEX_LABELS.get(row, row):<26} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def markdown_report(results: Mapping[str, Sequence[Mapping[str, float]]]) -> str:
    """Long-form Markdown record: per-fold values and the aggregate for each row."""
    lines = ["# Table 1 -- patient-level five-fold cross-validation", ""]
    lines += [
        "Regenerated by `python -m analysis.table1_fivefold` from the committed",
        "out-of-fold prediction dumps. Aggregation: mean-pool a patient's segment",
        "logits, then argmax. Mean +/- population s.d. over the five held-out folds.",
        "",
        "| Metric | " + " | ".join(f"{n} ({PARAMS[n]}M)" for n in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for row in ROWS:
        cells = [ascii_mean_sd([m[row] for m in fm]) for fm in results.values()]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    for name, per_fold in results.items():
        lines += ["", f"## {name}", "", "| Fold | " + " | ".join(ROWS) + " |",
                  "|---|" + "---|" * len(ROWS)]
        for fold, metrics in zip(FOLDS, per_fold, strict=True):
            lines.append(
                f"| {fold} | " + " | ".join(f"{metrics[r]:.4f}" for r in ROWS) + " |"
            )
    return "\n".join(lines) + "\n"


def json_payload(
    results: Mapping[str, Sequence[Mapping[str, float]]], preds_dir: Path
) -> dict[str, object]:
    """Machine-readable record of every cell, for the golden-table test."""
    return {
        "artifact": "Table 1 (patient-level five-fold cross-validation)",
        "preds_dir": preds_dir.name,
        "folds": list(FOLDS),
        "metric_order": list(ROWS),
        "models": {
            name: {
                "prefix": MODELS[name],
                "params_m": PARAMS[name],
                "per_fold": [dict(m) for m in per_fold],
                "mean_sd": {
                    row: dict(
                        zip(("mean", "sd"), mean_sd([m[row] for m in per_fold]), strict=True)
                    )
                    for row in ROWS
                },
            }
            for name, per_fold in results.items()
        },
    }


def print_confusion(preds_dir: Path) -> None:
    """Pooled out-of-fold confusion matrix per architecture.

    Both Challenge Score variants are functions of this matrix, so printing it
    makes the difference between them auditable without rerunning anything.
    """
    print()
    print_header("Pooled out-of-fold confusion (rows true, columns predicted)")
    for name, prefix in MODELS.items():
        total = None
        for fold in FOLDS:
            counts = confusion_counts(load_patient_level(fold_path(preds_dir, prefix, fold)))
            total = counts if total is None else total + counts
        assert total is not None  # require_all_folds already guaranteed five folds
        print(f"  {name}")
        for index, label in enumerate(("Present", "Unknown", "Absent")):
            print(f"    true {label:<8} " + "".join(f"{v:6d}" for v in total[index]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.table1_fivefold",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preds_dir_argument(parser)
    add_out_argument(parser, "build/table1")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print to stdout only; do not create the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preds_dir = resolve_preds_dir(args.preds_dir)

    results = collect(preds_dir)
    print_per_fold(results)
    print_summary(results)
    print_confusion(preds_dir)

    print_latex_comment("TABLE 1 -- paste into the manuscript")
    print(latex_table(results))
    print(
        "% note: the bold marker is placed on the largest mean in each row. In the\n"
        "% printed table the "
        + " and ".join(f"'{row}'" for row in BOLD_DIVERGENCE)
        + " markers sit on ResNet-18,\n"
        "% whose means are below HTS-AT's; the values themselves are unchanged."
    )

    if not args.no_write:
        out = resolve_out_dir(args)
        (out / "table1_fivefold.md").write_text(markdown_report(results))
        (out / "table1_fivefold.json").write_text(
            json.dumps(json_payload(results, preds_dir), indent=2) + "\n"
        )
        print(f"\nwrote table1_fivefold.md and table1_fivefold.json to {out.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
