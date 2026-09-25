"""Regenerate Table S10: the preprocessing ablation and the class-weight sweep.

Paper artifact
--------------
Supplementary Table S10 and the Sec. 3.6 running text: the +3.1 pp accuracy /
0.316 -> 0.125 Unknown-F1 trade of dropping multi-location restructuring, the
monotonic 10 s < 15 s < 30 s context ordering, the ~12 pp Macro-F1 contribution
of the 25-400 Hz bandpass, and the class-weight comparison of [2,5,1], [4,10,1]
and [6,15,1].

Every cell is computed from the committed prediction dumps with the same
patient-level rule as Table 1 (mean-pool a patient's segment logits, then argmax)
and the same Challenge Score implementation from :mod:`mirasol.metrics`. In the
original ablation driver the baseline row was hardcoded as the literals
0.8730/0.6784/0.9073, so it could drift away from the evaluator that produced the
other rows; here the baseline is evaluated from its own prediction dump like
every other condition.

Two different fixed-fold baselines
----------------------------------
The ablation block and the class-weight block do not share a baseline run:

* the ablation baseline is the legacy single-split ``tinyvit_3class`` run, which
  is the run the ablation conditions were compared against;
* the class-weight baseline is ``tinyvit_3class_fold0``, the fold-0 run of the
  main five-fold protocol.

Both are printed, each labelled with the experiment it comes from, so the two
sets of numbers are never confused. See ``docs/errata.md``.

Usage
-----
    python -m analysis.table_s10_ablation
    python -m analysis.table_s10_ablation --baseline-exp tinyvit_3class
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from analysis._common import (
    MAX_FOLD_PATIENTS,
    MIN_FOLD_PATIENTS,
    OFFICIAL_METRIC,
    add_out_argument,
    add_preds_dir_argument,
    d3,
    load_patient_level,
    print_header,
    print_latex_comment,
    resolve_out_dir,
    resolve_preds_dir,
    three_class_metrics,
)

#: Experiment name of the run the ablation conditions were compared against.
DEFAULT_BASELINE_EXP: Final[str] = "tinyvit_3class"

#: Experiment name of the class-weight sweep's own baseline.
DEFAULT_CW_BASELINE_EXP: Final[str] = "tinyvit_3class_fold0"

#: The four columns Table S10 prints, in order.
S10_COLUMNS: Final[tuple[str, ...]] = (
    "Accuracy",
    "Macro F1",
    "Challenge Score",
    "F1 (Unknown)",
)

#: Larger is better for every S10 column, so the bold marker is always the max.
ABLATION_GROUPS: Final[tuple[tuple[str, tuple[tuple[str, str], ...]], ...]] = (
    (
        "Location restructuring",
        (("Location: single-location only", "ablation_single_location"),),
    ),
    (
        "Window length",
        (
            ("Window: 10\\,s", "ablation_window_10s"),
            ("Window: 15\\,s", "ablation_window_15s"),
        ),
    ),
    (
        "Bandpass filter",
        (
            ("Filter: none", "ablation_no_bandpass"),
            ("Filter: 25--1000\\,Hz", "ablation_wide_bandpass"),
        ),
    ),
)

#: The class-weight sweep, in the order the supplement discusses it.
CLASS_WEIGHTS: Final[tuple[tuple[str, str], ...]] = (
    ("[2, 5, 1]", "tinyvit_cw_2_5_1_fold0"),
    ("[4, 10, 1] (selected)", DEFAULT_CW_BASELINE_EXP),
    ("[6, 15, 1]", "tinyvit_cw_6_15_1_fold0"),
)

BASELINE_LABEL: Final[str] = r"\textit{Baseline (30s / BP / Multi)}"


@dataclass(frozen=True)
class Condition:
    """One evaluated experiment: its label, its dump and its patient-level metrics."""

    label: str
    experiment: str
    n_patients: int
    metrics: Mapping[str, float]


def evaluate(preds_dir: Path, label: str, experiment: str) -> Condition:
    """Evaluate one experiment's prediction dump at the patient level."""
    path = preds_dir / f"{experiment}.data.csv"
    if not path.is_file():
        raise SystemExit(
            f"error: prediction dump for '{experiment}' not found at "
            f"'{path.name}'. results/predictions/ ships with the repository; pass "
            "--preds-dir to point elsewhere."
        )
    patient = load_patient_level(path)
    if not MIN_FOLD_PATIENTS <= len(patient) <= MAX_FOLD_PATIENTS:
        raise SystemExit(
            f"error: '{experiment}' covers {len(patient)} patients; every Table S10 "
            f"condition is a single held-out fold of "
            f"{MIN_FOLD_PATIENTS}-{MAX_FOLD_PATIENTS} patients."
        )
    return Condition(
        label=label,
        experiment=experiment,
        n_patients=len(patient),
        metrics=three_class_metrics(patient),
    )


def collect(preds_dir: Path, baseline_exp: str) -> list[Condition]:
    """The baseline row followed by the five ablation conditions."""
    conditions = [evaluate(preds_dir, BASELINE_LABEL, baseline_exp)]
    for _, entries in ABLATION_GROUPS:
        conditions += [evaluate(preds_dir, label, exp) for label, exp in entries]
    return conditions


def collect_class_weights(preds_dir: Path) -> list[Condition]:
    """The three class-weight settings of the Sec. 3.6 sweep."""
    return [evaluate(preds_dir, label, exp) for label, exp in CLASS_WEIGHTS]


def print_ascii(title: str, conditions: Sequence[Condition]) -> None:
    """ASCII table including the official Challenge Score and the patient count."""
    print()
    print_header(title)
    columns = S10_COLUMNS + (OFFICIAL_METRIC,)
    labels = {OFFICIAL_METRIC: "Chall. (official)"}
    header = f"  {'Condition':<32}{'n':>5}" + "".join(
        f"{labels.get(c, c):>19}" for c in columns
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for condition in conditions:
        plain = condition.label.replace("\\textit{", "").replace("}", "")
        plain = plain.replace("\\,", " ").replace("--", "-")
        line = f"  {plain:<32}{condition.n_patients:>5}"
        line += "".join(f"{condition.metrics[c]:>19.4f}" for c in columns)
        print(line)
    print("  experiments: " + ", ".join(c.experiment for c in conditions))


def latex_table(conditions: Sequence[Condition]) -> str:
    """The Table S10 tabular body, with the group rules the supplement uses."""
    best = {
        column: max(c.metrics[column] for c in conditions) for column in S10_COLUMNS
    }
    boundaries = {1, 2, 4}  # midrules after the baseline, location and window blocks

    lines: list[str] = []
    for index, condition in enumerate(conditions):
        if index in boundaries:
            lines.append(r"\midrule")
        cells = [condition.label]
        for column in S10_COLUMNS:
            value = condition.metrics[column]
            text = f"{value:.3f}"
            cells.append(
                rf"\textbf{{{text}}}" if value == best[column] else text
            )
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def print_findings(conditions: Sequence[Condition]) -> None:
    """The three Sec. 3.6 claims, recomputed rather than restated."""
    by_experiment = {c.experiment: c for c in conditions}
    baseline = conditions[0]
    print()
    print_header("Sec. 3.6 findings, recomputed")

    single = by_experiment.get("ablation_single_location")
    if single is not None:
        delta = 100 * (single.metrics["Accuracy"] - baseline.metrics["Accuracy"])
        print(
            f"  single-location: accuracy {delta:+.1f} pp "
            f"({baseline.metrics['Accuracy']:.3f} -> {single.metrics['Accuracy']:.3f}), "
            f"Unknown F1 {baseline.metrics['F1 (Unknown)']:.3f} -> "
            f"{single.metrics['F1 (Unknown)']:.3f}"
        )

    window = [
        by_experiment.get("ablation_window_10s"),
        by_experiment.get("ablation_window_15s"),
        baseline,
    ]
    if all(w is not None for w in window):
        scores = [w.metrics["Challenge Score"] for w in window if w is not None]
        ordering = "monotonic" if scores == sorted(scores) else "NOT monotonic"
        print(
            "  context length 10s < 15s < 30s Challenge Score: "
            + " < ".join(f"{s:.3f}" for s in scores)
            + f"  ({ordering})"
        )

    no_filter = by_experiment.get("ablation_no_bandpass")
    if no_filter is not None:
        gain = 100 * (
            baseline.metrics["Macro F1"] - no_filter.metrics["Macro F1"]
        )
        print(
            f"  25-400 Hz bandpass contributes {gain:+.1f} pp Macro F1 over no filter "
            f"({no_filter.metrics['Macro F1']:.3f} -> "
            f"{baseline.metrics['Macro F1']:.3f})"
        )


def print_class_weight_findings(conditions: Sequence[Condition]) -> None:
    """The class-weight sentence of Sec. 3.6, recomputed."""
    print()
    print_header("Class-weight sweep, recomputed (TinyViT, fold 0)")
    for condition in conditions:
        print(
            f"  {condition.label:<24} Challenge Score "
            f"{condition.metrics['Challenge Score']:.3f}  "
            f"Macro F1 {condition.metrics['Macro F1']:.3f}  "
            f"Unknown F1 {condition.metrics['F1 (Unknown)']:.3f}"
        )
    best_macro = max(conditions, key=lambda c: c.metrics["Macro F1"])
    best_unknown = max(conditions, key=lambda c: c.metrics["F1 (Unknown)"])
    print(
        f"  highest Macro F1: {best_macro.label}; "
        f"highest Unknown F1: {best_unknown.label}"
    )


def class_weight_latex(conditions: Sequence[Condition]) -> str:
    """A tabular body for the class-weight block, same columns as Table S10."""
    return "\n".join(
        " & ".join([c.label] + [d3(c.metrics[column]) for column in S10_COLUMNS])
        + r" \\"
        for c in conditions
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.table_s10_ablation",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preds_dir_argument(parser)
    add_out_argument(parser, "build/table_s10")
    parser.add_argument(
        "--baseline-exp",
        default=DEFAULT_BASELINE_EXP,
        metavar="NAME",
        help="experiment whose prediction dump supplies the baseline row",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print to stdout only; do not create the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preds_dir = resolve_preds_dir(args.preds_dir)

    ablation = collect(preds_dir, args.baseline_exp)
    print_ascii("Table S10: preprocessing ablation (fixed held-out fold)", ablation)
    print_findings(ablation)

    weights = collect_class_weights(preds_dir)
    print_ascii("Class-weight sweep (TinyViT, fold 0)", weights)
    print_class_weight_findings(weights)

    print_latex_comment("TABLE S10 -- preprocessing ablation")
    ablation_body = latex_table(ablation)
    print(ablation_body)
    print(
        f"% baseline row from '{args.baseline_exp}'; the class-weight block below uses "
        f"'{DEFAULT_CW_BASELINE_EXP}', a different run"
    )

    print_latex_comment("SEC. 3.6 -- class-weight sensitivity")
    weight_body = class_weight_latex(weights)
    print(weight_body)

    if not args.no_write:
        out = resolve_out_dir(args)
        (out / "table_s10.tex").write_text(ablation_body + "\n")
        (out / "class_weights.tex").write_text(weight_body + "\n")
        print(f"\nwrote table_s10.tex and class_weights.tex to {out.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
