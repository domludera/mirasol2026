"""Regenerate Supplementary Tables S1-S8.

Paper artifacts
---------------
* S1-S3  full diagnostic accuracy with Wilson 95% intervals for endpoints E1, E2
         and E3, under the deployed argmax rule, plus the two reference rules.
* S4     per-fold sensitivity and PPV for endpoint E1.
* S5     murmur-detection sensitivity by auscultatory-severity stratum.
* S6     referral burden and PPV at the cheapest operating point attaining each
         target sensitivity for E1.
* S7     the two-class murmur task, Unknown merged into Present.
* S8     the checkpoint-selection sensitivity analysis, delegated to
         :mod:`analysis.checkpoint_selection` so that exactly one generator
         exists for that table.

S1-S6 are computed by :mod:`analysis.clinical_endpoint`; S7 reads the two-class
prediction dumps directly; S8 reads ``results/checkpoint_accuracies.csv``. No
number in any of these tables is transcribed by hand.

Usage
-----
    python -m analysis.supp_tables
    python -m analysis.supp_tables --tables s1 s7 --out build/supp
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analysis._common import (
    BINARY_MODELS,
    FOLDS,
    PARAMS,
    add_circor_meta_argument,
    add_preds_dir_argument,
    assert_fold_cohort,
    d3,
    fold_path,
    latex_mean_sd,
    load_patient_level,
    print_latex_comment,
    require_all_folds,
    resolve_circor_meta,
    resolve_preds_dir,
    two_class_metrics,
)
from analysis.checkpoint_selection import (
    latex_table as checkpoint_latex_table,
)
from analysis.checkpoint_selection import (
    load_checkpoints,
    resolve_checkpoints_csv,
)
from analysis.checkpoint_selection import (
    summarise as summarise_checkpoints,
)
from analysis.clinical_endpoint import (
    ENDPOINTS,
    PRESENT,
    REFERENCE_RULES,
    ScreeningMetrics,
    endpoint_mask,
    load_all_models,
    load_reference,
    referral_cost_at_sensitivity,
    referral_mask,
    screening_metrics,
    stratum_mask,
    sweep_curve,
    wilson_ci,
)

DEPLOYED_RULE: Final[str] = "model_present"

#: Severity strata of Table S5, in printed order, with their LaTeX labels. The
#: combined "non-innocent" stratum is endpoint E3 itself and is reported in S3.
STRATA_LABELS: Final[Mapping[str, str]] = {
    "grade I/VI": "Systolic grade I/VI",
    "grade II/VI": "Systolic grade II/VI",
    "grade >=III/VI": r"Systolic grade $\geq$III/VI",
    "any diastolic murmur": "Any diastolic murmur",
    "murmur Present, outcome Abnormal": "Murmur present, status abnormal",
    "murmur Present, outcome Normal": "Murmur present, status normal",
}

#: Target sensitivities of Table S6.
SWEEP_TARGETS: Final[tuple[float, ...]] = (0.35, 0.40, 0.50, 0.60, 0.75)

TABLE_KEYS: Final[tuple[str, ...]] = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")


def ci(value: float, bounds: tuple[float, float]) -> str:
    """``.235 [.199, .276]``, or ``---`` when the proportion is undefined."""
    if not np.isfinite(value):
        return "---"
    return f"{d3(value)} [{d3(bounds[0])}, {d3(bounds[1])}]"


def _accuracy_cells(metrics: ScreeningMetrics) -> list[str]:
    """The five data cells shared by Tables S1-S3."""
    return [
        ci(metrics.sensitivity, metrics.sensitivity_ci),
        ci(metrics.specificity, metrics.specificity_ci),
        ci(metrics.ppv, metrics.ppv_ci),
        # Referring everyone leaves no negatives, so NPV is undefined rather than
        # zero; the printed table shows a single dash for that one cell.
        ci(metrics.npv, metrics.npv_ci),
        f"{metrics.nnr:.2f}" if np.isfinite(metrics.nnr) else "---",
    ]


def endpoint_table(frames: Mapping[str, pd.DataFrame], endpoint: str) -> str:
    """Table S1, S2 or S3 for one endpoint."""
    lines = []
    for name, preds in frames.items():
        metrics = screening_metrics(
            referral_mask(preds, DEPLOYED_RULE), endpoint_mask(preds, endpoint)
        )
        lines.append(
            " & ".join([f"{name} ({PARAMS[name]}M)"] + _accuracy_cells(metrics)) + r" \\"
        )
    lines.append(r"\midrule")
    any_model = next(iter(frames.values()))
    for rule, label in REFERENCE_RULES.items():
        metrics = screening_metrics(
            referral_mask(any_model, rule), endpoint_mask(any_model, endpoint)
        )
        lines.append(
            " & ".join([rf"\textit{{{label}}}"] + _accuracy_cells(metrics)) + r" \\"
        )
    return "\n".join(lines)


def per_fold_table(frames: Mapping[str, pd.DataFrame]) -> str:
    """Table S4: per-fold sensitivity then per-fold PPV for endpoint E1."""
    lines = []
    for name, preds in frames.items():
        sensitivity, ppv = [], []
        for fold in FOLDS:
            subset = preds[preds["fold"] == fold]
            metrics = screening_metrics(
                referral_mask(subset, DEPLOYED_RULE),
                endpoint_mask(subset, "abnormal_cardiac_status"),
            )
            sensitivity.append(metrics.sensitivity)
            ppv.append(metrics.ppv)
        cells = [name]
        cells += [d3(v) for v in sensitivity] + [latex_mean_sd(sensitivity)]
        cells += [d3(v) for v in ppv] + [latex_mean_sd(ppv)]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def severity_table(frames: Mapping[str, pd.DataFrame]) -> str:
    """Table S5: detection sensitivity by stratum, with Wilson intervals."""
    lines = []
    for key, label in STRATA_LABELS.items():
        cells = [label]
        for index, preds in enumerate(frames.values()):
            subset = preds[stratum_mask(preds, key)]
            total = len(subset)
            detected = int((subset["murmur_pred"] == PRESENT).sum())
            if index == 0:
                # Stratum sizes are identical across models; print them once.
                cells.append(str(total))
            sensitivity = detected / total if total else float("nan")
            cells.append(ci(sensitivity, wilson_ci(detected, total)))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def sweep_table(frames: Mapping[str, pd.DataFrame]) -> str:
    """Table S6: burden / PPV at the cheapest point reaching each target."""
    lines = []
    for name, preds in frames.items():
        curve = sweep_curve(preds, "abnormal_cardiac_status")
        cells = [name]
        for target in SWEEP_TARGETS:
            point = referral_cost_at_sensitivity(curve, target)
            cells.append(
                f"{100 * point['referral_rate']:.1f}\\,/\\,{d3(point['ppv'])}"
                if point
                else "---"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"% cells are referral burden (\%) / PPV at that operating point")
    return "\n".join(lines)


def two_class_table(preds_dir: Path) -> str:
    """Table S7: the two-class murmur task, Unknown merged into Present.

    The two-class dumps carry two logit columns; class 0 is murmur (Present or
    Unknown) and class 1 is Absent, so sensitivity and precision refer to class 0.
    """
    prefixes = {name: f"{key}_binary" for name, key in BINARY_MODELS.items()}
    require_all_folds(preds_dir, prefixes)
    lines = []
    for name, prefix in prefixes.items():
        rows, fold_sizes = [], []
        for fold in FOLDS:
            patient = load_patient_level(fold_path(preds_dir, prefix, fold))
            rows.append(two_class_metrics(patient))
            fold_sizes.append(len(patient))
        assert_fold_cohort(f"{name} (two-class)", fold_sizes)
        frame = pd.DataFrame(rows)
        cells = [name] + [
            latex_mean_sd(frame[key].tolist())
            for key in ("Accuracy", "Macro F1", "Sensitivity", "Precision", "Specificity")
        ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def checkpoint_table(checkpoints_csv: Path) -> str:
    """Table S8, produced by :mod:`analysis.checkpoint_selection`."""
    frame = load_checkpoints(checkpoints_csv)
    return checkpoint_latex_table(
        summarise_checkpoints(frame, selection="pooled", duplicates="count")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.supp_tables",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preds_dir_argument(parser)
    add_circor_meta_argument(parser)
    parser.add_argument(
        "--checkpoints-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help="checkpoint inventory for S8 (default: results/checkpoint_accuracies.csv)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=TABLE_KEYS,
        default=list(TABLE_KEYS),
        metavar="KEY",
        help=f"which tables to emit, from {list(TABLE_KEYS)}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="also write each table body to DIR/<key>.tex",
    )
    return parser


def _needs_predictions(tables: Sequence[str]) -> bool:
    return any(key in tables for key in ("s1", "s2", "s3", "s4", "s5", "s6"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = list(args.tables)
    preds_dir = resolve_preds_dir(args.preds_dir)

    bodies: dict[str, tuple[str, str]] = {}
    if _needs_predictions(tables):
        reference = load_reference(resolve_circor_meta(args.circor_meta))
        frames = load_all_models(preds_dir, reference)
        endpoint_keys = list(ENDPOINTS)
        for index, key in enumerate(("s1", "s2", "s3")):
            if key in tables:
                endpoint = endpoint_keys[index]
                bodies[key] = (
                    f"TABLE {key.upper()} -- {ENDPOINTS[endpoint]}",
                    endpoint_table(frames, endpoint),
                )
        if "s4" in tables:
            bodies["s4"] = (
                "TABLE S4 -- per-fold sensitivity and PPV, endpoint E1",
                per_fold_table(frames),
            )
        if "s5" in tables:
            bodies["s5"] = (
                "TABLE S5 -- severity-stratified detection sensitivity, Wilson 95% CI",
                severity_table(frames),
            )
        if "s6" in tables:
            bodies["s6"] = (
                "TABLE S6 -- referral burden to reach a target sensitivity (E1)",
                sweep_table(frames),
            )
    if "s7" in tables:
        bodies["s7"] = (
            "TABLE S7 -- two-class murmur task (Unknown merged into Present)",
            two_class_table(preds_dir),
        )
    if "s8" in tables:
        bodies["s8"] = (
            "TABLE S8 -- checkpoint-selection sensitivity analysis",
            checkpoint_table(resolve_checkpoints_csv(args.checkpoints_csv)),
        )

    for key in TABLE_KEYS:
        if key not in bodies:
            continue
        title, body = bodies[key]
        print_latex_comment(title)
        print(body)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for key, (_, body) in bodies.items():
            (args.out / f"{key}.tex").write_text(body + "\n")
        print(f"\nwrote {len(bodies)} table body file(s) to {args.out.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
