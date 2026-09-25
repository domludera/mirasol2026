"""Regenerate Table 2 and every number quoted in Sec. 3.2 of the paper.

Paper artifact
--------------
Table 2, "Screening operating characteristics of the deployed rule against three
clinical endpoints", plus all of the Sec. 3.2 running text: the endpoint
prevalences (455/941, 150/941, 50/941), the EfficientNet-B0 operating point
(11.7% referred at PPV 0.955, 95% CI 0.898-0.980, 117 referrals per 1,000 of
which 112 are confirmed abnormal), the E1 sensitivity range 0.231-0.288 against
the expert ceiling of 0.330, and the E3 range 0.920-0.980 at NPV >= 0.995.

Every cell is rendered straight from :mod:`analysis.clinical_endpoint`, so
nothing in the table is transcribed by hand. The emitted columns are exactly the
seven that Table 2 prints; specificity and the E3 NPV are reported in
Supplementary Tables S1-S3 instead, which ``analysis/supp_tables.py`` produces.

Usage
-----
    python -m analysis.table2_clinical
    python -m analysis.table2_clinical --circor-meta path/to/training_data.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from analysis._common import (
    add_circor_meta_argument,
    add_preds_dir_argument,
    d3,
    pct,
    print_latex_comment,
    resolve_circor_meta,
    resolve_preds_dir,
)
from analysis.clinical_endpoint import (
    ENDPOINTS,
    REFERENCE_RULES,
    ScreeningMetrics,
    abnormality_score,
    auroc,
    endpoint_mask,
    load_all_models,
    load_reference,
    referral_mask,
    screening_metrics,
)

#: One entry per Table 2 data column: (endpoint, attribute, prefer-larger).
#: ``None`` as the endpoint marks the rule-level referral-burden column, which is
#: shared across endpoints and is best when small.
COLUMNS: Final[tuple[tuple[str | None, str, bool], ...]] = (
    (None, "referral_rate", False),
    ("abnormal_cardiac_status", "sensitivity", True),
    ("abnormal_cardiac_status", "ppv", True),
    ("pathological_murmur", "sensitivity", True),
    ("pathological_murmur", "ppv", True),
    ("non_innocent_murmur", "sensitivity", True),
)

DEPLOYED_RULE: Final[str] = "model_present"


def _cell(value: float, attribute: str) -> str:
    """Render one cell: referral burden as a percentage, everything else as .xxx."""
    return pct(value) if attribute == "referral_rate" else d3(value)


def rule_metrics(preds, rule: str) -> Mapping[str | None, ScreeningMetrics]:
    """Screening metrics of one referral rule against each endpoint.

    The ``None`` key carries the endpoint-independent referral burden, computed
    against E1 purely because a reference standard is needed to form a confusion
    matrix; the burden itself is a property of the rule.
    """
    referred = referral_mask(preds, rule)
    metrics: dict[str | None, ScreeningMetrics] = {
        endpoint: screening_metrics(referred, endpoint_mask(preds, endpoint))
        for endpoint in ENDPOINTS
    }
    metrics[None] = metrics["abnormal_cardiac_status"]
    return metrics


def latex_table(frames: Mapping[str, object]) -> str:
    """The Table 2 tabular body: six model rows, then the expert reference row."""
    model_metrics = {
        name: rule_metrics(preds, DEPLOYED_RULE) for name, preds in frames.items()
    }
    best: list[float] = []
    for endpoint, attribute, prefer_larger in COLUMNS:
        values = [getattr(m[endpoint], attribute) for m in model_metrics.values()]
        best.append(max(values) if prefer_larger else min(values))

    lines: list[str] = []
    for name, metrics in model_metrics.items():
        cells = [name]
        for index, (endpoint, attribute, _) in enumerate(COLUMNS):
            value = getattr(metrics[endpoint], attribute)
            text = _cell(value, attribute)
            cells.append(rf"\textbf{{{text}}}" if value == best[index] else text)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    any_model = next(iter(frames.values()))
    expert = rule_metrics(any_model, "expert_annotation")
    cells = [rf"\textit{{{REFERENCE_RULES['expert_annotation']}}}"]
    cells += [
        _cell(getattr(expert[endpoint], attribute), attribute)
        for endpoint, attribute, _ in COLUMNS
    ]
    lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def print_prevalences(frames: Mapping[str, object], n_metadata: int) -> None:
    """The ``n=941`` note and the three endpoint prevalences of the caption."""
    any_model = next(iter(frames.values()))
    n = len(any_model)
    print(
        f"% n = {n} patients with out-of-fold predictions "
        f"(of {n_metadata} in the public partition)"
    )
    for endpoint, label in ENDPOINTS.items():
        positive = endpoint_mask(any_model, endpoint)
        total = int(positive.sum())
        print(f"% {label}: {total}/{n} ({total / n:.1%}) positive")


def print_text_numbers(frames: Mapping[str, object]) -> None:
    """Every range and interval quoted in the Sec. 3.2 running text."""
    print_latex_comment("SEC. 3.2 RUNNING-TEXT NUMBERS")
    for endpoint in ENDPOINTS:
        deployed = [
            screening_metrics(
                referral_mask(preds, DEPLOYED_RULE), endpoint_mask(preds, endpoint)
            )
            for preds in frames.values()
        ]
        areas = [
            auroc(abnormality_score(preds), endpoint_mask(preds, endpoint))
            for preds in frames.values()
        ]
        print(
            f"{endpoint:<26} "
            f"sens {min(m.sensitivity for m in deployed):.3f}"
            f"-{max(m.sensitivity for m in deployed):.3f}  "
            f"PPV {min(m.ppv for m in deployed):.3f}"
            f"-{max(m.ppv for m in deployed):.3f}  "
            f"NPV {min(m.npv for m in deployed):.3f}"
            f"-{max(m.npv for m in deployed):.3f}  "
            f"AUROC {min(areas):.3f}-{max(areas):.3f}"
        )

    e1 = [
        screening_metrics(
            referral_mask(preds, DEPLOYED_RULE),
            endpoint_mask(preds, "abnormal_cardiac_status"),
        )
        for preds in frames.values()
    ]
    print(
        f"referral rate range        {min(m.referral_rate for m in e1):.3f}"
        f"-{max(m.referral_rate for m in e1):.3f}"
    )
    print(
        f"NNR range                  {min(m.nnr for m in e1):.2f}"
        f"-{max(m.nnr for m in e1):.2f}"
    )

    headline = screening_metrics(
        referral_mask(frames["EfficientNet-B0"], DEPLOYED_RULE),
        endpoint_mask(frames["EfficientNet-B0"], "abnormal_cardiac_status"),
    )
    print("\nEfficientNet-B0 on abnormal cardiac status:")
    print(
        f"  sens {headline.sensitivity:.3f} "
        f"{tuple(round(v, 3) for v in headline.sensitivity_ci)}"
        f"  spec {headline.specificity:.3f} "
        f"{tuple(round(v, 3) for v in headline.specificity_ci)}"
    )
    print(
        f"  PPV  {headline.ppv:.3f} {tuple(round(v, 3) for v in headline.ppv_ci)}"
        f"  NPV  {headline.npv:.3f} {tuple(round(v, 3) for v in headline.npv_ci)}"
    )
    per_thousand = headline.per_thousand()
    print(
        "  per 1,000 screened: "
        + "  ".join(f"{k}={v:.0f}" for k, v in per_thousand.items())
    )

    for name in ("TinyViT", "EfficientNet-B0"):
        metrics = screening_metrics(
            referral_mask(frames[name], DEPLOYED_RULE),
            endpoint_mask(frames[name], "non_innocent_murmur"),
        )
        print(
            f"{name} non-innocent murmur: sens {metrics.sensitivity:.3f} "
            f"{tuple(round(v, 3) for v in metrics.sensitivity_ci)}  "
            f"NPV {metrics.npv:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.table2_clinical",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preds_dir_argument(parser)
    add_circor_meta_argument(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="also write the LaTeX table body to this file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preds_dir = resolve_preds_dir(args.preds_dir)
    meta_csv = resolve_circor_meta(args.circor_meta)

    reference = load_reference(meta_csv)
    frames = load_all_models(preds_dir, reference)

    print_prevalences(frames, len(reference))
    print_latex_comment("TABLE 2 -- operating characteristics, deployed argmax rule")
    body = latex_table(frames)
    print(body)
    print_text_numbers(frames)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body + "\n")
        print(f"\nwrote {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
