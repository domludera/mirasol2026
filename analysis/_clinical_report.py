"""Console report and JSON record for the clinical-endpoint analysis.

Serves ``analysis/clinical_endpoint.py``: this module holds only the printing and
serialisation of the five report sections, so that the library keeping the
definitions of the endpoints, the referral rules and the screening metrics stays
readable on its own.

The sections are:

1. ceiling analysis -- the expert murmur annotation and refer-all used directly as
   referral rules, bounding what any murmur detector can reach per endpoint;
2. the deployed argmax rule of each architecture against each endpoint;
3. referral burden per 1,000 patients screened, for endpoint E1;
4. murmur-detection sensitivity stratified by auscultatory severity;
5. the referral burden needed to reach a target sensitivity for E1.

The same quantities back Table 2 and Supplementary Tables S1-S6; this module is
the human-readable and JSON record of them, while ``analysis/table2_clinical.py``
and ``analysis/supp_tables.py`` emit the LaTeX bodies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from analysis._common import MODELS
from analysis.clinical_endpoint import (
    ENDPOINTS,
    PRESENT,
    REFERENCE_RULES,
    REFERRAL_RULES,
    STRATA,
    PatientReference,
    ScreeningMetrics,
    abnormality_score,
    auroc,
    endpoint_mask,
    load_all_models,
    load_reference,
    per_fold_spread,
    referral_cost_at_sensitivity,
    referral_mask,
    screening_metrics,
    stratum_mask,
    sweep_curve,
    wilson_ci,
)

RULE_WIDTH: Final[int] = 104


def _fmt(metrics: ScreeningMetrics) -> str:
    return (
        f"Sens {metrics.sensitivity:.3f} "
        f"[{metrics.sensitivity_ci[0]:.3f},{metrics.sensitivity_ci[1]:.3f}]  "
        f"Spec {metrics.specificity:.3f}  "
        f"PPV {metrics.ppv:.3f} [{metrics.ppv_ci[0]:.3f},{metrics.ppv_ci[1]:.3f}]  "
        f"NPV {metrics.npv:.3f}  Refer {metrics.referral_rate:.3f}  "
        f"NNR {metrics.nnr:5.2f}"
    )


def _banner(title: str) -> None:
    print("=" * RULE_WIDTH)
    print(title)
    print("=" * RULE_WIDTH)


def describe_reference(
    reference: Mapping[str, PatientReference], analysed: Sequence[str]
) -> dict[str, int]:
    """Prevalence of every reference standard over the analysed patients only.

    Restricted to the patients that actually have out-of-fold predictions (941 of
    the 942 in the public partition), so the printed counts are the 455/262 the
    paper reports rather than the whole-metadata 456/263.
    """
    records = [reference[p] for p in analysed]
    n = len(records)
    counts = {
        "expert outcome = Abnormal": sum(r.outcome_abnormal for r in records),
        "murmur = Present": sum(r.murmur == "Present" for r in records),
        "murmur = Unknown": sum(r.murmur == "Unknown" for r in records),
        "pathological murmur (Present & Abnormal)": sum(
            r.murmur == "Present" and r.outcome_abnormal for r in records
        ),
        "non-innocent murmur (>=III/VI or diastolic)": sum(
            r.significant_murmur for r in records
        ),
        "murmur Absent but outcome Abnormal": sum(
            r.murmur == "Absent" and r.outcome_abnormal for r in records
        ),
    }
    _banner(f"CirCor DigiScope reference standards over the {n} analysed patients")
    for label, value in counts.items():
        print(f"  {label:<46s} {value:4d}  ({value / n:6.1%})")
    return counts


def section_ceiling(any_model: pd.DataFrame) -> dict[str, object]:
    """Expert annotation and refer-all used directly as referral rules."""
    print()
    _banner(
        "1. Ceiling analysis -- expert murmur annotation as the referral rule\n"
        "   (upper bound on what any murmur detector can achieve per endpoint)"
    )
    ceiling: dict[str, object] = {}
    for endpoint, label in ENDPOINTS.items():
        print(f"  {label}")
        per_rule: dict[str, object] = {}
        for rule in REFERENCE_RULES:
            metrics = screening_metrics(
                referral_mask(any_model, rule), endpoint_mask(any_model, endpoint)
            )
            print(f"    {rule:<20s} {_fmt(metrics)}")
            per_rule[rule] = metrics.as_dict()
        ceiling[endpoint] = per_rule
    return ceiling


def section_deployed(frames: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """Each model's deployed argmax rule against each endpoint."""
    print()
    _banner("2. Deployed model as a referral rule, out-of-fold and patient level")
    per_endpoint: dict[str, object] = {}
    for endpoint, label in ENDPOINTS.items():
        print(f"\n  --- {label} ---")
        rows: dict[str, object] = {}
        for name, preds in frames.items():
            positive = endpoint_mask(preds, endpoint)
            entry: dict[str, object] = {
                "auroc": auroc(abnormality_score(preds), positive)
            }
            for rule in REFERRAL_RULES:
                metrics = screening_metrics(referral_mask(preds, rule), positive)
                entry[rule] = metrics.as_dict()
                if rule == "model_present":
                    print(f"    {name:<18s} {_fmt(metrics)}  AUROC {entry['auroc']:.3f}")
            annotated = preds.assign(
                _ref=positive, _rule=referral_mask(preds, "model_present")
            )
            sens_mu, sens_sd = per_fold_spread(annotated, "_rule", "_ref", "sensitivity")
            ppv_mu, ppv_sd = per_fold_spread(annotated, "_rule", "_ref", "ppv")
            entry["per_fold"] = {
                "sensitivity_mean": sens_mu, "sensitivity_sd": sens_sd,
                "ppv_mean": ppv_mu, "ppv_sd": ppv_sd,
            }
            rows[name] = entry
        per_endpoint[endpoint] = rows
    return per_endpoint


def section_burden(
    frames: Mapping[str, pd.DataFrame], any_model: pd.DataFrame
) -> dict[str, object]:
    """Referral burden per 1,000 patients screened, endpoint E1."""
    print()
    _banner("3. Referral burden per 1,000 screened (endpoint: abnormal cardiac status)")
    print(
        f"  {'rule':<28s}{'referred':>10s}{'true abn.':>11s}"
        f"{'false ref.':>12s}{'abn. missed':>13s}"
    )
    burden: dict[str, object] = {}
    positive = endpoint_mask(any_model, "abnormal_cardiac_status")

    def emit(label: str, metrics: ScreeningMetrics) -> None:
        counts = metrics.per_thousand()
        burden[label] = counts
        print(
            f"  {label:<28s}{counts['referred']:>10.0f}"
            f"{counts['true_positive_referrals']:>11.0f}"
            f"{counts['false_positive_referrals']:>12.0f}"
            f"{counts['missed_positives']:>13.0f}"
        )

    for rule in ("refer_all", "expert_annotation"):
        emit(rule, screening_metrics(referral_mask(any_model, rule), positive))
    for name, preds in frames.items():
        emit(
            name,
            screening_metrics(
                referral_mask(preds, "model_present"),
                endpoint_mask(preds, "abnormal_cardiac_status"),
            ),
        )
    return burden


#: Severity strata reported in Table S5 and plotted in Fig. S1(b).
def section_strata(frames: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """Murmur-detection sensitivity stratified by auscultatory severity."""
    print()
    _banner("4. Murmur-detection sensitivity stratified by auscultatory severity")
    print("  " + "stratum".ljust(34) + "".join(f"{k:>18s}" for k in frames))
    output: dict[str, object] = {}
    for label in STRATA:
        line = "  " + label.ljust(34)
        cells: dict[str, object] = {}
        for name, preds in frames.items():
            subset = preds[stratum_mask(preds, label)]
            n = len(subset)
            detected = int((subset["murmur_pred"] == PRESENT).sum())
            sensitivity = detected / n if n else float("nan")
            line += f"{detected:>6d}/{n:<4d}{sensitivity:>7.3f}"
            cells[name] = {
                "n": n,
                "detected": detected,
                "sensitivity": sensitivity,
                "ci": wilson_ci(detected, n),
            }
        print(line)
        output[label] = cells
    return output


def section_operating_points(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, object]]:
    """Referral burden required to reach a target sensitivity for E1."""
    print()
    _banner("5. Referral burden required to reach a target sensitivity for E1")
    curves: dict[str, list[dict[str, float]]] = {}
    points: dict[str, object] = {}
    for name, preds in frames.items():
        curve = sweep_curve(preds, "abnormal_cardiac_status")
        curves[name] = curve
        line = f"  {name:<18s}"
        per_target: dict[str, object] = {}
        for target in (0.40, 0.50, 0.60, 0.75):
            point = referral_cost_at_sensitivity(curve, target)
            per_target[f"{target:.2f}"] = point
            line += (
                f"  sens>={target:.2f}: refer {point['referral_rate']:.2f} "
                f"(PPV {point['ppv']:.2f})"
                if point
                else f"  sens>={target:.2f}: n/a"
            )
        print(line)
        points[name] = per_target
    return curves, points


def report(
    preds_dir: Path,
    meta_csv: Path,
    out_dir: Path,
    *,
    dump_curves: bool = False,
) -> dict[str, object]:
    """Run every section, write the JSON record and the patient-level CSVs."""
    reference = load_reference(meta_csv)
    frames = load_all_models(preds_dir, reference)
    any_model = next(iter(frames.values()))
    analysed = list(any_model["patient"])

    results: dict[str, object] = {
        "n_patients_metadata": len(reference),
        "n_patients_analysed": len(analysed),
        "reference_counts": describe_reference(reference, analysed),
        "ceiling": section_ceiling(any_model),
        "endpoints": section_deployed(frames),
        "referral_burden_per_1000": section_burden(frames, any_model),
        "severity_strata": section_strata(frames),
    }
    curves, operating_points = section_operating_points(frames)
    results["operating_points"] = operating_points

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clinical_endpoint_results.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n"
    )
    for name, preds in frames.items():
        preds.to_csv(out_dir / f"patient_level_{MODELS[name]}.csv", index=False)
    if dump_curves:
        # About 1.5 MB: one row per distinct threshold per architecture.
        payload = {name: curve for name, curve in curves.items()}
        (out_dir / "threshold_curves.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n"
        )
    print(f"\nwrote results to {out_dir.name}/")
    return results


