"""Clinical-endpoint evaluation of the murmur-screening models: the Table 2 library.

Paper artifacts
---------------
Directly: Table 2 and all of Sec. 3.2 of "A Low-Cost AI Stethoscope for
Resource-Constrained Heart Murmur Screening" (MIRASOL @ MICCAI 2026).
Indirectly: Supplementary Tables S1-S6 and Figure S1, which are emitted by
``analysis/supp_tables.py`` and ``analysis/figures.py`` on top of this module.

Why this endpoint
-----------------
A murmur is an acoustic finding, not an indication for referral, so reporting
murmur classification alone says little about whether a screening device is
clinically useful. CirCor DigiScope ships an *expert-assigned clinical outcome*
(Normal / Abnormal) per patient, recorded by the examining pediatric cardiologist
independently of the digital recordings and of the murmur annotation. The models
in this paper were trained only on the murmur annotation, so scoring them against
the clinical outcome is an unbiased held-out endpoint evaluation: no model ever
saw an outcome label. Sensitivity, specificity, PPV, NPV, referral burden and
number-needed-to-refer against that reference standard are what a screening
programme actually needs, and they are what this module computes.

Everything is derived from the same out-of-fold prediction dumps as Table 1,
using the same aggregation rule: mean-pool a patient's segment logits, then
argmax. The continuous screening score is the softmax of the pooled logits summed
over the *Present* and *Unknown* classes.

Data requirement
----------------
Only ``training_data.csv`` from CirCor DigiScope (about 114 KB) -- no audio.
It is not redistributed here; see ``data/README.md``.

Usage
-----
    python -m analysis.clinical_endpoint
    python -m analysis.clinical_endpoint --circor-meta path/to/training_data.csv \\
        --preds-dir results/predictions --out build/clinical_endpoint --dump-curves
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analysis._common import (
    ABSENT,
    FOLDS,
    HIGH_GRADE_SYSTOLIC,
    MODELS,
    PRESENT,
    UNKNOWN,
    add_circor_meta_argument,
    add_out_argument,
    add_preds_dir_argument,
    fold_path,
    load_patient_level,
    require_all_folds,
    resolve_circor_meta,
    resolve_out_dir,
    resolve_preds_dir,
)

__all__ = [
    "ENDPOINTS",
    "HIGH_GRADE_SYSTOLIC",
    "MODELS",
    "PRESENT",
    "PatientReference",
    "REFERRAL_RULES",
    "ScreeningMetrics",
    "attach_reference",
    "auroc",
    "endpoint_mask",
    "load_all_models",
    "load_out_of_fold",
    "load_reference",
    "per_fold_spread",
    "referral_cost_at_sensitivity",
    "referral_mask",
    "screening_metrics",
    "sweep_curve",
    "wilson_ci",
]

#: Columns of ``training_data.csv`` that this analysis dereferences.
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"Patient ID", "Murmur", "Outcome", "Systolic murmur grading",
     "Diastolic murmur grading"}
)

#: Read when present, never required: only used for descriptive output.
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("Age",)

#: Value the CirCor CSV uses for an absent grading once NaNs are filled.
NO_GRADE: Final[str] = "nan"


# --------------------------------------------------------------------------
# Reference standards
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PatientReference:
    """Per-patient reference labels drawn verbatim from the CirCor metadata."""

    murmur: str  # Present / Absent / Unknown (auscultation annotation)
    outcome_abnormal: bool  # expert clinical outcome (Normal / Abnormal)
    systolic_grade: str  # "I/VI", "II/VI", "III/VI", ... or NO_GRADE
    diastolic_grade: str
    age_group: str

    @property
    def has_diastolic(self) -> bool:
        return self.diastolic_grade != NO_GRADE

    @property
    def significant_murmur(self) -> bool:
        """Murmur with features clinical guidance calls non-innocent."""
        return self.systolic_grade in HIGH_GRADE_SYSTOLIC or self.has_diastolic


def load_reference(path: Path) -> dict[str, PatientReference]:
    """Load the CirCor patient metadata into immutable reference records."""
    frame = pd.read_csv(path, dtype=str).fillna(NO_GRADE)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise SystemExit(
            f"error: '{path.name}' is missing required column(s): {sorted(missing)}. "
            "Expected the unmodified training_data.csv from CirCor DigiScope."
        )
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = NO_GRADE
    return {
        str(row["Patient ID"]): PatientReference(
            murmur=row["Murmur"],
            outcome_abnormal=(row["Outcome"] == "Abnormal"),
            systolic_grade=row["Systolic murmur grading"],
            diastolic_grade=row["Diastolic murmur grading"],
            age_group=row["Age"],
        )
        for _, row in frame.iterrows()
    }


# --------------------------------------------------------------------------
# Out-of-fold model predictions
# --------------------------------------------------------------------------
def load_fold(preds_dir: Path, prefix: str, fold: int) -> pd.DataFrame:
    """Patient-level predictions for one held-out fold.

    Uses the project-wide aggregation rule (mean-pool segment logits, then
    argmax). The softmax of the pooled logits gives the continuous screening
    score used for the threshold sweeps.
    """
    patient = load_patient_level(fold_path(preds_dir, prefix, fold))
    if patient.n_classes != 3:
        raise SystemExit(
            f"error: '{prefix}_fold{fold}' has {patient.n_classes} logit columns; "
            "the clinical-endpoint analysis needs the three-class dumps."
        )
    probabilities = patient.probabilities
    return pd.DataFrame(
        {
            "patient": patient.patients,
            "fold": fold,
            "l0": patient.logits[:, PRESENT],
            "l1": patient.logits[:, UNKNOWN],
            "l2": patient.logits[:, ABSENT],
            "murmur_target": patient.targets,
            "p_present": probabilities[:, PRESENT],
            "p_unknown": probabilities[:, UNKNOWN],
            "p_absent": probabilities[:, ABSENT],
            "murmur_pred": patient.predictions,
        }
    )


def load_out_of_fold(preds_dir: Path, prefix: str) -> pd.DataFrame:
    """Concatenate the five held-out folds into one out-of-fold prediction set."""
    pooled = pd.concat(
        [load_fold(preds_dir, prefix, fold) for fold in FOLDS], ignore_index=True
    )
    duplicated = pooled.loc[pooled["patient"].duplicated(), "patient"].unique()
    if duplicated.size:
        raise SystemExit(
            f"error: {prefix}: folds overlap at the patient level for "
            f"{duplicated.size} patient(s), e.g. {sorted(duplicated)[:5]}."
        )
    return pooled


def attach_reference(
    preds: pd.DataFrame, reference: Mapping[str, PatientReference]
) -> pd.DataFrame:
    """Join model predictions with the clinical reference standards."""
    unknown_ids = sorted(set(preds["patient"]) - set(reference))
    if unknown_ids:
        raise SystemExit(
            f"error: {len(unknown_ids)} predicted patient(s) are absent from the "
            f"CirCor metadata, e.g. {unknown_ids[:5]}."
        )
    refs = [reference[p] for p in preds["patient"]]
    return preds.assign(
        outcome_abnormal=[r.outcome_abnormal for r in refs],
        murmur_label=[r.murmur for r in refs],
        systolic_grade=[r.systolic_grade for r in refs],
        has_diastolic=[r.has_diastolic for r in refs],
        significant_murmur=[r.significant_murmur for r in refs],
        age_group=[r.age_group for r in refs],
    )


def load_all_models(
    preds_dir: Path, reference: Mapping[str, PatientReference]
) -> dict[str, pd.DataFrame]:
    """Out-of-fold, reference-joined predictions for all six architectures."""
    require_all_folds(preds_dir, MODELS)
    return {
        name: attach_reference(load_out_of_fold(preds_dir, prefix), reference)
        for name, prefix in MODELS.items()
    }


# --------------------------------------------------------------------------
# Screening operating characteristics
# --------------------------------------------------------------------------
def wilson_ci(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    Preferred over the normal approximation for the small strata reported here
    (n=46 high-grade systolic murmurs, n=5 diastolic murmurs).
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (z / denominator) * np.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class ScreeningMetrics:
    """Diagnostic-accuracy summary of a binary referral rule."""

    n: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def sensitivity(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) else float("nan")

    @property
    def ppv(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def npv(self) -> float:
        return self.tn / (self.tn + self.fn) if (self.tn + self.fn) else float("nan")

    @property
    def referral_rate(self) -> float:
        """Fraction of screened patients sent for confirmatory assessment."""
        return (self.tp + self.fp) / self.n if self.n else float("nan")

    @property
    def nnr(self) -> float:
        """Number needed to refer: patients referred per true positive found."""
        return 1.0 / self.ppv if self.ppv else float("inf")

    @property
    def sensitivity_ci(self) -> tuple[float, float]:
        return wilson_ci(self.tp, self.tp + self.fn)

    @property
    def specificity_ci(self) -> tuple[float, float]:
        return wilson_ci(self.tn, self.tn + self.fp)

    @property
    def ppv_ci(self) -> tuple[float, float]:
        return wilson_ci(self.tp, self.tp + self.fp)

    @property
    def npv_ci(self) -> tuple[float, float]:
        return wilson_ci(self.tn, self.tn + self.fn)

    def per_thousand(self) -> dict[str, float]:
        """Decision-analytic summary per 1,000 patients screened."""
        scale = 1000.0 / self.n if self.n else float("nan")
        return {
            "referred": (self.tp + self.fp) * scale,
            "true_positive_referrals": self.tp * scale,
            "false_positive_referrals": self.fp * scale,
            "missed_positives": self.fn * scale,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "sensitivity": self.sensitivity, "sensitivity_ci": self.sensitivity_ci,
            "specificity": self.specificity, "specificity_ci": self.specificity_ci,
            "ppv": self.ppv, "ppv_ci": self.ppv_ci,
            "npv": self.npv, "npv_ci": self.npv_ci,
            "referral_rate": self.referral_rate,
            "nnr": self.nnr,
            "per_thousand": self.per_thousand(),
        }


def screening_metrics(referred: np.ndarray, positive: np.ndarray) -> ScreeningMetrics:
    """Confusion counts for a referral rule against a reference standard."""
    referred = np.asarray(referred, dtype=bool)
    positive = np.asarray(positive, dtype=bool)
    if referred.shape != positive.shape:
        raise ValueError("referral and reference arrays must be the same length")
    return ScreeningMetrics(
        n=int(referred.size),
        tp=int(np.sum(referred & positive)),
        fp=int(np.sum(referred & ~positive)),
        fn=int(np.sum(~referred & positive)),
        tn=int(np.sum(~referred & ~positive)),
    )


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC with ties averaged."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    sorted_scores = scores[order]
    start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_fold_spread(
    preds: pd.DataFrame, referred_col: str, positive_col: str, metric: str
) -> tuple[float, float]:
    """Mean and s.d. of a screening metric across the five held-out folds."""
    values = []
    for fold in FOLDS:
        subset = preds[preds["fold"] == fold]
        metrics = screening_metrics(
            subset[referred_col].to_numpy(), subset[positive_col].to_numpy()
        )
        values.append(getattr(metrics, metric))
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(np.nanstd(array))


# --------------------------------------------------------------------------
# Clinical endpoints and referral rules
# --------------------------------------------------------------------------
#  E1  the cardiologist's holistic diagnosis -- the endpoint that decides whether
#      a screened child needs onward specialist assessment.
#  E2  a murmur the cardiologist also judged abnormal, i.e. a pathological murmur
#      as opposed to any murmur at all.
#  E3  a murmur whose auscultatory features clinical guidance treats as
#      non-innocent (>=III/VI systolic, or any diastolic murmur).
ENDPOINTS: Final[Mapping[str, str]] = {
    "abnormal_cardiac_status": "Abnormal cardiac status (expert cardiologist)",
    "pathological_murmur": "Pathological murmur (murmur + abnormal status)",
    "non_innocent_murmur": "Non-innocent murmur (>=III/VI or diastolic)",
}

REFERRAL_RULES: Final[Mapping[str, str]] = {
    "model_present": "model predicts Present (deployed argmax rule)",
    "model_present_or_unknown": "model predicts Present or Unknown",
}

REFERENCE_RULES: Final[Mapping[str, str]] = {
    "expert_annotation": "Expert murmur annotation",
    "refer_all": "Refer every patient",
}


def endpoint_mask(preds: pd.DataFrame, endpoint: str) -> np.ndarray:
    """Boolean reference-standard vector for one clinical endpoint."""
    if endpoint == "abnormal_cardiac_status":
        return preds["outcome_abnormal"].to_numpy(dtype=bool)
    if endpoint == "pathological_murmur":
        return (
            (preds["murmur_label"] == "Present") & preds["outcome_abnormal"]
        ).to_numpy(dtype=bool)
    if endpoint == "non_innocent_murmur":
        return preds["significant_murmur"].to_numpy(dtype=bool)
    raise ValueError(f"unknown endpoint: {endpoint!r}")


def referral_mask(preds: pd.DataFrame, rule: str) -> np.ndarray:
    """Boolean referral vector for one decision rule."""
    if rule == "model_present":
        return (preds["murmur_pred"] == PRESENT).to_numpy(dtype=bool)
    if rule == "model_present_or_unknown":
        return preds["murmur_pred"].isin([PRESENT, UNKNOWN]).to_numpy(dtype=bool)
    if rule == "expert_annotation":
        return (preds["murmur_label"] == "Present").to_numpy(dtype=bool)
    if rule == "refer_all":
        return np.ones(len(preds), dtype=bool)
    raise ValueError(f"unknown referral rule: {rule!r}")


def abnormality_score(preds: pd.DataFrame) -> np.ndarray:
    """Continuous screening score: pooled-softmax P(Present) + P(Unknown)."""
    return (preds["p_present"] + preds["p_unknown"]).to_numpy(dtype=float)


def sweep_curve(preds: pd.DataFrame, endpoint: str) -> list[dict[str, float]]:
    """Sensitivity / referral-burden trade-off over all distinct thresholds."""
    score = abnormality_score(preds)
    positive = endpoint_mask(preds, endpoint)
    thresholds = np.unique(np.concatenate([[0.0], score, [1.0 + 1e-9]]))
    rows = []
    for threshold in thresholds:
        metrics = screening_metrics(score >= threshold, positive)
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
                "ppv": metrics.ppv,
                "npv": metrics.npv,
                "referral_rate": metrics.referral_rate,
            }
        )
    return rows


def referral_cost_at_sensitivity(
    curve: Sequence[Mapping[str, float]], target: float
) -> dict[str, float] | None:
    """Cheapest operating point reaching at least ``target`` sensitivity."""
    feasible = [row for row in curve if row["sensitivity"] >= target]
    return dict(min(feasible, key=lambda r: r["referral_rate"])) if feasible else None


# --------------------------------------------------------------------------
# Severity strata
# --------------------------------------------------------------------------
STRATA: Final[Mapping[str, str]] = {
    "grade I/VI": "systolic_grade == I/VI",
    "grade II/VI": "systolic_grade == II/VI",
    "grade >=III/VI": "systolic_grade in HIGH_GRADE_SYSTOLIC",
    "any diastolic murmur": "has_diastolic",
    "non-innocent (>=III/VI or dias.)": "significant_murmur",
    "murmur Present, outcome Abnormal": "murmur_label == Present & abnormal",
    "murmur Present, outcome Normal": "murmur_label == Present & normal",
}


def stratum_mask(preds: pd.DataFrame, label: str) -> np.ndarray:
    """Boolean membership vector for one severity stratum."""
    if label == "grade I/VI":
        return (preds["systolic_grade"] == "I/VI").to_numpy(dtype=bool)
    if label == "grade II/VI":
        return (preds["systolic_grade"] == "II/VI").to_numpy(dtype=bool)
    if label == "grade >=III/VI":
        return preds["systolic_grade"].isin(HIGH_GRADE_SYSTOLIC).to_numpy(dtype=bool)
    if label == "any diastolic murmur":
        return preds["has_diastolic"].to_numpy(dtype=bool)
    if label == "non-innocent (>=III/VI or dias.)":
        return preds["significant_murmur"].to_numpy(dtype=bool)
    if label == "murmur Present, outcome Abnormal":
        return (
            (preds["murmur_label"] == "Present") & preds["outcome_abnormal"]
        ).to_numpy(dtype=bool)
    if label == "murmur Present, outcome Normal":
        return (
            (preds["murmur_label"] == "Present") & ~preds["outcome_abnormal"]
        ).to_numpy(dtype=bool)
    raise ValueError(f"unknown stratum: {label!r}")


# --------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.clinical_endpoint",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preds_dir_argument(parser)
    add_circor_meta_argument(parser)
    add_out_argument(parser, "build/clinical_endpoint")
    parser.add_argument(
        "--dump-curves",
        action="store_true",
        help="also write the full threshold sweeps (about 1.5 MB of JSON)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Imported here rather than at module scope: the report module builds on this
    # one, so a top-level import would close a cycle.
    from analysis._clinical_report import report

    args = build_parser().parse_args(argv)
    report(
        resolve_preds_dir(args.preds_dir),
        resolve_circor_meta(args.circor_meta),
        resolve_out_dir(args),
        dump_curves=args.dump_curves,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
