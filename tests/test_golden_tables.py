"""Freeze the published tables against the scripts that regenerate them.

Table 1 and Table 2 are the paper's two main results tables, and both are
regenerated from the committed out-of-fold prediction dumps.  A refactor that
moved a published number by one in the third decimal would otherwise be
invisible, so every cell is pinned here, transcribed from the camera-ready.

Three layers of defence per table:

1. the machine-readable payload the script writes (``table1_fivefold.json``),
   cell by cell;
2. the human-readable table it prints, so the LaTeX body pasted into the
   manuscript carries the same numbers;
3. an independent recomputation from the raw CSVs, using the reference pooling
   implementation in ``conftest``, so a bug shared by the analysis helpers
   cannot make both of the above agree on a wrong value.

Table 2 needs CirCor's ``training_data.csv``, which is never redistributed;
those tests skip without it.  Table 1 needs only what the repository ships.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    PATHS,
    REPO_ROOT,
    THREE_CLASS_MODELS,
    prediction_csv,
    predictions_dir,
    reference_patient_pool,
    require_circor_meta,
)

from mirasol.metrics import (
    official_physionet_weighted_accuracy,
    project_challenge_score,
)

SCRIPT_TIMEOUT_S = 900
NUMBER = re.compile(r"-?\d*\.\d+")

#: Table 1, transcribed cell by cell as (mean, s.d.).  Column order is the
#: paper's: MobileNetV3-S, FastViT, EffNet-B0, TinyViT, ResNet-18, HTS-AT.
CAMERA_READY_TABLE1: dict[str, tuple[tuple[float, float], ...]] = {
    "Challenge Score": ((.886, .008), (.888, .009), (.889, .014), (.894, .015),
                        (.896, .011), (.875, .020)),
    "Accuracy": ((.848, .013), (.847, .013), (.850, .018), (.853, .017),
                 (.858, .016), (.834, .028)),
    "Macro F1": ((.582, .040), (.562, .028), (.578, .045), (.604, .048),
                 (.617, .024), (.618, .054)),
    "Precision (Present)": ((.936, .053), (.945, .059), (.951, .044),
                            (.938, .036), (.905, .044), (.835, .075)),
    "Recall (Present)": ((.603, .042), (.620, .037), (.583, .066), (.624, .067),
                         (.673, .082), (.729, .065)),
    "F1 (Present)": ((.731, .024), (.746, .020), (.721, .058), (.748, .052),
                     (.767, .053), (.774, .040)),
    "F1 (Unknown)": ((.107, .093), (.033, .067), (.104, .093), (.151, .097),
                     (.169, .053), (.185, .141)),
}

#: Substring identifying each Table 1 row in the printed table, short enough to
#: match the abbreviated labels ("Precision (Pres.)").
TABLE1_ROW_LABELS: dict[str, str] = {
    "Challenge Score": "Challenge Score ",
    "Accuracy": "Accuracy",
    "Macro F1": "Macro F1",
    "Precision (Present)": "Precision",
    "Recall (Present)": "Recall",
    "F1 (Present)": "F1 (Pres",
    "F1 (Unknown)": "F1 (Unk",
}

#: The one cell where the camera-ready and a recomputation from the shipped
#: predictions differ in the third decimal: MobileNetV3-S F1 (Unknown) s.d. is
#: printed as .093 and measures .0922.  Recorded in ``docs/errata.md``.
ROUNDING_EXCEPTIONS: frozenset[tuple[str, str, str]] = frozenset(
    {("F1 (Unknown)", "MobileNetV3-S", "sd")}
)

#: What the scripts therefore print for those cells, to three decimals.
MEASURED_OVERRIDES: dict[tuple[str, str, str], float] = {
    ("F1 (Unknown)", "MobileNetV3-S", "sd"): 0.092,
}

#: Key of the official metric in the script's payload.
OFFICIAL_ROW = "Challenge Score (official PhysioNet)"

#: The official PhysioNet weighted accuracy of the same runs, measured from the
#: same dumps.  Not in the paper: the repository reports it because the two
#: metrics rank the architectures differently (``docs/errata.md``).
OFFICIAL_CHALLENGE_SCORE: tuple[tuple[float, float], ...] = (
    (.690, .030), (.693, .026), (.684, .042), (.705, .041), (.727, .043),
    (.741, .043),
)

#: Parameter counts in the Table 1 column headers, in millions.
PUBLISHED_PARAMS: tuple[float, ...] = (1.52, 3.26, 4.01, 5.07, 11.2, 27.6)

#: Table 2: referral burden (%), E1 sensitivity, E1 PPV, E2 sensitivity,
#: E2 PPV, E3 sensitivity.
CAMERA_READY_TABLE2: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("MobileNetV3-S", (12.3, .235, .922, .673, .871, .920)),
    ("FastViT", (12.5, .240, .924, .693, .881, .940)),
    ("EfficientNet-B0", (11.7, .231, .955, .673, .918, .960)),
    ("TinyViT", (12.6, .246, .941, .713, .899, .980)),
    ("ResNet-18", (14.2, .264, .896, .760, .851, .940)),
    ("HTS-AT", (16.8, .288, .829, .780, .741, .940)),
    ("Expert murmur annotation", (19.0, .330, .838, 1.000, .838, 1.000)),
)

#: Endpoint prevalences quoted in the Table 2 caption and Sec. 3.2.
TABLE2_PREVALENCES: tuple[str, ...] = ("455/941", "150/941", "50/941")


# --------------------------------------------------------------------------
# Output matching
# --------------------------------------------------------------------------
def numbers_in(text: str) -> list[float]:
    """Every decimal number in ``text``, rounded to three decimals."""
    return [round(float(match.group()), 3) for match in NUMBER.finditer(text)]


def contains_in_order(haystack: Sequence[float], needles: Sequence[float]) -> bool:
    """True if ``needles`` occur in ``haystack`` in order."""
    position = 0
    for needle in needles:
        while position < len(haystack) and haystack[position] != needle:
            position += 1
        if position == len(haystack):
            return False
        position += 1
    return True


def assert_row_printed(
    output: str, label: str, expected: Sequence[float], context: str
) -> None:
    """Fail unless some line of ``output`` carries ``expected`` in order."""
    lowered = label.lower()
    for line in output.splitlines():
        if lowered in line.lower() and contains_in_order(numbers_in(line), expected):
            return
    pytest.fail(
        f"{context}: no printed line matching {label!r} carries the published "
        f"values {list(expected)}.\n----- output tail -----\n{output[-3000:]}"
    )


def interleave(cells: Sequence[tuple[float, float]]) -> list[float]:
    """(mean, s.d.) pairs flattened in Table 1 column order."""
    return [value for cell in cells for value in cell]


def printed_cells(metric: str) -> list[tuple[float, float]]:
    """Published cells with the documented rounding divergence applied.

    The scripts print the measured value; the camera-ready printed .093 where
    the data give .0922, so that one cell is looked up in
    :data:`MEASURED_OVERRIDES` instead of being expected verbatim.
    """
    cells: list[tuple[float, float]] = []
    for (name, _), (mean, sd) in zip(
        THREE_CLASS_MODELS, CAMERA_READY_TABLE1[metric], strict=True
    ):
        cells.append(
            (
                MEASURED_OVERRIDES.get((metric, name, "mean"), mean),
                MEASURED_OVERRIDES.get((metric, name, "sd"), sd),
            )
        )
    return cells


# --------------------------------------------------------------------------
# Running the analysis entry points
# --------------------------------------------------------------------------
def script_env() -> dict[str, str]:
    """Environment that lets ``python -m analysis.<script>`` run in place."""
    env = dict(os.environ)
    entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["MPLBACKEND"] = "Agg"
    return env


def run_analysis(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Execute an analysis entry point from the repository root."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", f"analysis.{module}", *args],
        cwd=REPO_ROOT,
        env=script_env(),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_S,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], module: str) -> str:
    if result.returncode != 0:
        pytest.fail(
            f"analysis.{module} exited {result.returncode}\n"
            f"----- stdout -----\n{result.stdout[-2000:]}\n"
            f"----- stderr -----\n{result.stderr[-2000:]}"
        )
    return result.stdout


@pytest.fixture(scope="module")
def table1(preds_dir, tmp_path_factory) -> tuple[str, dict]:
    """Run the Table 1 script once; return its output and JSON payload."""
    del preds_dir
    out = tmp_path_factory.mktemp("table1")
    stdout = require_success(
        run_analysis("table1_fivefold", "--out", str(out)), "table1_fivefold"
    )
    payload = json.loads((out / "table1_fivefold.json").read_text())
    return stdout, payload


@pytest.fixture(scope="module")
def table2(preds_dir) -> str:
    """Run the Table 2 script once, or skip if CirCor's metadata is absent."""
    del preds_dir
    meta = require_circor_meta()
    return require_success(
        run_analysis("table2_clinical", "--circor-meta", str(meta)), "table2_clinical"
    )


# --------------------------------------------------------------------------
# Table 1: payload
# --------------------------------------------------------------------------
def test_table1_payload_has_the_published_layout(table1: tuple[str, dict]) -> None:
    """All six architectures, in column order, and all eight metric rows."""
    _, payload = table1
    assert payload["metric_order"] == list(CAMERA_READY_TABLE1) + [OFFICIAL_ROW]
    assert list(payload["models"]) == [name for name, _ in THREE_CLASS_MODELS]
    assert payload["folds"] == [0, 1, 2, 3, 4]
    for (name, prefix), params in zip(
        THREE_CLASS_MODELS, PUBLISHED_PARAMS, strict=True
    ):
        entry = payload["models"][name]
        assert entry["prefix"] == prefix
        assert float(entry["params_m"]) == params
        assert len(entry["per_fold"]) == 5


@pytest.mark.parametrize("metric", list(CAMERA_READY_TABLE1))
def test_table1_cells_are_frozen(metric: str, table1: tuple[str, dict]) -> None:
    """Every Table 1 cell, mean and s.d., against the camera-ready."""
    _, payload = table1
    for (name, _), (mean, sd) in zip(
        THREE_CLASS_MODELS, CAMERA_READY_TABLE1[metric], strict=True
    ):
        got = payload["models"][name]["mean_sd"][metric]
        assert got["mean"] == pytest.approx(mean, abs=1e-3), f"{metric} / {name} mean"
        assert got["sd"] == pytest.approx(sd, abs=1e-3), f"{metric} / {name} s.d."
        if (metric, name, "mean") not in ROUNDING_EXCEPTIONS:
            assert round(got["mean"], 3) == mean, f"{metric} / {name} mean"
        if (metric, name, "sd") not in ROUNDING_EXCEPTIONS:
            assert round(got["sd"], 3) == sd, f"{metric} / {name} s.d."


def test_the_one_documented_rounding_divergence_is_still_the_only_one(
    table1: tuple[str, dict],
) -> None:
    """MobileNetV3-S F1 (Unknown) s.d.: printed .093, measures .0922."""
    _, payload = table1
    got = payload["models"]["MobileNetV3-S"]["mean_sd"]["F1 (Unknown)"]["sd"]
    assert round(got, 3) == 0.092
    assert got == pytest.approx(0.0922, abs=5e-5)


def test_official_challenge_score_is_frozen(table1: tuple[str, dict]) -> None:
    """The metric the paper's caption names but does not report."""
    _, payload = table1
    for (name, _), (mean, sd) in zip(
        THREE_CLASS_MODELS, OFFICIAL_CHALLENGE_SCORE, strict=True
    ):
        got = payload["models"][name]["mean_sd"][OFFICIAL_ROW]
        assert round(got["mean"], 3) == mean, f"official / {name} mean"
        assert round(got["sd"], 3) == sd, f"official / {name} s.d."


def test_the_two_metrics_rank_the_architectures_differently(
    table1: tuple[str, dict],
) -> None:
    """The divergence ``docs/errata.md`` documents, asserted rather than claimed."""
    _, payload = table1

    def best(row: str) -> str:
        return max(
            payload["models"], key=lambda name: payload["models"][name]["mean_sd"][row]["mean"]
        )

    assert best("Challenge Score") == "ResNet-18"
    assert best(OFFICIAL_ROW) == "HTS-AT"


def test_architecture_independence_survives_both_metrics(
    table1: tuple[str, dict],
) -> None:
    """The paper's central claim: best and worst overlap within one s.d."""
    _, payload = table1
    for row in ("Challenge Score", OFFICIAL_ROW):
        cells = [payload["models"][name]["mean_sd"][row] for name in payload["models"]]
        best = max(cells, key=lambda cell: cell["mean"])
        worst = min(cells, key=lambda cell: cell["mean"])
        assert worst["mean"] + worst["sd"] >= best["mean"] - best["sd"], row


# --------------------------------------------------------------------------
# Table 1: printed table
# --------------------------------------------------------------------------
@pytest.mark.parametrize("metric", list(CAMERA_READY_TABLE1))
def test_table1_printed_row_is_frozen(metric: str, table1: tuple[str, dict]) -> None:
    stdout, _ = table1
    assert_row_printed(
        stdout,
        TABLE1_ROW_LABELS[metric],
        interleave(printed_cells(metric)),
        f"analysis/table1_fivefold.py, Table 1 row {metric!r}",
    )


def test_table1_prints_both_challenge_score_variants(table1: tuple[str, dict]) -> None:
    """Every table script must report the official metric side by side."""
    stdout, _ = table1
    assert re.search(r"official", stdout, re.IGNORECASE)
    assert_row_printed(
        stdout,
        "official",
        interleave(OFFICIAL_CHALLENGE_SCORE),
        "analysis/table1_fivefold.py, official PhysioNet row",
    )


def test_table1_script_refuses_to_print_a_partial_table(
    tmp_path: Path, preds_dir: Path
) -> None:
    """A missing fold must be an error, not a five-fold mean over four folds."""
    del preds_dir
    incomplete = tmp_path / "predictions"
    incomplete.mkdir()
    dropped = "mobilenetv3_3class_fold3.data.csv"
    for source in sorted(predictions_dir().glob("*.data.csv")):
        if source.name != dropped:
            (incomplete / source.name).symlink_to(source.resolve())
    assert not (incomplete / dropped).exists()

    result = run_analysis(
        "table1_fivefold", "--preds-dir", str(incomplete), "--no-write"
    )
    assert result.returncode != 0, (
        "the script printed a table with a fold file missing; it must name the "
        "missing files and exit non-zero"
    )
    assert dropped in result.stdout + result.stderr


# --------------------------------------------------------------------------
# Table 1: independent recomputation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("row", "published"),
    [
        ("Challenge Score", CAMERA_READY_TABLE1["Challenge Score"]),
        (OFFICIAL_ROW, OFFICIAL_CHALLENGE_SCORE),
    ],
)
def test_challenge_scores_recompute_from_the_raw_csvs(
    row: str, published: tuple[tuple[float, float], ...], preds_dir
) -> None:
    """Recomputed with the reference pooling, bypassing the analysis helpers."""
    del preds_dir
    scorer = (
        project_challenge_score
        if row == "Challenge Score"
        else official_physionet_weighted_accuracy
    )
    for (name, prefix), (mean, sd) in zip(THREE_CLASS_MODELS, published, strict=True):
        per_fold = []
        for fold in range(5):
            pooled = reference_patient_pool(prediction_csv(prefix, fold))
            per_fold.append(
                scorer(pooled["target"].to_numpy(), pooled["pred"].to_numpy())
            )
        assert float(np.mean(per_fold)) == pytest.approx(mean, abs=1e-3), f"{row}/{name}"
        assert float(np.std(per_fold)) == pytest.approx(sd, abs=1e-3), f"{row}/{name}"


# --------------------------------------------------------------------------
# Table 2
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("label", "cells"), CAMERA_READY_TABLE2)
def test_table2_printed_row_is_frozen(
    label: str, cells: tuple[float, ...], table2: str
) -> None:
    assert_row_printed(
        table2, label, cells, f"analysis/table2_clinical.py, Table 2 row {label!r}"
    )


def test_table2_reports_the_published_cohort_and_prevalences(table2: str) -> None:
    """n=941 of 942, and the three endpoint prevalences of the caption."""
    assert "941" in table2
    for prevalence in TABLE2_PREVALENCES:
        assert prevalence in table2, prevalence


def test_table2_emits_exactly_the_seven_published_columns(table2: str) -> None:
    """Specificity and E3 NPV were dropped by hand from a nine-column draft.

    The docstring's promise that nothing is transcribed by hand only holds if
    the script emits the shipped columns and no others.
    """
    rows = [
        line
        for line in table2.splitlines()
        if line.rstrip().endswith(r"\\") and "textbf{(" not in line
    ]
    assert len(rows) == len(CAMERA_READY_TABLE2)
    for line in rows:
        assert len(line.split("&")) == 7, line


def test_table2_needs_only_the_metadata_csv() -> None:
    """The clinical endpoints must not require the full audio download."""
    meta = require_circor_meta()
    assert meta.name == "training_data.csv"
    assert meta.stat().st_size < 1_000_000
    assert PATHS.circor_audio != meta
