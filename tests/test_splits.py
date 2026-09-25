"""Tests for the released five-fold patient partition.

``data/splits/folds.csv`` discharges the abstract's promise to release the data
splits, and every per-fold number in the paper (Table 1, Table 2, Tables S1-S8)
is conditioned on it.  The supplement describes the recipe as "deterministic
round-robin over sorted patient identifiers with seed 42" and omits the shuffle
the implementation performs, so the CSV - not the prose - is ground truth.

Asserted here: the CSV's schema and cohort; that the recipe in
``mirasol.data.splits`` reproduces it; that a second, independent
implementation of the recipe reproduces it too; that the partition implied by
``results/predictions/`` agrees with it for all six three-class models and all
four two-class models; and that the literal reading of the supplement does
*not* reproduce it, which is why the file ships.

No CirCor audio or metadata is required: the patient identifiers come from the
committed prediction dumps.
"""

from __future__ import annotations

import csv
import random
from collections import Counter

import pytest
from conftest import (
    BINARY_MODELS,
    EXCLUDED_PATIENT,
    FOLD_PATIENT_COUNTS,
    FOLD_SEGMENT_COUNTS,
    LITERAL_READING_MISMATCHES,
    N_FOLDS,
    N_PATIENTS,
    PATHS,
    SPLIT_SEED,
    THREE_CLASS_MODELS,
    literal_reading_fold_assignment,
    prediction_csv,
    read_prediction_frame,
    reference_fold_assignment,
)

from mirasol.data.splits import (
    EXPECTED_FOLD_SIZES,
    EXPECTED_PATIENT_COUNT,
    FOLDS_CSV_HEADER,
    compare_partitions,
    fold_assignment,
    fold_counts,
    observed_partition,
    patient_id_from_segment_name,
    patients_from_predictions,
    read_folds_csv,
    write_folds_csv,
)


@pytest.fixture(scope="module")
def folds() -> dict[str, int]:
    """The released ``patient_id,fold`` table."""
    path = PATHS.folds_csv
    if not path.is_file():
        pytest.fail(
            f"{path.name} is missing: the released data splits must be committed"
        )
    return read_folds_csv(path)


@pytest.fixture(scope="module")
def implied(preds_dir) -> dict[str, int]:
    """The partition the committed prediction dumps encode."""
    return observed_partition(preds_dir)


# --------------------------------------------------------------------------
# Shape of the released CSV
# --------------------------------------------------------------------------
def test_folds_csv_has_the_documented_header_and_941_rows() -> None:
    with PATHS.folds_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == FOLDS_CSV_HEADER == ("patient_id", "fold")
    assert len(rows) == N_PATIENTS + 1, "941 data rows plus one header row"
    assert all(len(row) == 2 for row in rows)


def test_folds_csv_holds_941_unique_numeric_pseudonyms(folds: dict[str, int]) -> None:
    assert len(folds) == N_PATIENTS == EXPECTED_PATIENT_COUNT
    assert all(patient.isdigit() for patient in folds), (
        "identifiers must be CirCor's own numeric pseudonyms, nothing else"
    )


def test_fold_sizes_are_the_documented_ones(folds: dict[str, int]) -> None:
    counts = Counter(folds.values())
    assert sorted(counts) == list(range(N_FOLDS))
    assert tuple(counts[fold] for fold in range(N_FOLDS)) == FOLD_PATIENT_COUNTS
    assert EXPECTED_FOLD_SIZES == FOLD_PATIENT_COUNTS
    assert fold_counts(folds) == FOLD_PATIENT_COUNTS
    assert sum(FOLD_PATIENT_COUNTS) == N_PATIENTS


def test_the_documented_dropped_patient_is_absent(folds: dict[str, int]) -> None:
    """941 of 942: patient 50321's recordings never matched the loader."""
    assert EXCLUDED_PATIENT not in folds


# --------------------------------------------------------------------------
# The recipe
# --------------------------------------------------------------------------
def test_independent_recipe_reproduces_folds_csv(folds: dict[str, int]) -> None:
    """sorted(ids) -> Random(42).shuffle -> i % 5, reimplemented in the test."""
    assert reference_fold_assignment(folds) == folds


def test_package_recipe_reproduces_folds_csv(folds: dict[str, int]) -> None:
    assert fold_assignment(folds) == folds


def test_recipe_is_independent_of_the_caller_s_order(folds: dict[str, int]) -> None:
    """The recipe sorts internally, so enumeration order cannot change a fold."""
    assert fold_assignment(sorted(folds, reverse=True)) == folds
    assert fold_assignment(sorted(folds, key=int)) == folds


def test_recipe_depends_on_string_sorting(folds: dict[str, int]) -> None:
    """"1001" < "999": sorting numerically instead deals a different split.

    Pinned because the identifiers look numeric, so a future cleanup could
    plausibly "fix" the sort and silently invalidate every per-fold number.
    """
    numeric_order = sorted(folds, key=int)
    random.Random(SPLIT_SEED).shuffle(numeric_order)
    numeric_recipe = {pid: index % N_FOLDS for index, pid in enumerate(numeric_order)}
    assert numeric_recipe != folds


def test_a_different_seed_gives_a_different_split(folds: dict[str, int]) -> None:
    assert fold_assignment(folds, seed=SPLIT_SEED + 1) != folds


def test_literal_reading_of_the_supplement_does_not_reproduce_the_splits(
    folds: dict[str, int],
) -> None:
    """Why folds.csv ships: the published prose omits the shuffle.

    Round-robin over merely sorted identifiers misplaces 742 of the 941
    patients, so no per-fold number is reconstructible from the prose alone.
    """
    literal = literal_reading_fold_assignment(folds)
    mismatches = sum(1 for pid, fold in folds.items() if literal[pid] != fold)
    assert mismatches == LITERAL_READING_MISMATCHES


def test_write_folds_csv_round_trips(folds: dict[str, int], tmp_path) -> None:
    written = write_folds_csv(folds, tmp_path / "folds.csv")
    assert read_folds_csv(written) == folds
    assert written.read_text(encoding="utf-8") == PATHS.folds_csv.read_text(
        encoding="utf-8"
    ), "the committed CSV must be exactly what the writer produces"


# --------------------------------------------------------------------------
# Agreement with the committed predictions
# --------------------------------------------------------------------------
def test_prediction_dumps_enumerate_exactly_the_941_patients(preds_dir) -> None:
    assert len(patients_from_predictions(preds_dir)) == N_PATIENTS


def test_folds_csv_matches_the_partition_implied_by_the_predictions(
    folds: dict[str, int], implied: dict[str, int]
) -> None:
    missing, extra, disagreements = compare_partitions(folds, implied)
    assert missing == ()
    assert extra == ()
    assert disagreements == ()


@pytest.mark.parametrize("prefix", [prefix for _, prefix in THREE_CLASS_MODELS])
def test_every_three_class_model_shares_one_partition(
    prefix: str, folds: dict[str, int], preds_dir
) -> None:
    del preds_dir
    for fold in range(N_FOLDS):
        frame = read_prediction_frame(prediction_csv(prefix, fold))
        assert len(frame) == FOLD_SEGMENT_COUNTS[fold]
        patients = set(frame["patient"])
        assert patients == {pid for pid, f in folds.items() if f == fold}


@pytest.mark.parametrize("prefix", BINARY_MODELS)
def test_two_class_runs_use_the_same_partition(
    prefix: str, folds: dict[str, int], preds_dir
) -> None:
    """Table S7 must be comparable with Table 1, so the folds must match."""
    del preds_dir
    for fold in range(N_FOLDS):
        frame = read_prediction_frame(prediction_csv(prefix, fold))
        assert frame.shape[1] == 5, "two logits, target, filename, patient"
        assert len(frame) == FOLD_SEGMENT_COUNTS[fold]
        assert set(frame["patient"]) == {pid for pid, f in folds.items() if f == fold}


def test_no_patient_leaks_between_folds(folds: dict[str, int], preds_dir) -> None:
    """Patient-level splitting is the paper's stated leakage control."""
    del preds_dir
    for _, prefix in THREE_CLASS_MODELS:
        seen: dict[str, int] = {}
        for fold in range(N_FOLDS):
            frame = read_prediction_frame(prediction_csv(prefix, fold))
            for patient in frame["patient"].unique():
                assert patient not in seen, f"{prefix}: {patient} leaks between folds"
                seen[patient] = fold
        assert seen == folds


def test_patient_id_extraction_handles_the_shipped_naming() -> None:
    assert patient_id_from_segment_name("50043_AV.wav") == "50043"
    assert patient_id_from_segment_name("50043_AV-2.wav") == "50043"
    with pytest.raises(ValueError):
        patient_id_from_segment_name("50043.wav")
