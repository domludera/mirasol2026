# SPDX-License-Identifier: MIT
"""The canonical patient-level five-fold recipe, as executable code.

The supplement describes the splits as "deterministic round-robin over sorted patient
identifiers with seed 42".  The implementation sorts and **then shuffles**, and a
literal reading of the prose therefore reconstructs 742 of 941 patients into the wrong
fold.  The three steps actually used are::

    patients = sorted(set(patient_ids))     # string sort, not numeric
    random.Random(42).shuffle(patients)     # CPython's Mersenne Twister
    fold_of[patients[i]] = i % 5            # round robin

which yields 189 / 188 / 188 / 188 / 188 = 941 patients.  ``data/splits/folds.csv`` is
the ground truth shipped alongside this module so that nobody has to re-derive it, and
``tests/test_splits.py`` asserts the file, the recipe and the partition implied by
``results/predictions/`` all agree.

Note the two different seeds in this project: **42** splits the data (here), **123777**
seeds training (``mirasol.config.defaults.random_seed``).

Command line::

    # Recompute from the committed predictions and rewrite data/splits/folds.csv
    python -m mirasol.data.splits --source predictions --write

    # Verify the shipped file without touching it (exit 1 on any mismatch)
    python -m mirasol.data.splits --source predictions --check

    # Recompute from the CirCor download instead (needs the audio directory)
    python -m mirasol.data.splits --source audio --write

Paper artifacts served: ``data/splits/folds.csv``, docs/splits.md, and the fold
structure behind Table 1, Table 2 and Tables S1-S8.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from mirasol import paths

#: Seed for the patient shuffle.  Distinct from the training seed (123777).
SPLIT_SEED: Final[int] = 42

#: Number of cross-validation folds.
N_FOLDS: Final[int] = 5

#: Patients that survive the ``{patient}_{location}.wav`` loader; see
#: :mod:`mirasol.data.prepare` for the 942 -> 941 accounting.
EXPECTED_PATIENT_COUNT: Final[int] = 941

#: Per-fold patient counts for the published 941-patient split.
EXPECTED_FOLD_SIZES: Final[tuple[int, ...]] = (189, 188, 188, 188, 188)

#: Header of ``data/splits/folds.csv``.
FOLDS_CSV_HEADER: Final[tuple[str, str]] = ("patient_id", "fold")

#: Matches the ``..._fold<N>.data.csv`` prediction dumps, whose fold index is the
#: held-out fold the file was evaluated on.
_FOLD_FILE_RE: Final[re.Pattern[str]] = re.compile(r"_fold(\d+)\.data\.csv$")

__all__ = [
    "EXPECTED_FOLD_SIZES",
    "EXPECTED_PATIENT_COUNT",
    "FOLDS_CSV_HEADER",
    "N_FOLDS",
    "SPLIT_SEED",
    "compare_partitions",
    "fold_assignment",
    "fold_counts",
    "main",
    "observed_partition",
    "patient_id_from_segment_name",
    "patients_from_predictions",
    "read_folds_csv",
    "shuffled_patients",
    "write_folds_csv",
]


# --------------------------------------------------------------------------- #
# The recipe
# --------------------------------------------------------------------------- #

def shuffled_patients(
    patient_ids: Iterable[str], seed: int = SPLIT_SEED
) -> tuple[str, ...]:
    """Return the patient order the folds are dealt from.

    Deduplicates, sorts as **strings** (``"1001" < "999"``, which is what the
    implementation did and therefore what the published folds encode), then shuffles
    with ``random.Random(seed)``.  The result depends on CPython's Mersenne Twister:
    another Python implementation, or a future change to ``random.shuffle``, would
    produce a different deal, which is why ``data/splits/folds.csv`` ships.

    Args:
        patient_ids: Patient identifiers, with or without duplicates.
        seed: Shuffle seed; 42 for every published run.

    Returns:
        The shuffled patients, as an immutable tuple.

    Raises:
        ValueError: If no patient identifiers were supplied.
    """
    patients = sorted(set(patient_ids))
    if not patients:
        raise ValueError("no patient identifiers to split")
    rng = random.Random(seed)
    rng.shuffle(patients)
    return tuple(patients)


def fold_assignment(
    patient_ids: Iterable[str],
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
) -> dict[str, int]:
    """Assign each patient to a fold by round robin over :func:`shuffled_patients`.

    Args:
        patient_ids: Patient identifiers.
        n_folds: Number of folds; 5 for every published run.
        seed: Shuffle seed.

    Returns:
        A mapping ``patient_id -> fold``, ordered by patient id so that the derived
        CSV is stable.

    Raises:
        ValueError: If ``n_folds`` is not positive.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    ordered = shuffled_patients(patient_ids, seed=seed)
    assignment = {pid: i % n_folds for i, pid in enumerate(ordered)}
    return {pid: assignment[pid] for pid in sorted(assignment)}


def fold_counts(
    assignment: Mapping[str, int], n_folds: int = N_FOLDS
) -> tuple[int, ...]:
    """Count patients per fold.

    Args:
        assignment: Mapping ``patient_id -> fold``.
        n_folds: Number of folds to report, so empty folds still appear.

    Returns:
        A tuple of length ``n_folds``.

    Raises:
        ValueError: If any fold index is outside ``range(n_folds)``.
    """
    counts = [0] * n_folds
    for pid, fold in assignment.items():
        if not 0 <= fold < n_folds:
            raise ValueError(
                f"patient {pid!r} has fold {fold}, outside range(0, {n_folds})"
            )
        counts[fold] += 1
    return tuple(counts)


# --------------------------------------------------------------------------- #
# folds.csv
# --------------------------------------------------------------------------- #

def write_folds_csv(assignment: Mapping[str, int], path: Path) -> Path:
    """Write ``patient_id,fold`` rows, sorted by patient id.

    ``data/splits/`` is version-controlled - it is the one directory this repository
    both tracks and generates - so this function writes there deliberately and does
    not route through :meth:`mirasol.paths.Paths.ensure`, which refuses tracked
    prefixes.  It is the only writer of that file.

    Args:
        assignment: Mapping ``patient_id -> fold``.
        path: Destination CSV, created along with its parent directory.

    Returns:
        ``path``.

    Raises:
        ValueError: If ``assignment`` is empty.
    """
    if not assignment:
        raise ValueError("refusing to write an empty folds.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FOLDS_CSV_HEADER)
        for pid in sorted(assignment):
            writer.writerow([pid, assignment[pid]])
    return path


def read_folds_csv(path: Path) -> dict[str, int]:
    """Read a ``patient_id,fold`` CSV.

    Args:
        path: The CSV to read.

    Returns:
        Mapping ``patient_id -> fold``.

    Raises:
        ValueError: If the header is not ``patient_id,fold``, a fold index is not an
            integer, or a patient appears twice.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
        if header != FOLDS_CSV_HEADER:
            raise ValueError(
                f"{path} has header {header}, expected {FOLDS_CSV_HEADER}"
            )
        assignment: dict[str, int] = {}
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{path}:{line_number} has {len(row)} fields, expected 2"
                )
            pid, raw_fold = row[0].strip(), row[1].strip()
            try:
                fold = int(raw_fold)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} fold {raw_fold!r} is not an integer"
                ) from exc
            if pid in assignment:
                raise ValueError(f"{path}:{line_number} duplicates patient {pid!r}")
            assignment[pid] = fold
    return assignment


# --------------------------------------------------------------------------- #
# Patient enumeration from the committed predictions
# --------------------------------------------------------------------------- #

def patient_id_from_segment_name(name: str) -> str:
    """Extract the CirCor patient id from a segment filename.

    Segment names look like ``50043_AV.wav`` or ``50043_AV-2.wav``: the patient id is
    everything before the first underscore.  These identifiers are CirCor's own public
    pseudonyms and carry no personal information.

    Args:
        name: Segment filename as recorded in a prediction dump.

    Returns:
        The patient id.

    Raises:
        ValueError: If ``name`` has no underscore, so no patient id can be read.
    """
    head, sep, _ = name.strip().partition("_")
    if not sep or not head:
        raise ValueError(
            f"segment name {name!r} is not of the form <patient_id>_<location>.wav"
        )
    return head


def _fold_prediction_files(preds_dir: Path) -> list[tuple[Path, int]]:
    """Return the ``*_fold<N>.data.csv`` dumps and their held-out fold index."""
    files = []
    for path in sorted(preds_dir.glob("*.data.csv")):
        match = _FOLD_FILE_RE.search(path.name)
        if match is not None:
            files.append((path, int(match.group(1))))
    return files


def _patients_in_prediction_file(path: Path) -> set[str]:
    """Read the patient ids present in one prediction dump.

    The dumps are headerless ``logit0,logit1,logit2,target,filename`` rows.
    """
    patients: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{path}:{line_number} has {len(row)} fields; expected the "
                    "logit0,logit1,logit2,target,filename schema"
                )
            patients.add(patient_id_from_segment_name(row[-1]))
    return patients


def patients_from_predictions(preds_dir: Path) -> tuple[str, ...]:
    """Enumerate every patient appearing in the committed prediction dumps.

    This is the download-free way to reconstruct the split: the union over the five
    held-out folds is exactly the 941 analysed patients.

    Args:
        preds_dir: Directory of ``*.data.csv`` dumps (``results/predictions``).

    Returns:
        Sorted patient ids.

    Raises:
        FileNotFoundError: If no ``*_fold<N>.data.csv`` file is present.
    """
    files = _fold_prediction_files(preds_dir)
    if not files:
        raise FileNotFoundError(
            f"no *_fold<N>.data.csv prediction dumps in {preds_dir}; the split can "
            "only be reconstructed from the committed predictions or from the CirCor "
            "download (--source audio)"
        )
    patients: set[str] = set()
    for path, _ in files:
        patients |= _patients_in_prediction_file(path)
    return tuple(sorted(patients))


def observed_partition(preds_dir: Path) -> dict[str, int]:
    """Recover the patient/fold partition the prediction dumps actually encode.

    Each ``*_fold<N>.data.csv`` file holds the predictions for held-out fold ``N``, so
    the patients it mentions belong to fold ``N``.  Files from different models and
    from the 2-class task all share one split, which makes this a genuine consistency
    check rather than a restatement.

    Args:
        preds_dir: Directory of ``*.data.csv`` dumps.

    Returns:
        Mapping ``patient_id -> fold``, ordered by patient id.

    Raises:
        FileNotFoundError: If no fold dumps are present.
        ValueError: If two dumps place the same patient in different folds.
    """
    files = _fold_prediction_files(preds_dir)
    if not files:
        raise FileNotFoundError(
            f"no *_fold<N>.data.csv prediction dumps in {preds_dir}"
        )
    partition: dict[str, int] = {}
    evidence: dict[str, str] = {}
    for path, fold in files:
        for pid in _patients_in_prediction_file(path):
            previous = partition.get(pid)
            if previous is None:
                partition[pid] = fold
                evidence[pid] = path.name
            elif previous != fold:
                raise ValueError(
                    f"patient {pid!r} appears in fold {previous} "
                    f"({evidence[pid]}) and fold {fold} ({path.name}); the prediction "
                    "dumps do not describe a single partition"
                )
    return {pid: partition[pid] for pid in sorted(partition)}


def compare_partitions(
    expected: Mapping[str, int], actual: Mapping[str, int]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, int, int], ...]]:
    """Diff two patient/fold partitions.

    Args:
        expected: Reference partition.
        actual: Partition under test.

    Returns:
        ``(missing, extra, disagreements)`` where ``missing`` are patients only in
        ``expected``, ``extra`` only in ``actual``, and ``disagreements`` are
        ``(patient_id, expected_fold, actual_fold)`` triples.
    """
    missing = tuple(sorted(set(expected) - set(actual)))
    extra = tuple(sorted(set(actual) - set(expected)))
    disagreements = tuple(
        (pid, expected[pid], actual[pid])
        for pid in sorted(set(expected) & set(actual))
        if expected[pid] != actual[pid]
    )
    return missing, extra, disagreements


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _patients_from_audio(resolved: paths.Paths) -> tuple[str, ...]:
    """Enumerate the patients that survive the loader, from the CirCor download.

    :mod:`mirasol.data.prepare` depends on this module for the fold recipe, so the
    import is local to break the cycle rather than to make it optional.
    """
    from mirasol.data.prepare import surviving_patient_ids

    return surviving_patient_ids(
        audio_dir=resolved.require_circor_audio(),
        meta_csv=resolved.require_circor_meta(),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m mirasol.data.splits`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m mirasol.data.splits",
        description=(
            "Recompute the published patient-level five-fold split and write or "
            "verify data/splits/folds.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("predictions", "audio"),
        default="predictions",
        help=(
            "where to enumerate patients from: the committed prediction dumps (no "
            "download needed) or the CirCor audio directory"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="destination CSV (default: <splits-dir>/folds.csv)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the CSV instead of only printing the fold counts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compare the recipe against the existing CSV and, for --source "
            "predictions, against the partition the dumps encode; exit 1 on any "
            "mismatch"
        ),
    )
    parser.add_argument("--n-folds", type=int, default=N_FOLDS, help="number of folds")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="shuffle seed")
    parser.add_argument(
        "--expect-patients",
        type=int,
        default=EXPECTED_PATIENT_COUNT,
        help="fail if a different number of patients is found; 0 disables the check",
    )
    paths.add_path_arguments(
        parser, include=("circor_root", "circor_meta", "audio_dir", "preds_dir", "splits_dir")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m mirasol.data.splits``.

    Returns:
        ``0`` on success, ``1`` if a requested check failed.
    """
    args = _build_parser().parse_args(argv)
    resolved = paths.paths_from_args(args)
    out_path = args.out if args.out is not None else resolved.folds_csv

    if args.source == "predictions":
        preds_dir = resolved.require_preds_dir()
        patient_ids = patients_from_predictions(preds_dir)
        print(f"patients from {preds_dir}: {len(patient_ids)}")
    else:
        patient_ids = _patients_from_audio(resolved)
        print(f"patients from {resolved.circor_audio}: {len(patient_ids)}")

    if args.expect_patients and len(patient_ids) != args.expect_patients:
        print(
            f"ERROR: found {len(patient_ids)} patients, expected "
            f"{args.expect_patients}. Pass --expect-patients 0 to override.",
            file=sys.stderr,
        )
        return 1

    assignment = fold_assignment(patient_ids, n_folds=args.n_folds, seed=args.seed)
    counts = fold_counts(assignment, n_folds=args.n_folds)
    print(
        f"recipe: sorted -> Random({args.seed}).shuffle -> i % {args.n_folds}\n"
        f"fold sizes: {' / '.join(str(c) for c in counts)} = {sum(counts)}"
    )

    failed = False
    if args.check:
        if out_path.exists():
            stored = read_folds_csv(out_path)
            missing, extra, disagreements = compare_partitions(stored, assignment)
            print(
                f"{out_path}: {len(stored)} rows, {len(missing)} missing, "
                f"{len(extra)} extra, {len(disagreements)} fold disagreements"
            )
            if missing or extra or disagreements:
                failed = True
                for pid, want, got in disagreements[:10]:
                    print(f"  {pid}: csv fold {want} vs recipe fold {got}")
        else:
            print(f"ERROR: {out_path} does not exist", file=sys.stderr)
            failed = True

        if args.source == "predictions":
            encoded = observed_partition(resolved.preds_dir)
            missing, extra, disagreements = compare_partitions(encoded, assignment)
            print(
                f"predictions: {len(encoded)} patients, {len(missing)} missing, "
                f"{len(extra)} extra, {len(disagreements)} fold disagreements"
            )
            if missing or extra or disagreements:
                failed = True
                for pid, want, got in disagreements[:10]:
                    print(f"  {pid}: predictions fold {want} vs recipe fold {got}")

    if args.write:
        if failed:
            print("refusing to write after a failed check", file=sys.stderr)
            return 1
        write_folds_csv(assignment, out_path)
        print(f"wrote {out_path} ({len(assignment)} rows)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
