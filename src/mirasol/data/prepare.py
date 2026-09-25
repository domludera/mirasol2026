# SPDX-License-Identifier: MIT
"""Build the prepared fold file the training pipeline reads, from the CirCor download.

One script for all three published dataset variants, replacing the two near-identical
copies (``src/prepare_data.py`` and ``src/prepare_data_ablation.py``) that differed
mainly in their label map:

- ``--num-classes 3`` (default): Present=0, Unknown=1, Absent=2. This is the paper's
  main protocol - Table 1, Table 2, Tables S1-S6, S8, S10.
- ``--num-classes 2``: Present or Unknown = 0, Absent = 1. Unknown is merged into
  Present because a screening tool should refer an ambiguous case, not dismiss it.
  This is Table S7.
- The ablation flags ``--single-location``, ``--no-bandpass``, ``--bandpass-low``,
  ``--bandpass-high`` and ``--clip-seconds`` reproduce the Table S10 rows.

Pipeline, per recording (all of it lives in :mod:`mirasol.signal.mel`, so the numpy
edge implementation cannot drift from it): load at 4 kHz, causal 4th-order Butterworth
bandpass 25-400 Hz, per-recording z-score, split into non-overlapping 30 s windows
with a short tail tiled rather than zero-padded.  The self-concatenation that turns a
30 s window into the models' 2001 mel frames happens later, in
:class:`mirasol.data.dataset.ESC_Dataset`.

Why n = 941 and not 942
-----------------------
Recordings are matched by the exact name ``{patient_id}_{location}.wav`` built from the
metadata's ``Recording locations:`` column.  CirCor stores repeat recordings of one
location as ``{patient_id}_{location}_1.wav`` / ``_2.wav``, which that rule does not
match.  The behaviour is kept **as is**, because it is what produced the published
n = 941, but every run now prints the full accounting: the unmatched files, the
metadata entries with no exact WAV (found by globbing ``{patient_id}_{location}*.wav``),
and any patient that loses all of its recordings.  On the authors' copy of the dataset
that is 43 duplicate-location files over 17 patients, plus 3 non-CirCor files that had
been dropped into the directory by hand, and patient 50321 - whose four recordings are
all ``50321_AV_1/_2`` and ``50321_MV_1/_2`` - is the single patient excluded outright.

Command line::

    python -m mirasol.data.prepare --num-classes 3            # train-circor-data-30s.npy
    python -m mirasol.data.prepare --num-classes 2            # ...-30s-binary.npy
    python -m mirasol.data.prepare --clip-seconds 10          # Table S10 window ablation
    python -m mirasol.data.prepare --single-location          # Table S10 single location
    python -m mirasol.data.prepare --no-bandpass              # Table S10 no filter
    python -m mirasol.data.prepare --bandpass-high 1000       # Table S10 wide filter

Paper artifacts served: the input to every trained model, and Table S10's five
ablation dataset variants.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from mirasol import paths
from mirasol.data.splits import N_FOLDS, SPLIT_SEED, shuffled_patients
from mirasol.signal.mel import (
    BandpassConfig,
    bandpass_filter,
    segment_waveform,
    zscore_normalise,
)

#: Class names by ``--num-classes``, index = integer target.
CLASS_NAMES: Final[Mapping[int, tuple[str, ...]]] = {
    3: ("Present", "Unknown", "Absent"),
    2: ("Present", "Absent"),
}

#: Murmur status -> integer target, per ``--num-classes``.
_TARGETS: Final[Mapping[int, Mapping[str, int]]] = {
    3: {"Present": 0, "Unknown": 1, "Absent": 2},
    2: {"Present": 0, "Unknown": 0, "Absent": 1},
}

#: Metadata columns this script reads.
COLUMN_PATIENT_ID: Final[str] = "Patient ID"
COLUMN_RECORDING_LOCATIONS: Final[str] = "Recording locations:"
COLUMN_MURMUR: Final[str] = "Murmur"
COLUMN_MURMUR_LOCATIONS: Final[str] = "Murmur locations"
COLUMN_MOST_AUDIBLE: Final[str] = "Most audible location"

DEFAULT_SAMPLE_RATE: Final[int] = 4000
DEFAULT_CLIP_SECONDS: Final[int] = 30

__all__ = [
    "CLASS_NAMES",
    "LoaderReport",
    "PatientInfo",
    "Recording",
    "audit_recordings",
    "filter_single_location",
    "main",
    "murmur_target",
    "output_name_for",
    "parse_metadata",
    "surviving_patient_ids",
]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Recording:
    """One expected WAV file: which patient and location it belongs to."""

    patient: str
    location: str
    target: int


@dataclass(frozen=True)
class PatientInfo:
    """Patient-level metadata needed by the single-location ablation."""

    murmur_status: str
    murmur_locations: tuple[str, ...]
    most_audible: str | None
    rec_locations: tuple[str, ...]
    target: int


def murmur_target(murmur_status: str, num_classes: int) -> int:
    """Map a CirCor ``Murmur`` value to an integer target.

    Args:
        murmur_status: ``"Present"``, ``"Unknown"`` or ``"Absent"``.
        num_classes: 3 for the main protocol, 2 for Table S7.

    Returns:
        The integer class index.

    Raises:
        ValueError: If ``num_classes`` is unsupported, or ``murmur_status`` is not one
            of the three documented values.  The original code silently mapped
            anything unrecognised to Absent, which would have quietly accepted the
            contradictory murmur encoding used by the ``train_meta_data_backup/_old``
            variants of the CSV.
    """
    try:
        mapping = _TARGETS[num_classes]
    except KeyError as exc:
        raise ValueError(
            f"num_classes must be 2 or 3, got {num_classes}"
        ) from exc
    try:
        return mapping[murmur_status]
    except KeyError as exc:
        raise ValueError(
            f"unexpected Murmur value {murmur_status!r}; expected one of "
            f"{sorted(mapping)}. Use the official training_data.csv from PhysioNet."
        ) from exc


def _split_locations(raw: str) -> tuple[str, ...]:
    """Split a ``+``-separated location list, dropping blanks."""
    return tuple(part.strip() for part in raw.split("+") if part.strip())


def parse_metadata(
    csv_path: Path, num_classes: int = 3
) -> tuple[dict[str, Recording], dict[str, PatientInfo]]:
    """Parse CirCor's ``training_data.csv``.

    Args:
        csv_path: Path to ``training_data.csv`` (116,962 B; not redistributed here).
        num_classes: 3 or 2, selecting the label map.

    Returns:
        ``(recordings, patients)`` where ``recordings`` maps the expected WAV filename
        ``{patient_id}_{location}.wav`` to a :class:`Recording`, and ``patients`` maps
        the patient id to a :class:`PatientInfo`.

    Raises:
        ValueError: If a required column is missing, a patient id repeats, or a Murmur
            value is not recognised.
    """
    recordings: dict[str, Recording] = {}
    patients: dict[str, PatientInfo] = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {COLUMN_PATIENT_ID, COLUMN_RECORDING_LOCATIONS, COLUMN_MURMUR}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): {sorted(missing)}"
            )
        for row in reader:
            patient_id = row[COLUMN_PATIENT_ID].strip()
            if patient_id in patients:
                raise ValueError(f"{csv_path} lists patient {patient_id!r} twice")
            rec_locations = _split_locations(row[COLUMN_RECORDING_LOCATIONS])
            murmur_status = row[COLUMN_MURMUR].strip()
            target = murmur_target(murmur_status, num_classes)

            most_audible = (row.get(COLUMN_MOST_AUDIBLE) or "").strip()
            murmur_locations_raw = (row.get(COLUMN_MURMUR_LOCATIONS) or "").strip()
            patients[patient_id] = PatientInfo(
                murmur_status=murmur_status,
                murmur_locations=(
                    _split_locations(murmur_locations_raw)
                    if murmur_locations_raw and murmur_locations_raw != "nan"
                    else ()
                ),
                most_audible=most_audible if most_audible not in ("", "nan") else None,
                rec_locations=rec_locations,
                target=target,
            )
            for location in rec_locations:
                recordings[f"{patient_id}_{location}.wav"] = Recording(
                    patient=patient_id, location=location, target=target
                )

    return recordings, patients


def filter_single_location(
    recordings: Mapping[str, Recording], patients: Mapping[str, PatientInfo]
) -> dict[str, Recording]:
    """Keep one recording per patient (Table S10's single-location ablation).

    Present patients contribute their most audible location - falling back to the
    first annotated murmur location, then the first recorded location - so the
    ablation does not accidentally discard the audible murmur.  Unknown and Absent
    patients contribute their first recorded location.

    Args:
        recordings: All matched recordings, keyed by WAV filename.
        patients: Patient-level metadata.

    Returns:
        The selected subset of ``recordings``.
    """
    selected: dict[str, Recording] = {}
    for patient_id, info in patients.items():
        if info.murmur_status == "Present":
            chosen = (
                info.most_audible
                or (info.murmur_locations[0] if info.murmur_locations else None)
                or (info.rec_locations[0] if info.rec_locations else None)
            )
        else:
            chosen = info.rec_locations[0] if info.rec_locations else None

        wav_name = f"{patient_id}_{chosen}.wav" if chosen else ""
        if wav_name in recordings:
            selected[wav_name] = recordings[wav_name]
            continue
        # Fall back to the first recording of this patient that exists at all.
        for location in info.rec_locations:
            fallback = f"{patient_id}_{location}.wav"
            if fallback in recordings:
                selected[fallback] = recordings[fallback]
                break
    return selected


# --------------------------------------------------------------------------- #
# Loader audit: the 942 -> 941 accounting
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LoaderReport:
    """What the ``{patient_id}_{location}.wav`` matching rule kept and dropped."""

    metadata_patients: int
    expected_recordings: int
    wav_files_on_disk: int
    matched: tuple[str, ...]
    duplicate_location_files: tuple[str, ...]
    foreign_files: tuple[str, ...]
    metadata_without_exact_wav: tuple[str, ...]
    variants_found: Mapping[str, tuple[str, ...]]
    patients_kept: tuple[str, ...]
    patients_dropped: tuple[str, ...]

    def render(self) -> str:
        """Return the ASCII-only accounting block printed at the end of every run."""
        affected = sorted(
            {name.split("_", 1)[0] for name in self.metadata_without_exact_wav}
        )
        lines = [
            "Recording accounting (the reason n = "
            f"{len(self.patients_kept)} and not {self.metadata_patients}):",
            f"  metadata patients                 : {self.metadata_patients}",
            f"  metadata recordings expected      : {self.expected_recordings}",
            f"  wav files in the audio directory  : {self.wav_files_on_disk}",
            f"  matched {{pid}}_{{loc}}.wav exactly    : {len(self.matched)}",
            f"  duplicate-location wav files      : "
            f"{len(self.duplicate_location_files)} (named {{pid}}_{{loc}}_N.wav)",
            f"  non-CirCor wav files ignored      : {len(self.foreign_files)}",
            f"  metadata entries with no exact wav: "
            f"{len(self.metadata_without_exact_wav)} over {len(affected)} patient(s)",
            f"  patients kept                     : {len(self.patients_kept)}",
            f"  patients dropped entirely         : "
            f"{len(self.patients_dropped)}"
            + (f" ({', '.join(self.patients_dropped)})" if self.patients_dropped else ""),
        ]
        for expected in self.metadata_without_exact_wav:
            found = self.variants_found.get(expected, ())
            lines.append(
                f"    missing {expected} ; on disk instead: "
                + (", ".join(found) if found else "nothing")
            )
        for name in self.foreign_files:
            lines.append(f"    ignored (no metadata entry): {name}")
        return "\n".join(lines)


def audit_recordings(
    audio_dir: Path,
    recordings: Mapping[str, Recording],
    patients: Mapping[str, PatientInfo],
) -> LoaderReport:
    """Compare the metadata's expected filenames against what is on disk.

    Args:
        audio_dir: Directory of CirCor ``*.wav`` files.
        recordings: Expected recordings, keyed by ``{patient_id}_{location}.wav``.
        patients: Patient-level metadata, used to tell CirCor files from strays.

    Returns:
        A :class:`LoaderReport`.
    """
    on_disk = sorted(path.name for path in audio_dir.glob("*.wav"))
    on_disk_set = set(on_disk)

    matched = tuple(name for name in on_disk if name in recordings)
    unmatched = [name for name in on_disk if name not in recordings]
    foreign = tuple(
        name for name in unmatched if name.split("_", 1)[0] not in patients
    )
    duplicate_location = tuple(name for name in unmatched if name not in foreign)

    missing = tuple(sorted(name for name in recordings if name not in on_disk_set))
    variants: dict[str, tuple[str, ...]] = {}
    for name in missing:
        stem = name.removesuffix(".wav")
        variants[name] = tuple(
            sorted(path.name for path in audio_dir.glob(f"{stem}*.wav"))
        )

    kept = sorted({recordings[name].patient for name in matched})
    dropped = tuple(sorted(set(patients) - set(kept)))
    return LoaderReport(
        metadata_patients=len(patients),
        expected_recordings=len(recordings),
        wav_files_on_disk=len(on_disk),
        matched=matched,
        duplicate_location_files=duplicate_location,
        foreign_files=foreign,
        metadata_without_exact_wav=missing,
        variants_found=variants,
        patients_kept=tuple(kept),
        patients_dropped=dropped,
    )


def surviving_patient_ids(audio_dir: Path, meta_csv: Path) -> tuple[str, ...]:
    """Return the patients that survive the loader, sorted.

    This is the audio-derived source for :mod:`mirasol.data.splits`; it reads the
    metadata and lists the directory but decodes no audio, so it needs neither librosa
    nor the whole corpus to be readable at speed.

    Args:
        audio_dir: Directory of CirCor ``*.wav`` files.
        meta_csv: Path to ``training_data.csv``.

    Returns:
        Sorted patient ids, 941 for the full dataset.
    """
    recordings, patients = parse_metadata(meta_csv)
    return audit_recordings(audio_dir, recordings, patients).patients_kept


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PrepareSummary:
    """Outcome of one preparation run."""

    output_path: Path
    num_classes: int
    clip_samples: int
    fold_sizes: tuple[int, ...]
    class_counts: tuple[int, ...]
    report: LoaderReport


def output_name_for(num_classes: int, clip_seconds: int) -> str:
    """Return the conventional ``.npy`` filename for a variant.

    The 3-class 30 s name is ``train-circor-data-30s.npy`` and the 2-class one is
    ``train-circor-data-30s-binary.npy``; those are the names the experiment driver's
    task presets expect, so they are derived here rather than typed twice.
    """
    suffix = "-binary" if num_classes == 2 else ""
    return f"train-circor-data-{clip_seconds}s{suffix}.npy"


def _load_waveform(wav_path: Path, sample_rate: int) -> np.ndarray:
    """Read a WAV file as mono float32 at ``sample_rate``.

    librosa is imported here rather than at module scope so that the metadata-only
    helpers (:func:`surviving_patient_ids`, used by ``mirasol.data.splits``) work with
    the torch-free ``[analysis]`` install.
    """
    import librosa

    audio, _ = librosa.load(str(wav_path), sr=sample_rate)
    return audio


def _prepare_segments(
    audio_dir: Path,
    recordings: Mapping[str, Recording],
    *,
    sample_rate: int,
    clip_samples: int,
    bandpass: BandpassConfig | None,
    progress_every: int = 200,
) -> dict[str, list[dict[str, object]]]:
    """Turn every matched recording into windowed, normalised segments.

    Returns:
        ``patient_id -> [segment, ...]`` in sorted-filename order, where each segment
        is the ``{name, target, waveform}`` dict the fold file stores.
    """
    wav_files = sorted(audio_dir.glob("*.wav"))
    by_patient: dict[str, list[dict[str, object]]] = defaultdict(list)
    processed = 0

    for index, wav_path in enumerate(wav_files, start=1):
        recording = recordings.get(wav_path.name)
        if recording is None:
            continue

        audio = _load_waveform(wav_path, sample_rate)
        if bandpass is not None:
            audio = bandpass_filter(audio, sample_rate, bandpass)
        audio = zscore_normalise(audio)
        windows = segment_waveform(audio, clip_samples)

        stem = wav_path.name.removesuffix(".wav")
        for window_index, window in enumerate(windows):
            name = (
                f"{stem}-{window_index}.wav" if len(windows) > 1 else wav_path.name
            )
            by_patient[recording.patient].append(
                {
                    "name": name,
                    "target": recording.target,
                    "waveform": window.astype(np.float32),
                }
            )
        processed += 1
        if index % progress_every == 0 or index == len(wav_files):
            print(f"       {index}/{len(wav_files)} files scanned ...")

    print(f"       matched and processed {processed} recordings")
    return dict(by_patient)


def _deal_folds(
    by_patient: Mapping[str, Sequence[dict[str, object]]],
    n_folds: int,
    seed: int,
) -> list[list[dict[str, object]]]:
    """Deal patients into folds with the canonical recipe, keeping segments together.

    Uses :func:`mirasol.data.splits.shuffled_patients` so the fold recipe exists in
    exactly one place, and extends each fold in shuffled-patient order, which is what
    fixes the segment order inside the released ``.npy`` files.
    """
    folds: list[list[dict[str, object]]] = [[] for _ in range(n_folds)]
    for index, patient_id in enumerate(shuffled_patients(by_patient, seed=seed)):
        folds[index % n_folds].extend(by_patient[patient_id])
    return folds


def _save_folds(folds: Sequence[Sequence[dict[str, object]]], out_path: Path) -> None:
    """Write the ragged fold structure as a ``(n_folds,)`` object array.

    Built element by element rather than with ``np.array(folds, dtype=object)``, which
    collapses to a 2-D array whenever the folds happen to be equal length.
    """
    array = np.empty(len(folds), dtype=object)
    for index, fold in enumerate(folds):
        array[index] = list(fold)
    np.save(str(out_path), array, allow_pickle=True)


def run_pipeline(args: argparse.Namespace, resolved: paths.Paths) -> PrepareSummary:
    """Run the full preparation pipeline and write the fold file.

    Args:
        args: Parsed command-line arguments.
        resolved: Resolved repository paths.

    Returns:
        A :class:`PrepareSummary`.

    Raises:
        ValueError: If no recording could be matched, so the output would be empty.
    """
    audio_dir = resolved.require_circor_audio()
    meta_csv = resolved.require_circor_meta()
    sample_rate = args.sample_rate
    clip_samples = sample_rate * args.clip_seconds
    class_names = CLASS_NAMES[args.num_classes]

    output_dir = resolved.ensure(
        args.output_dir if args.output_dir is not None else resolved.circor_root
    )
    out_path = output_dir / (
        args.output_name or output_name_for(args.num_classes, args.clip_seconds)
    )

    print(f"[1/4] Parsing metadata from {meta_csv} ...")
    recordings, patients = parse_metadata(meta_csv, args.num_classes)
    print(f"       {len(recordings)} recordings, {len(patients)} patients in metadata.")

    report = audit_recordings(audio_dir, recordings, patients)
    if args.single_location:
        recordings = filter_single_location(recordings, patients)
        print(f"       [ABLATION] single-location: {len(recordings)} recordings kept.")

    bandpass = (
        None
        if args.no_bandpass
        else BandpassConfig(low=args.bandpass_low, high=args.bandpass_high)
    )
    filter_desc = (
        "NONE" if bandpass is None else f"{bandpass.low}-{bandpass.high} Hz order 4"
    )
    print(
        f"[2/4] Processing WAVs from {audio_dir} (sr={sample_rate}, "
        f"clip={args.clip_seconds}s, filter={filter_desc}) ..."
    )
    by_patient = _prepare_segments(
        audio_dir,
        recordings,
        sample_rate=sample_rate,
        clip_samples=clip_samples,
        bandpass=bandpass,
    )
    if not by_patient:
        raise ValueError(
            f"no WAV in {audio_dir} matched a metadata entry; check --audio-dir and "
            "--circor-meta point at the same CirCor download."
        )

    class_counts = [0] * args.num_classes
    for segments in by_patient.values():
        for segment in segments:
            class_counts[int(segment["target"])] += 1
    total_segments = sum(class_counts)
    print(
        f"       {total_segments} segments; class balance: "
        + ", ".join(
            f"{count} {name}" for name, count in zip(class_names, class_counts, strict=True)
        )
    )

    print(f"[3/4] Splitting {len(by_patient)} patients into {args.n_folds} folds ...")
    folds = _deal_folds(by_patient, args.n_folds, args.seed)
    for index, fold in enumerate(folds):
        per_class = [0] * args.num_classes
        for segment in fold:
            per_class[int(segment["target"])] += 1
        print(
            f"       Fold {index}: {len(fold):>5} segments  ("
            + ", ".join(
                f"{name}={count}" for name, count in zip(class_names, per_class, strict=True)
            )
            + ")"
        )

    print(f"[4/4] Saving to {out_path} ...")
    _save_folds(folds, out_path)
    loaded = np.load(str(out_path), allow_pickle=True)
    print(
        f"       saved OK: folds={loaded.shape[0]}, "
        f"total_segments={sum(len(fold) for fold in loaded)}, "
        f"waveform_len={len(loaded[0][0]['waveform'])}"
    )

    print(report.render())
    return PrepareSummary(
        output_path=out_path,
        num_classes=args.num_classes,
        clip_samples=clip_samples,
        fold_sizes=tuple(len(fold) for fold in folds),
        class_counts=tuple(class_counts),
        report=report,
    )


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m mirasol.data.prepare`` argument parser.

    Both spellings of every multi-word flag are accepted (``--num-classes`` and
    ``--num_classes``), because the underscore forms appear in the shipped run logs.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mirasol.data.prepare",
        description="Prepare the CirCor DigiScope dataset for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-classes", "--num_classes", dest="num_classes", type=int,
        choices=sorted(CLASS_NAMES), default=3,
        help="3 = Present/Unknown/Absent (main protocol), 2 = Present|Unknown/Absent",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, default=None,
        help="where to write the .npy file (default: the CirCor root)",
    )
    parser.add_argument(
        "--output-name", "--output_name", dest="output_name", default=None,
        help="output filename (default: derived from --num-classes/--clip-seconds)",
    )
    parser.add_argument(
        "--sample-rate", "--sample_rate", dest="sample_rate", type=int,
        default=DEFAULT_SAMPLE_RATE, help="target sample rate in Hz (4000 = native)",
    )
    parser.add_argument(
        "--clip-seconds", "--clip_seconds", dest="clip_seconds", type=int,
        default=DEFAULT_CLIP_SECONDS, help="window length in seconds (30, 15 or 10)",
    )
    parser.add_argument(
        "--n-folds", "--n_folds", dest="n_folds", type=int, default=N_FOLDS,
        help="number of patient-level folds",
    )
    parser.add_argument(
        "--seed", type=int, default=SPLIT_SEED,
        help="patient shuffle seed; 42 for every published run",
    )

    ablation = parser.add_argument_group("Table S10 ablations")
    ablation.add_argument(
        "--single-location", "--single_location", dest="single_location",
        action="store_true", help="keep only one recording per patient",
    )
    ablation.add_argument(
        "--no-bandpass", "--no_bandpass", dest="no_bandpass",
        action="store_true", help="disable the bandpass filter entirely",
    )
    ablation.add_argument(
        "--bandpass-low", "--bandpass_low", dest="bandpass_low", type=float,
        default=BandpassConfig().low, help="bandpass low cutoff in Hz",
    )
    ablation.add_argument(
        "--bandpass-high", "--bandpass_high", dest="bandpass_high", type=float,
        default=BandpassConfig().high, help="bandpass high cutoff in Hz",
    )

    paths.add_path_arguments(
        parser, include=("circor_root", "circor_meta", "audio_dir")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m mirasol.data.prepare`` and ``mirasol-prepare``.

    Returns:
        ``0`` on success; ``1`` with a message on stderr if a required input is
        missing or the metadata and audio directory disagree completely.
    """
    args = _build_parser().parse_args(argv)
    if args.clip_seconds <= 0:
        print("ERROR: --clip-seconds must be positive", file=sys.stderr)
        return 1
    try:
        run_pipeline(args, paths.paths_from_args(args))
    except (paths.MissingInputError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
