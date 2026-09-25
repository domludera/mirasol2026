#!/usr/bin/env python3
"""Unified audio preparation pipeline for the CIRCOr dataset.

Reads raw WAV files + metadata CSV, applies bandpass filtering, z-score
normalisation, and fixed-length windowing (30 s in the paper), then writes patient-level k-fold
splits as a single .npy file consumable by ESC_Dataset.

Usage
-----
    python -m src.prepare_data \
        --raw_dir    data/circor/training_data \
        --csv_file   data/circor/training_data.csv \
        --output_dir data/circor \
        --sample_rate 4000 \
        --clip_seconds 30 \
        --n_folds 5 \
        --task 3class            # or: binary
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from scipy import signal, stats


# ── helpers ──────────────────────────────────────────────────────────────────

def bandpass_filter(audio: np.ndarray, sr: int,
                    low: float = 25.0, high: float = 400.0,
                    order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, audio)


def zscore_normalise(audio: np.ndarray) -> np.ndarray:
    """Z-score normalise (mean=0, std=1). Returns zeros if std is 0."""
    std = np.std(audio)
    if std < 1e-8:
        return np.zeros_like(audio)
    return (audio - np.mean(audio)) / std


def window_to_clips(audio: np.ndarray, clip_len: int) -> list[np.ndarray]:
    """Split *audio* into fixed-length clips of *clip_len* samples.

    Longer audio  → non-overlapping segments; a short tail is padded by
                    repeating itself.
    Shorter audio → tiled to fill one full clip.
    """
    clips = []
    if len(audio) <= clip_len:
        # Repeat-pad to fill one clip
        tiled = np.tile(audio, (clip_len // len(audio)) + 1)
        clips.append(tiled[:clip_len])
    else:
        # Segment into non-overlapping clips
        for start in range(0, len(audio), clip_len):
            chunk = audio[start : start + clip_len]
            if len(chunk) < clip_len:
                # Pad the tail by repeating itself
                tiled = np.tile(chunk, (clip_len // len(chunk)) + 1)
                chunk = tiled[:clip_len]
            clips.append(chunk)
    return clips


# ── metadata ─────────────────────────────────────────────────────────────────

# Patient-level "Murmur" column -> class index, per task.
# binary merges Unknown into Present: a screening tool should flag ambiguous
# cases for referral, not dismiss them.
LABEL_MAPS = {
    "3class": {"Present": 0, "Unknown": 1, "Absent": 2},
    "binary": {"Present": 0, "Unknown": 0, "Absent": 1},
}
CLASS_NAMES = {
    "3class": ["Present", "Unknown", "Absent"],
    "binary": ["Present", "Absent"],
}


def parse_metadata(csv_path: str, task: str = "3class") -> dict[str, dict]:
    """Parse the CIRCOr training_data.csv and return per-recording metadata.

    Returns a dict keyed by WAV filename (e.g. ``"2530_AV.wav"``) with values
    ``{"patient": str, "location": str, "murmur": int}``, where *murmur* is the
    patient's class index under ``LABEL_MAPS[task]``.
    """
    label_map = LABEL_MAPS[task]
    records: dict[str, dict] = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            patient_id = row["Patient ID"]
            rec_locs = row["Recording locations:"].split("+")
            target = label_map[row["Murmur"]]

            for loc in rec_locs:
                loc = loc.strip()
                if not loc:
                    continue
                wav_name = f"{patient_id}_{loc}.wav"

                records[wav_name] = {
                    "patient": patient_id,
                    "location": loc,
                    "murmur": target,
                }

    return records


# ── fold splitting ───────────────────────────────────────────────────────────

def patient_level_folds(samples: list[dict], n_folds: int,
                        seed: int = 42) -> list[list[dict]]:
    """Split *samples* into *n_folds* folds at the patient level.

    All recordings belonging to the same patient are kept together so there is
    no data leakage between folds.
    """
    # Group samples by patient
    patient_map: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        patient_map[s["patient"]].append(s)

    # Deterministic shuffle of patients
    patients = sorted(patient_map.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    # Round-robin assignment
    folds: list[list[dict]] = [[] for _ in range(n_folds)]
    for i, pid in enumerate(patients):
        folds[i % n_folds].extend(patient_map[pid])

    return folds


# ── main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)
    sr = args.sample_rate
    clip_len = sr * args.clip_seconds

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Parse metadata
    print(f"[1/4] Parsing metadata from {csv_path} ...")
    meta = parse_metadata(str(csv_path), args.task)
    print(f"       Found {len(meta)} recordings in CSV.")

    # 2) Process each WAV
    print(f"[2/4] Processing WAVs from {raw_dir} (sr={sr}, clip={args.clip_seconds}s) ...")
    all_samples: list[dict] = []
    skipped = 0
    processed = 0

    wav_files = sorted(raw_dir.glob("*.wav"))
    total = len(wav_files)

    for idx, wav_path in enumerate(wav_files, 1):
        wav_name = wav_path.name
        if wav_name not in meta:
            skipped += 1
            continue

        info = meta[wav_name]

        # Load at native sample rate
        audio, file_sr = librosa.load(str(wav_path), sr=sr)

        # Bandpass filter
        audio = bandpass_filter(audio, sr)

        # Z-score normalise
        audio = zscore_normalise(audio)

        # Window into fixed-length clips
        clips = window_to_clips(audio, clip_len)

        for ci, clip in enumerate(clips):
            seg_name = f"{wav_name.removesuffix('.wav')}-{ci}.wav" if len(clips) > 1 else wav_name
            all_samples.append({
                "name": seg_name,
                "target": info["murmur"],
                "waveform": clip.astype(np.float32),
                "patient": info["patient"],
            })

        processed += 1
        if idx % 200 == 0 or idx == total:
            print(f"       {idx}/{total} files processed ...")

    print(f"       Done. {processed} files → {len(all_samples)} segments "
          f"({skipped} skipped, no metadata).")

    # 3) Count class balance
    names = CLASS_NAMES[args.task]

    def balance(samples: list[dict]) -> str:
        return ", ".join(f"{name}={sum(1 for s in samples if s['target'] == i)}"
                         for i, name in enumerate(names))

    print(f"       Class balance ({args.task}): {balance(all_samples)}")

    # 4) Split into folds
    print(f"[3/4] Splitting into {args.n_folds} patient-level folds ...")
    folds = patient_level_folds(all_samples, args.n_folds, seed=args.seed)

    for i, fold in enumerate(folds):
        print(f"       Fold {i}: {len(fold):>5} segments  ({balance(fold)})")

    # Strip the 'patient' key before saving (not needed by ESC_Dataset)
    for fold in folds:
        for s in fold:
            del s["patient"]

    # 5) Save
    out_path = output_dir / (args.output_name or default_output_name(args.task, args.clip_seconds))
    print(f"[4/4] Saving to {out_path} ...")
    np.save(str(out_path), np.array(folds, dtype=object), allow_pickle=True)

    # Quick sanity check
    loaded = np.load(str(out_path), allow_pickle=True)
    print(f"       Saved ✓  shape={loaded.shape}, "
          f"waveform len={len(loaded[0][0]['waveform'])}, "
          f"dtype={loaded[0][0]['waveform'].dtype}")
    print("Done!")


def default_output_name(task: str, clip_seconds: int) -> str:
    suffix = "" if task == "3class" else f"-{task}"
    return f"train-circor-data-{clip_seconds}s{suffix}.npy"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CIRCOr audio data for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir", default="data/circor/training_data",
                        help="Directory containing raw WAV + metadata files")
    parser.add_argument("--csv_file", default="data/circor/training_data.csv",
                        help="Path to training_data.csv from the CIRCOr dataset")
    parser.add_argument("--output_dir", default="data/circor",
                        help="Directory to write the output .npy file")
    parser.add_argument("--task", choices=sorted(LABEL_MAPS), default="3class",
                        help="3class: Present/Unknown/Absent; binary: Present+Unknown vs Absent")
    parser.add_argument("--output_name", default=None,
                        help="Name of the output .npy file (default: train-circor-data-<clip>s[-binary].npy)")
    parser.add_argument("--sample_rate", type=int, default=4000,
                        help="Target sample rate in Hz (4000 = native)")
    parser.add_argument("--clip_seconds", type=int, default=30,
                        help="Clip length in seconds")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of patient-level CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fold shuffling")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
