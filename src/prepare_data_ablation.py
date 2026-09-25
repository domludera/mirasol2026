#!/usr/bin/env python3
"""Extended data preparation for ablation studies.

Adds flags for:
  --single_location   Keep only one recording per patient
  --no_bandpass       Disable bandpass filtering
  --bandpass_low      Override low cutoff (default 25 Hz)
  --bandpass_high     Override high cutoff (default 400 Hz)
  --clip_seconds      Window length (10, 15, 30)

Usage examples:
    # Baseline (same as current pipeline)
    python -m src.prepare_data_ablation --clip_seconds 30

    # Single-location ablation
    python -m src.prepare_data_ablation --clip_seconds 30 --single_location

    # Window length ablation
    python -m src.prepare_data_ablation --clip_seconds 10
    python -m src.prepare_data_ablation --clip_seconds 15

    # Bandpass ablation: no filter
    python -m src.prepare_data_ablation --clip_seconds 30 --no_bandpass

    # Bandpass ablation: wide filter
    python -m src.prepare_data_ablation --clip_seconds 30 --bandpass_high 1000
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from scipy import signal


# ── helpers ──────────────────────────────────────────────────────────────────

def bandpass_filter(audio, sr, low=25.0, high=400.0, order=4):
    """Apply a Butterworth bandpass filter."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, audio)


def zscore_normalise(audio):
    """Z-score normalise (mean=0, std=1)."""
    std = np.std(audio)
    if std < 1e-8:
        return np.zeros_like(audio)
    return (audio - np.mean(audio)) / std


def window_to_clips(audio, clip_len):
    """Split audio into fixed-length clips."""
    clips = []
    if len(audio) <= clip_len:
        tiled = np.tile(audio, (clip_len // len(audio)) + 1)
        clips.append(tiled[:clip_len])
    else:
        for start in range(0, len(audio), clip_len):
            chunk = audio[start : start + clip_len]
            if len(chunk) < clip_len:
                tiled = np.tile(chunk, (clip_len // len(chunk)) + 1)
                chunk = tiled[:clip_len]
            clips.append(chunk)
    return clips


# ── metadata ─────────────────────────────────────────────────────────────────

def parse_metadata(csv_path):
    """Parse CIRCOr training_data.csv.

    Returns:
        records: dict keyed by WAV filename -> {patient, location, murmur(int)}
        patient_info: dict keyed by patient_id -> {murmur_status, murmur_locations,
                                                   most_audible, rec_locations}
    """
    records = {}
    patient_info = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            patient_id = row["Patient ID"]
            rec_locs = [l.strip() for l in row["Recording locations:"].split("+") if l.strip()]
            murmur_status = row["Murmur"]

            if murmur_status == "Present":
                target = 0
            elif murmur_status == "Unknown":
                target = 1
            else:
                target = 2

            # Store patient-level info
            murmur_locs_raw = row.get("Murmur locations", "nan")
            most_audible = row.get("Most audible location", "nan")
            murmur_locs = []
            if murmur_locs_raw and murmur_locs_raw != "nan":
                murmur_locs = [l.strip() for l in murmur_locs_raw.split("+") if l.strip()]

            patient_info[patient_id] = {
                "murmur_status": murmur_status,
                "murmur_locations": murmur_locs,
                "most_audible": most_audible if most_audible != "nan" else None,
                "rec_locations": rec_locs,
                "target": target,
            }

            for loc in rec_locs:
                wav_name = f"{patient_id}_{loc}.wav"
                records[wav_name] = {
                    "patient": patient_id,
                    "location": loc,
                    "murmur": target,
                }

    return records, patient_info


def filter_single_location(records, patient_info):
    """Keep only one recording per patient.

    Strategy:
      - Present patients → Most Audible Location (fallback: first murmur location)
      - Absent/Unknown   → first listed recording location
    """
    selected = {}

    for pid, pinfo in patient_info.items():
        if pinfo["murmur_status"] == "Present":
            # Use most audible location, or first murmur location, or first rec location
            chosen_loc = (
                pinfo["most_audible"]
                or (pinfo["murmur_locations"][0] if pinfo["murmur_locations"] else None)
                or pinfo["rec_locations"][0]
            )
        else:
            chosen_loc = pinfo["rec_locations"][0]

        wav_name = f"{pid}_{chosen_loc}.wav"
        if wav_name in records:
            selected[wav_name] = records[wav_name]
        else:
            # Fallback: use first available recording for this patient
            for loc in pinfo["rec_locations"]:
                fallback_name = f"{pid}_{loc}.wav"
                if fallback_name in records:
                    selected[fallback_name] = records[fallback_name]
                    break

    return selected


# ── fold splitting ───────────────────────────────────────────────────────────

def patient_level_folds(samples, n_folds, seed=42):
    """Split samples into n_folds at the patient level."""
    patient_map = defaultdict(list)
    for s in samples:
        patient_map[s["patient"]].append(s)

    patients = sorted(patient_map.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    folds = [[] for _ in range(n_folds)]
    for i, pid in enumerate(patients):
        folds[i % n_folds].extend(patient_map[pid])

    return folds


# ── main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(args):
    raw_dir = Path(args.raw_dir)
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)
    sr = args.sample_rate
    clip_len = sr * args.clip_seconds

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Parse metadata
    print(f"[1/4] Parsing metadata from {csv_path} ...")
    records, patient_info = parse_metadata(str(csv_path))
    print(f"       Found {len(records)} recordings, {len(patient_info)} patients.")

    # 2) Apply single-location filter if requested
    if args.single_location:
        records = filter_single_location(records, patient_info)
        print(f"       [ABLATION] Single-location mode: {len(records)} recordings kept.")

    # 3) Determine filter settings
    if args.no_bandpass:
        filter_desc = "NONE"
    else:
        filter_desc = f"{args.bandpass_low}–{args.bandpass_high} Hz"

    print(f"[2/4] Processing WAVs (sr={sr}, clip={args.clip_seconds}s, "
          f"filter={filter_desc}) ...")

    all_samples = []
    skipped = 0
    processed = 0

    wav_files = sorted(raw_dir.glob("*.wav"))
    total = len(wav_files)

    for idx, wav_path in enumerate(wav_files, 1):
        wav_name = wav_path.name
        if wav_name not in records:
            skipped += 1
            continue

        info = records[wav_name]
        audio, _ = librosa.load(str(wav_path), sr=sr)

        # Bandpass filter (unless disabled)
        if not args.no_bandpass:
            audio = bandpass_filter(audio, sr, low=args.bandpass_low,
                                    high=args.bandpass_high)

        # Z-score normalise
        audio = zscore_normalise(audio)

        # Window into clips
        clips = window_to_clips(audio, clip_len)

        for ci, clip in enumerate(clips):
            seg_name = (f"{wav_name.removesuffix('.wav')}-{ci}.wav"
                        if len(clips) > 1 else wav_name)
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
          f"({skipped} skipped).")

    # Class balance
    n_present = sum(1 for s in all_samples if s["target"] == 0)
    n_unknown = sum(1 for s in all_samples if s["target"] == 1)
    n_absent = sum(1 for s in all_samples if s["target"] == 2)
    print(f"       Class balance: {n_present} Present, {n_unknown} Unknown, "
          f"{n_absent} Absent")

    # 4) Split into folds
    print(f"[3/4] Splitting into {args.n_folds} patient-level folds ...")
    folds = patient_level_folds(all_samples, args.n_folds, seed=args.seed)

    for i, fold in enumerate(folds):
        n_p = sum(1 for s in fold if s["target"] == 0)
        n_u = sum(1 for s in fold if s["target"] == 1)
        n_a = sum(1 for s in fold if s["target"] == 2)
        print(f"       Fold {i}: {len(fold):>5} segments  "
              f"(Present={n_p}, Unknown={n_u}, Absent={n_a})")

    # Strip patient key before saving
    for fold in folds:
        for s in fold:
            del s["patient"]

    # 5) Save
    out_path = output_dir / args.output_name
    print(f"[4/4] Saving to {out_path} ...")
    np.save(str(out_path), np.array(folds, dtype=object), allow_pickle=True)

    loaded = np.load(str(out_path), allow_pickle=True)
    total_segs = sum(len(f) for f in loaded)
    print(f"       Saved ✓  folds={loaded.shape[0]}, total_segments={total_segs}, "
          f"waveform_len={len(loaded[0][0]['waveform'])}")
    print("Done!")
    return total_segs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CIRCOr data with ablation options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir", default="data/circor/training_data")
    parser.add_argument("--csv_file", default="data/circor/training_data.csv")
    parser.add_argument("--output_dir", default="data/circor")
    parser.add_argument("--output_name", default="ablation.npy")
    parser.add_argument("--sample_rate", type=int, default=4000)
    parser.add_argument("--clip_seconds", type=int, default=30)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Ablation flags
    parser.add_argument("--single_location", action="store_true",
                        help="Keep only one recording per patient")
    parser.add_argument("--no_bandpass", action="store_true",
                        help="Disable bandpass filtering")
    parser.add_argument("--bandpass_low", type=float, default=25.0,
                        help="Bandpass low cutoff (Hz)")
    parser.add_argument("--bandpass_high", type=float, default=400.0,
                        help="Bandpass high cutoff (Hz)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
