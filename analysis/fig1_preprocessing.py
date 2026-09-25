"""
Fig. 1: raw PCG -> 25-400 Hz Butterworth bandpass -> z-score, using the same
helpers as src/prepare_data.py.

Usage:
    python analysis/fig1_preprocessing.py [--wav data/circor/training_data/50008_MV.wav]
"""
import argparse
import os
import sys

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prepare_data import bandpass_filter, zscore_normalise

SAMPLE_RATE = 4000
SECONDS_SHOWN = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the preprocessing pipeline (Fig. 1)")
    parser.add_argument("--wav", default="data/circor/training_data/50008_MV.wav")
    parser.add_argument("--out", default="analysis/figures/fig1_preprocessing.png")
    args = parser.parse_args()

    raw, sr = librosa.load(args.wav, sr=SAMPLE_RATE)
    filtered = bandpass_filter(raw, sr)
    normalised = zscore_normalise(filtered)

    n = min(len(raw), SECONDS_SHOWN * sr)
    t = np.arange(n) / sr
    panels = [
        (raw, f"Raw PCG ({sr // 1000} kHz)"),
        (filtered, "4th-order Butterworth bandpass (25-400 Hz)"),
        (normalised, "z-score normalised"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 2.5))
    for ax, (y, title) in zip(axes, panels):
        ax.plot(t, y[:n], linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_xlim(0, t[-1])
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
