#!/usr/bin/env python3
"""Generate a pre-computed mel-spectrogram input for Orange Pi benchmarking.

Saves a .npy file that the benchmark script can load directly,
avoiding the need to install librosa/scipy on the edge device.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import esc_config as config

def main():
    # Use the same waveform_to_mel_spectrogram from export_onnx.py
    from export_onnx import waveform_to_mel_spectrogram

    # Generate a realistic synthetic waveform
    rng = np.random.RandomState(42)
    waveform = rng.randn(config.clip_samples).astype(np.float32) * 0.01

    # Add heart-like low-frequency content
    t = np.linspace(0, config.clip_seconds, config.clip_samples, dtype=np.float32)
    waveform += 0.1 * np.sin(2 * np.pi * 50 * t)
    waveform += 0.05 * np.sin(2 * np.pi * 120 * t)

    # Double waveform (matches ESC_Dataset)
    waveform = np.concatenate([waveform, waveform])

    mel = waveform_to_mel_spectrogram(waveform)
    # Add batch dim: (1, T, mel_bins) -> (1, 1, T, mel_bins)
    mel = mel[np.newaxis, :]

    print(f"Mel shape: {mel.shape}, dtype: {mel.dtype}")
    print(f"Min: {mel.min():.2f}, Max: {mel.max():.2f}, Mean: {mel.mean():.2f}")

    out_path = os.path.join("exports", "bench_input.npy")
    np.save(out_path, mel)
    print(f"Saved to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
