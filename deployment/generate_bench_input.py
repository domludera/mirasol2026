#!/usr/bin/env python3
"""Regenerate ``deployment/edge/bench_input.npy``, the input to Table 4.

Paper artifact: Table 4 (on-device latency, memory, size).  Every row of that
table times the SAME pre-computed log-mel tensor, so that a reproducer on their
own Orange Pi is measuring the same work.  The tensor is committed to the
repository (512,384 B) because it is purely synthetic -- ``RandomState(42)``
noise plus 50 Hz and 120 Hz sinusoids standing in for heart-sound
fundamentals -- and contains no patient data whatsoever.

This script only needs to be run if the front end changes.  The committed file
has sha256

    f8a7c2d832b305de234c16f31c66e476065574915f1204d98966c78f17235d05

and ``tests/test_mel_parity.py`` asserts that the front end still reproduces it,
which pins the paper's input representation.

Torch-free: the log-mel front end comes from :mod:`mirasol.signal.mel`, not from
the export module, so nothing here imports PyTorch.

Usage
-----
    python -m deployment.generate_bench_input
    python deployment/generate_bench_input.py --output /tmp/bench_input.npy
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from mirasol.paths import REPO_ROOT
from mirasol.signal.mel import waveform_to_mel_spectrogram

# The synthetic waveform recipe. These values define the committed file; change
# any of them and the sha256 above no longer holds.
SEED: Final[int] = 42
NOISE_AMPLITUDE: Final[float] = 0.01
TONES_HZ: Final[tuple[tuple[float, float], ...]] = ((50.0, 0.1), (120.0, 0.05))
CLIP_REPEATS: Final[int] = 2

DEFAULT_OUTPUT: Final[Path] = (
    REPO_ROOT / "deployment" / "edge" / "bench_input.npy"
)


def synthetic_waveform(sample_rate: int, clip_seconds: int) -> np.ndarray:
    """Deterministic stand-in for a 30 s phonocardiogram.

    Gaussian noise at ``NOISE_AMPLITUDE`` plus the ``TONES_HZ`` sinusoids.  No
    patient data is involved, which is what makes the resulting tensor safe to
    commit.
    """
    clip_samples = sample_rate * clip_seconds
    rng = np.random.RandomState(SEED)
    waveform = rng.randn(clip_samples).astype(np.float32) * NOISE_AMPLITUDE
    seconds = np.linspace(0, clip_seconds, clip_samples, dtype=np.float32)
    for frequency, amplitude in TONES_HZ:
        waveform += amplitude * np.sin(2 * np.pi * frequency * seconds)
    return waveform


def build_bench_input(sample_rate: int, clip_seconds: int) -> np.ndarray:
    """Log-mel tensor of the synthetic clip, shaped for the ONNX graphs.

    The clip is concatenated with itself before the STFT, exactly as the
    training and evaluation dataset does, giving the ``(1, 1, 2001, 64)`` input
    every released graph declares.
    """
    waveform = synthetic_waveform(sample_rate, clip_seconds)
    tiled = np.concatenate([waveform] * CLIP_REPEATS)
    mel = waveform_to_mel_spectrogram(tiled)
    return np.ascontiguousarray(mel[np.newaxis, :, :, :], dtype=np.float32)


def sha256_of_file(path: Path) -> str:
    """sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deployment.generate_bench_input",
        description="Regenerate the synthetic Table 4 benchmark input.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="destination .npy (default: deployment/edge/bench_input.npy)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=None,
        help="override the configured sample rate (default: from "
             "mirasol.config.defaults)",
    )
    parser.add_argument(
        "--clip-seconds", type=int, default=None,
        help="override the configured clip length in seconds",
    )
    parser.add_argument(
        "--sha256-file", type=Path, default=None,
        help="also write a 'sha256sum -c'-compatible checksum file, so the "
             "provenance hash quoted in docs/edge_setup.md is generated rather "
             "than typed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from mirasol.config import defaults

    sample_rate = (
        defaults.sample_rate if args.sample_rate is None else args.sample_rate
    )
    clip_seconds = (
        defaults.clip_seconds if args.clip_seconds is None else args.clip_seconds
    )
    if sample_rate <= 0 or clip_seconds <= 0:
        raise ValueError(
            f"sample_rate ({sample_rate}) and clip_seconds ({clip_seconds}) "
            "must both be positive"
        )

    mel = build_bench_input(sample_rate, clip_seconds)
    print(f"sample_rate={sample_rate} Hz, clip={clip_seconds} s x {CLIP_REPEATS}")
    print(f"shape={mel.shape} dtype={mel.dtype}")
    print(f"min={mel.min():.3f} max={mel.max():.3f} mean={mel.mean():.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The committed artifact was saved from a transposed (Fortran-ordered)
    # array, so its .npy header records fortran_order: True. Writing the same
    # memory order keeps a regenerated file BYTE-identical to the committed one
    # -- and therefore keeps the published sha256 above verifiable -- instead of
    # merely value-identical. np.load returns the same array either way.
    np.save(args.output, np.asfortranarray(mel))
    digest = sha256_of_file(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} B)")
    print(f"sha256 {digest}")

    if args.sha256_file is not None:
        args.sha256_file.parent.mkdir(parents=True, exist_ok=True)
        args.sha256_file.write_text(
            f"{digest}  {args.output.name}\n", encoding="utf-8"
        )
        print(f"wrote {args.sha256_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
