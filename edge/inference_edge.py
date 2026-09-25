#!/usr/bin/env python3
"""
Lightweight inference script for heart-sound classification on edge devices.

Designed for the Orange Pi Zero 2W (Cortex-A53 aarch64 Linux).
Dependencies: onnxruntime, numpy, librosa  (NO PyTorch required)

Usage:
    python edge/inference_edge.py --model exports/tinyvit/model_int8_static.onnx --audio recording.wav
    python edge/inference_edge.py --model exports/tinyvit/model_int8_static.onnx --audio_dir /path/to/wavs/
"""

import argparse
import os
import sys
import time

import librosa
import numpy as np
import onnxruntime as ort
from scipy import signal

# ── Audio / mel-spectrogram settings (must match training config) ────────────
SAMPLE_RATE = 4000
CLIP_SECONDS = 30
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS  # 120,000

N_FFT = 128
HOP_LENGTH = 120
N_MELS = 64
FMIN = 25
FMAX = 2000

# ── Preprocessing settings (must match prepare_data.py) ─────────────────────
BANDPASS_LOW = 25.0
BANDPASS_HIGH = 400.0
BANDPASS_ORDER = 4

# Label names keyed by the number of model outputs (3-class or legacy binary models)
CLASS_NAMES_BY_COUNT = {
    2: ["Present", "Absent"],
    3: ["Present", "Unknown", "Absent"],
}


def class_names_for(num_classes: int) -> list:
    return CLASS_NAMES_BY_COUNT.get(num_classes, [f"Class {i}" for i in range(num_classes)])


def window_to_clips(audio: np.ndarray, clip_len: int = CLIP_SAMPLES) -> list:
    """Fixed-length clips exactly as in src/prepare_data.py: non-overlapping
    segments, with short recordings / tails filled by repeating themselves."""
    clips = []
    for start in range(0, max(len(audio), 1), clip_len):
        chunk = audio[start : start + clip_len]
        if len(chunk) < clip_len:
            chunk = np.tile(chunk, (clip_len // len(chunk)) + 1)[:clip_len]
        clips.append(chunk)
    return clips


def clip_to_mel(clip: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram of one clip, shape (1, 1, time_steps, mel_bins)."""
    # ESC_Dataset doubles the waveform during training/testing
    waveform = np.concatenate([clip, clip])

    # STFT → power spectrogram
    stft = librosa.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    power = np.abs(stft) ** 2

    # Mel filter bank → log mel
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel = mel_basis @ power
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))

    return log_mel.T[np.newaxis, np.newaxis, :, :].astype(np.float32)


def preprocess_waveform(waveform: np.ndarray) -> list:
    """Bandpass + z-score a 4 kHz waveform and return one mel-spectrogram per 30 s clip."""
    # Bandpass filter (matches prepare_data.py)
    sos = signal.butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH],
                        btype="bandpass", fs=SAMPLE_RATE, output="sos")
    waveform = signal.sosfilt(sos, waveform)

    # Z-score normalise (matches prepare_data.py)
    std = np.std(waveform)
    if std < 1e-8:
        waveform = np.zeros_like(waveform)
    else:
        waveform = (waveform - np.mean(waveform)) / std

    return [clip_to_mel(clip) for clip in window_to_clips(waveform)]


def preprocess_audio(audio_path: str) -> list:
    """Load a .wav file and produce the mel-spectrogram(s) for the ONNX model."""
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(waveform) == 0:
        raise ValueError(f"{audio_path} contains no audio")
    return preprocess_waveform(waveform)


def run_inference(session: ort.InferenceSession, mels: list) -> dict:
    """Run the model on every clip and mean-pool the logits (as in the paper's evaluation)."""
    input_meta = session.get_inputs()[0]
    dtype = np.float16 if input_meta.type == "tensor(float16)" else np.float32

    t0 = time.perf_counter()
    clip_logits = [session.run(None, {input_meta.name: mel.astype(dtype)})[0][0] for mel in mels]
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = np.mean(np.asarray(clip_logits, dtype=np.float64), axis=0)  # (num_classes,)

    # Apply softmax for confidence scores
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    class_names = class_names_for(len(probabilities))

    return {
        "class_id": predicted_class,
        "class_name": class_names[predicted_class],
        "confidence": confidence,
        "probabilities": {name: float(p) for name, p in zip(class_names, probabilities)},
        "latency_ms": latency_ms,
        "num_clips": len(mels),
        "logits": logits.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Heart-sound classification on edge (Orange Pi / ARM Linux)"
    )
    parser.add_argument(
        "--model", required=True, help="Path to the ONNX model (.onnx)"
    )
    parser.add_argument(
        "--audio", help="Path to a single .wav file"
    )
    parser.add_argument(
        "--audio_dir", help="Path to a directory of .wav files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed output"
    )
    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        parser.error("Provide --audio or --audio_dir")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4  # Cortex-A53 has 4 cores

    session = ort.InferenceSession(
        args.model,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]
    print(f"Input: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")

    # ── Collect audio files ──────────────────────────────────────────────
    audio_files = []
    if args.audio:
        audio_files.append(args.audio)
    if args.audio_dir:
        for f in sorted(os.listdir(args.audio_dir)):
            if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                audio_files.append(os.path.join(args.audio_dir, f))

    if not audio_files:
        print("No audio files found.")
        sys.exit(1)

    # ── Run inference ────────────────────────────────────────────────────
    print(f"\nProcessing {len(audio_files)} file(s)...\n")
    print(f"{'File':<40} {'Prediction':<12} {'Confidence':>10} {'Latency':>10}")
    print("-" * 75)

    results = []
    for audio_path in audio_files:
        try:
            mels = preprocess_audio(audio_path)
            result = run_inference(session, mels)
            result["file"] = os.path.basename(audio_path)
            results.append(result)

            print(
                f"{result['file']:<40} "
                f"{result['class_name']:<12} "
                f"{result['confidence']:>9.1%} "
                f"{result['latency_ms']:>8.1f} ms"
            )

            if args.verbose:
                for cls, prob in result["probabilities"].items():
                    print(f"    {cls}: {prob:.4f}")
                print()

        except Exception as e:
            print(f"{os.path.basename(audio_path):<40} ERROR: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    if len(results) > 1:
        latencies = [r["latency_ms"] for r in results]
        present_count = sum(1 for r in results if r["class_name"] == "Present")
        print(f"\n{'='*75}")
        print(f"Summary: {len(results)} files processed")
        print(f"  Murmur present: {present_count}/{len(results)}")
        print(f"  Avg latency: {np.mean(latencies):.1f} ms")
        print(f"  Min/Max latency: {np.min(latencies):.1f} / {np.max(latencies):.1f} ms")


if __name__ == "__main__":
    main()
