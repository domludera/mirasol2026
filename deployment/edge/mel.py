"""Torch-free log-mel front end for the MIRASOL heart-murmur models.

VENDORED COPY of ``src/mirasol/signal/mel.py``.  It is duplicated here on
purpose: ``deployment/edge/`` is the one directory a user copies onto the
Orange Pi Zero 2W, and the device must never need PyTorch, timm, librosa or
the ``mirasol`` package installed.  ``tests/`` asserts that this module and
the packaged one produce identical arrays, so the two cannot drift.

Paper artifacts served
----------------------
* Table 4 (on-device latency / memory) -- ``bench_input.npy`` is the output of
  this front end applied to a synthetic waveform, and is the only input to the
  benchmark.
* Sec. 3.6 / "run it on your Pi" -- ``infer.py`` uses this module so that the
  on-device input representation is the one the models were trained on.

What the paper's input representation is, exactly
-------------------------------------------------
Per recording (``src/mirasol/data/prepare.py``, unchanged here):

1. decode to mono float32 at 4000 Hz (CirCor's native rate),
2. 4th-order Butterworth bandpass, 25-400 Hz, applied with ``sosfilt``
   (causal, *not* zero-phase -- ``sosfiltfilt`` would change the numbers),
3. z-score normalisation over the whole recording,
4. split into non-overlapping 30 s clips; a short tail (and a recording
   shorter than 30 s) is REPEAT-TILED to 30 s, never zero-padded.

Then per clip, at train and eval time (``mirasol/data/dataset.py``):

5. the 30 s clip is concatenated with itself to 60 s,
6. STFT (n_fft = win_length = 128, hop = 120, periodic Hann, centred,
   reflect-padded) -> power spectrogram,
7. Slaney mel filter bank (64 bins, 25-2000 Hz, area-normalised),
8. ``10 * log10(max(mel, 1e-10))``, i.e. torchlibrosa's ``LogmelFilterBank``
   with ``ref=1.0``, ``amin=1e-10`` and ``top_db=None`` -- no dB clipping and
   no per-clip reference, which is why absolute levels matter and why the
   z-score in step 3 is load-bearing.

The result is a single-channel (1, 2001, 64) float32 "image".

Numerical fidelity (measured, not assumed)
------------------------------------------
This module reimplements the pieces of ``librosa`` that the export and
data-preparation scripts used, in numpy + scipy only:

* ``stft`` is BIT-IDENTICAL to ``librosa.stft`` (``np.array_equal`` holds for
  signal lengths 64 ... 240001), which requires reproducing librosa's segmented
  centre-padding, whose final analysis frame reflects a 64-sample tail
  sub-array rather than the whole signal.  That quirk is replicated
  deliberately rather than "fixed", because every published input went through
  it.
* ``mel_filter_bank()`` is bit-identical to
  ``librosa.filters.mel(htk=False, norm='slaney')``.
* the full chain reproduces the committed ``bench_input.npy`` exactly
  (``np.array_equal``), and reproduces the waveforms stored in the prepared
  training ``.npy`` exactly from the source WAVs.
* against the torchlibrosa front end actually used inside the PyTorch models
  (conv-based STFT, float32 throughout) the agreement is 1.75e-06 dB mean /
  6.8e-03 dB max absolute error over a 60 s clip whose values span about 48 dB
  -- i.e. the ONNX graphs see the same input the trained models saw.

Dependencies: numpy, scipy.  Nothing else.
"""

# Portions ported from librosa (ISC), (c) 2013--2023 librosa development team
# See THIRD_PARTY.md section 9. SPDX-License-Identifier: MIT AND ISC

from __future__ import annotations

from functools import lru_cache
from typing import Final

import numpy as np
from scipy import fft as _fft
from scipy import signal as _signal

# ── Signal-processing constants (must match mirasol.config.defaults) ──────────
SAMPLE_RATE: Final[int] = 4000
CLIP_SECONDS: Final[int] = 30
CLIP_SAMPLES: Final[int] = SAMPLE_RATE * CLIP_SECONDS  # 120_000
WINDOW_SIZE: Final[int] = 128  # n_fft, and also win_length
HOP_SIZE: Final[int] = 120
MEL_BINS: Final[int] = 64
FMIN: Final[float] = 25.0
FMAX: Final[float] = 2000.0

# torchlibrosa LogmelFilterBank(ref=1.0, amin=1e-10, top_db=None)
AMIN: Final[float] = 1e-10

# ── Preprocessing constants (must match mirasol.data.prepare) ────────────────
BANDPASS_LOW: Final[float] = 25.0
BANDPASS_HIGH: Final[float] = 400.0
BANDPASS_ORDER: Final[int] = 4

# The 30 s clip is concatenated with itself before the STFT.
CLIP_REPEATS: Final[int] = 2

# Number of STFT frames the models were exported for: 60 s at hop 120,
# centre-padded -> 1 + 240000 // 120 = 2001.
TIME_STEPS: Final[int] = 1 + (CLIP_SAMPLES * CLIP_REPEATS) // HOP_SIZE

_MEL_BREAK_HZ: Final[float] = 1000.0
_MEL_LINEAR_SLOPE: Final[float] = 200.0 / 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Slaney mel scale
# ═════════════════════════════════════════════════════════════════════════════

def hz_to_mel(frequencies: np.ndarray | float) -> np.ndarray:
    """Convert Hz to Slaney mels (linear below 1 kHz, logarithmic above).

    Matches ``librosa.hz_to_mel(..., htk=False)`` for both scalars and arrays.
    """
    hz = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    mels = hz / _MEL_LINEAR_SLOPE
    min_log_mel = _MEL_BREAK_HZ / _MEL_LINEAR_SLOPE
    logstep = np.log(6.4) / 27.0
    log_region = hz >= _MEL_BREAK_HZ
    mels[log_region] = min_log_mel + np.log(hz[log_region] / _MEL_BREAK_HZ) / logstep
    return mels


def mel_to_hz(mels: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`hz_to_mel`.  Matches ``librosa.mel_to_hz(htk=False)``."""
    mel = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    hz = _MEL_LINEAR_SLOPE * mel
    min_log_mel = _MEL_BREAK_HZ / _MEL_LINEAR_SLOPE
    logstep = np.log(6.4) / 27.0
    log_region = mel >= min_log_mel
    hz[log_region] = _MEL_BREAK_HZ * np.exp(logstep * (mel[log_region] - min_log_mel))
    return hz


@lru_cache(maxsize=4)
def mel_filter_bank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = WINDOW_SIZE,
    n_mels: int = MEL_BINS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Slaney (area-normalised) triangular mel filter bank.

    Returns a read-only ``(n_mels, 1 + n_fft // 2)`` float32 matrix, identical
    to ``librosa.filters.mel(htk=False, norm='slaney', dtype=np.float32)``.
    """
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_f = mel_to_hz(
        np.linspace(hz_to_mel(fmin)[0], hz_to_mel(fmax)[0], n_mels + 2)
    )

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalisation: approximately constant energy per channel.
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]

    if not np.all(weights.max(axis=1) > 0):
        raise ValueError(
            f"Empty mel filters: n_mels={n_mels} is too many for n_fft={n_fft} "
            f"at sample_rate={sample_rate} over {fmin:.1f}-{fmax:.1f} Hz."
        )

    weights.setflags(write=False)
    return weights


@lru_cache(maxsize=4)
def _analysis_window(n_fft: int) -> np.ndarray:
    """Periodic Hann analysis window, float64, as librosa/scipy produce it."""
    window = np.asarray(
        _signal.get_window("hann", n_fft, fftbins=True), dtype=np.float64
    )
    window.setflags(write=False)
    return window


# ═════════════════════════════════════════════════════════════════════════════
# STFT
# ═════════════════════════════════════════════════════════════════════════════

def _frame(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice ``x`` into overlapping columns of ``frame_length`` samples."""
    n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
    if n_frames < 1:
        raise ValueError(
            f"signal of {x.shape[-1]} samples is shorter than one "
            f"{frame_length}-sample analysis frame"
        )
    offsets = np.arange(frame_length)[:, np.newaxis]
    starts = hop_length * np.arange(n_frames)[np.newaxis, :]
    return x[offsets + starts]


def _rfft_frames(frames: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Window ``frames`` (float64 arithmetic) and take the real FFT."""
    return _fft.rfft(window[:, np.newaxis] * frames, axis=0).astype(np.complex64)


def stft(
    waveform: np.ndarray,
    n_fft: int = WINDOW_SIZE,
    hop_length: int = HOP_SIZE,
) -> np.ndarray:
    """Centred, reflect-padded STFT, bit-identical to ``librosa.stft``.

    ``librosa`` avoids copying the whole signal by padding only the head and
    the tail region and framing the middle in place.  That is reproduced here
    verbatim -- including the fact that the final frame reflects a short tail
    sub-array -- because ``bench_input.npy`` and every published ONNX input
    were produced by that exact code path.

    Args:
        waveform: 1-D real signal.
        n_fft: FFT size, also the window length.
        hop_length: frame advance in samples.

    Returns:
        complex64 array of shape ``(1 + n_fft // 2, n_frames)``.
    """
    y = np.ascontiguousarray(waveform, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {y.shape}")
    half = n_fft // 2
    window = _analysis_window(n_fft)

    # Which frames depend on head padding, and which on tail padding.
    start_k = int(np.ceil(half / hop_length))
    tail_k = (y.shape[-1] + half - n_fft) // hop_length + 1

    if tail_k <= start_k:
        # Head and tail overlap: pad the whole signal, as librosa does.
        padded = np.pad(y, (half, half), mode="reflect")
        return _rfft_frames(_frame(padded, n_fft, hop_length), window)

    head = np.pad(y[: (start_k - 1) * hop_length - half + n_fft + 1], (half, 0),
                  mode="reflect")
    head_frames = _frame(head, n_fft, hop_length)[:, :start_k]

    if tail_k * hop_length - half + n_fft <= y.shape[-1] + half:
        tail = np.pad(y[tail_k * hop_length - half :], (0, half), mode="reflect")
        tail_frames = _frame(tail, n_fft, hop_length)
    else:
        tail_frames = np.empty((n_fft, 0), dtype=y.dtype)

    middle_frames = _frame(y[start_k * hop_length - half :], n_fft, hop_length)

    return np.concatenate(
        (
            _rfft_frames(head_frames, window),
            _rfft_frames(middle_frames, window),
            _rfft_frames(tail_frames, window),
        ),
        axis=1,
    )


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = WINDOW_SIZE,
    hop_length: int = HOP_SIZE,
    n_mels: int = MEL_BINS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Log-mel spectrogram of an already-tiled waveform.

    Args:
        waveform: 1-D float waveform (normally 60 s = 240_000 samples).
        sample_rate, n_fft, hop_length, n_mels, fmin, fmax: front-end settings;
            the defaults are the paper's.

    Returns:
        float32 array of shape ``(1, n_frames, n_mels)`` -- channel-first with
        one channel, the layout the exported ONNX graphs expect (after adding
        a batch dimension).
    """
    power = np.abs(stft(waveform, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel = mel_filter_bank(sample_rate, n_fft, n_mels, fmin, fmax) @ power
    log_mel = 10.0 * np.log10(np.maximum(mel, AMIN))
    return np.ascontiguousarray(log_mel.T[np.newaxis, :, :], dtype=np.float32)


# The historical name from ``export_onnx.py``, which the shipped provenance logs
# and the rest of the package refer to. Exposed here as well so that this
# vendored module and ``mirasol.signal.mel`` present exactly the same API and can
# be swapped for one another without touching a caller.
waveform_to_mel_spectrogram = log_mel_spectrogram


# ═════════════════════════════════════════════════════════════════════════════
# Waveform conditioning (must stay identical to mirasol.data.prepare)
# ═════════════════════════════════════════════════════════════════════════════

def bandpass_filter(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Causal Butterworth bandpass, exactly as used to build the .npy datasets.

    Uses ``sosfilt`` (single-pass, causal).  ``sosfiltfilt`` would be
    zero-phase and would *not* reproduce the training inputs.
    """
    if not 0.0 < low < high < sample_rate / 2.0:
        raise ValueError(
            f"bandpass {low}-{high} Hz is not inside (0, {sample_rate / 2}) Hz"
        )
    sos = _signal.butter(order, [low, high], btype="bandpass", fs=sample_rate,
                         output="sos")
    return _signal.sosfilt(sos, waveform)


def zscore_normalise(waveform: np.ndarray) -> np.ndarray:
    """Z-score a recording; return silence unchanged rather than dividing by ~0."""
    std = float(np.std(waveform))
    if std < 1e-8:
        return np.zeros_like(waveform)
    return (waveform - np.mean(waveform)) / std


def tile_to_length(waveform: np.ndarray, length: int) -> np.ndarray:
    """Repeat-tile ``waveform`` to exactly ``length`` samples.

    This is the padding the training data used.  Zero-padding a short
    recording instead would present the model with a silent tail that never
    occurs in training and biases every logit; the released ``infer.py``
    therefore tiles.
    """
    if waveform.shape[-1] == 0:
        raise ValueError("cannot tile an empty waveform")
    if waveform.shape[-1] >= length:
        return np.ascontiguousarray(waveform[:length])
    repeats = length // waveform.shape[-1] + 1
    return np.ascontiguousarray(np.tile(waveform, repeats)[:length])


def segment_waveform(
    waveform: np.ndarray, clip_samples: int = CLIP_SAMPLES
) -> tuple[np.ndarray, ...]:
    """Split a recording into non-overlapping fixed-length clips.

    Mirrors ``window_to_clips`` in the data-preparation script: a recording
    shorter than one clip, and any short tail of a longer recording, is
    repeat-tiled up to ``clip_samples``.

    Returns:
        A tuple of 1-D float32 arrays, each exactly ``clip_samples`` long.
    """
    if clip_samples <= 0:
        raise ValueError(f"clip_samples must be positive, got {clip_samples}")
    audio = np.asarray(waveform)
    if audio.shape[-1] <= clip_samples:
        return (tile_to_length(audio, clip_samples).astype(np.float32),)
    clips = [
        tile_to_length(audio[start : start + clip_samples], clip_samples).astype(
            np.float32
        )
        for start in range(0, audio.shape[-1], clip_samples)
    ]
    return tuple(clips)


def clip_to_model_input(clip: np.ndarray, repeats: int = CLIP_REPEATS) -> np.ndarray:
    """Turn one fixed-length clip into the ONNX graphs' input tensor.

    The clip is concatenated with itself (``repeats`` copies, 2 in the paper),
    exactly as the training/eval dataset does, then converted to log-mel.

    Returns:
        float32 array of shape ``(1, 1, n_frames, n_mels)``.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    tiled = np.concatenate([np.asarray(clip, dtype=np.float32)] * repeats)
    return log_mel_spectrogram(tiled)[np.newaxis, :, :, :]


def preprocess_recording(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    clip_samples: int = CLIP_SAMPLES,
    bandpass: bool = True,
) -> tuple[np.ndarray, ...]:
    """Full recording -> one model input per non-overlapping 30 s window.

    Args:
        waveform: 1-D float waveform already at ``sample_rate``.
        sample_rate: sampling rate of ``waveform``; must be the training rate.
        clip_samples: analysis window length in samples.
        bandpass: apply the 25-400 Hz bandpass.  Only set this False to
            reproduce the no-filter ablation (Table S10).

    Returns:
        A tuple of ``(1, 1, n_frames, n_mels)`` float32 arrays, in time order.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"waveform is at {sample_rate} Hz but the models were trained at "
            f"{SAMPLE_RATE} Hz; resample before calling preprocess_recording"
        )
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"expected a mono 1-D waveform, got shape {audio.shape}")
    if bandpass:
        audio = bandpass_filter(audio, sample_rate)
    audio = zscore_normalise(audio)
    return tuple(clip_to_model_input(clip) for clip in
                 segment_waveform(audio, clip_samples))
