"""The paper's phonocardiogram preprocessing chain, in NumPy and SciPy only.

This module is the single definition of the input representation used by every
number in the paper. It serves Fig. 1 (the preprocessing illustration), the
mel-spectrogram fed to all six architectures during five-fold cross-validation
(Table 1), and the pre-computed tensor benchmarked on the Orange Pi (Table 4).

Chain, in order (Sec. 2.2 of the paper)
---------------------------------------
1. Resample to 4 kHz (done by the caller when reading the WAV).
2. 4th-order Butterworth bandpass, 25-400 Hz, applied **causally** with
   :func:`scipy.signal.sosfilt`.
3. Per-recording z-score normalisation.
4. Tile-and-crop into non-overlapping 30 s windows; a window shorter than 30 s
   is filled by repeating itself rather than zero-padded.
5. Concatenate the window with itself, giving 60 s / 240 000 samples.
6. Log-mel spectrogram: 64 mel bins, window 128, hop 120, 25-2000 Hz, power
   spectrum, ``10 * log10(max(mel, 1e-10))``, no top-dB clipping and no
   reference normalisation.

The result has shape ``(1, 2001, 64)``; with a batch axis the ONNX graphs take
``(1, 1, 2001, 64)``.

Why steps 2, 4 and 5 look odd
-----------------------------
The causal filter, the repeat-tiling and the self-concatenation are all
faithful reproductions of the pipeline the published models were trained with,
not design recommendations. Step 5 in particular exists because the training
dataset wrapper doubled every waveform; the exported graphs' input length
therefore depends on it, so it must not be "cleaned up".

Numerical equivalence
---------------------
Steps 2-4 reproduce ``src/prepare_data.py`` from the original source tree and
step 6 reproduces ``export_onnx.py:waveform_to_mel_spectrogram``, which was
implemented with librosa. Here the STFT and the mel filterbank are written out
directly so that no librosa install is needed on the edge device. Both are
byte-for-byte reproductions of librosa's ``stft`` and ``filters.mel`` for the
parameters above, including librosa's float32 down-cast of the complex
spectrum; ``tests/test_mel_parity.py`` pins the result against the committed
``deployment/edge/bench_input.npy``.
"""

# Portions ported from librosa (ISC), (c) 2013--2023 librosa development team
# See THIRD_PARTY.md section 9. SPDX-License-Identifier: MIT AND ISC

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import rfft
from scipy.signal import butter, get_window, sosfilt

__all__ = [
    "BandpassConfig",
    "DEFAULT_BANDPASS",
    "DEFAULT_MEL",
    "MelConfig",
    "bandpass",
    "bandpass_filter",
    "concatenate_with_self",
    "log_mel_spectrogram",
    "mel_config_from_mapping",
    "mel_filterbank",
    "num_frames",
    "preprocess_recording",
    "segment_waveform",
    "tile_to_length",
    "waveform_to_mel_spectrogram",
    "waveform_to_model_input",
    "zscore",
    "zscore_normalise",
]

#: Floor applied inside the logarithm, matching torchlibrosa's ``amin``.
LOG_AMIN: float = 1e-10

#: Below this standard deviation a recording is treated as silence and z-scoring
#: returns zeros instead of amplifying numerical noise.
SILENCE_STD: float = 1e-8

#: Bytes per STFT block. ``librosa.util.MAX_MEM_BLOCK``; see :func:`_stft` for
#: why the block size has to match rather than merely be large enough.
MAX_MEM_BLOCK: int = 2**8 * 2**10


@dataclass(frozen=True)
class MelConfig:
    """Log-mel spectrogram parameters. Immutable and hashable.

    The defaults are the values used for every published number; they are
    mirrored in :mod:`mirasol.config.defaults` for the training entry points and
    the two are asserted equal in CI.
    """

    sample_rate: int = 4000
    clip_seconds: int = 30
    window_size: int = 128
    hop_size: int = 120
    mel_bins: int = 64
    fmin: float = 25.0
    fmax: float = 2000.0

    def __post_init__(self) -> None:
        if self.window_size % 2 != 0:
            raise ValueError(f"window_size must be even, got {self.window_size}")
        if self.hop_size <= 0:
            raise ValueError(f"hop_size must be positive, got {self.hop_size}")
        if not 0.0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError(
                f"require 0 <= fmin < fmax <= sample_rate/2, got fmin={self.fmin}, "
                f"fmax={self.fmax}, sample_rate={self.sample_rate}"
            )

    @property
    def clip_samples(self) -> int:
        """Samples in one 30 s analysis window (120 000 at 4 kHz)."""
        return self.sample_rate * self.clip_seconds

    @property
    def model_samples(self) -> int:
        """Samples actually seen by the network, after self-concatenation."""
        return 2 * self.clip_samples

    @property
    def time_steps(self) -> int:
        """Spectrogram frames for one model input (2001 for the paper's setup)."""
        return num_frames(self.model_samples, self.window_size, self.hop_size)

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        """ONNX input shape, ``(batch, channel, time, mel)`` = ``(1, 1, 2001, 64)``."""
        return (1, 1, self.time_steps, self.mel_bins)


@dataclass(frozen=True)
class BandpassConfig:
    """Butterworth bandpass parameters, separated so ablations can vary them.

    Table S10 sweeps this: ``low=25, high=400`` is the deployed setting,
    ``high=1000`` the wide variant, and the no-filter variant is expressed by
    passing ``apply_bandpass=False`` rather than by a degenerate band.
    """

    low: float = 25.0
    high: float = 400.0
    order: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.low < self.high:
            raise ValueError(
                f"require 0 < low < high, got low={self.low}, high={self.high}"
            )
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")


#: The published configuration. Treat as constant.
DEFAULT_MEL: MelConfig = MelConfig()
DEFAULT_BANDPASS: BandpassConfig = BandpassConfig()


def mel_config_from_mapping(values: Mapping[str, object] | object) -> MelConfig:
    """Build a :class:`MelConfig` from a config object or mapping.

    Accepts either a mapping or anything with the attributes, so training code
    can derive the spectrogram parameters from its resolved configuration
    instead of restating them - the drift this module exists to prevent.
    """
    if isinstance(values, Mapping):
        get = values.__getitem__
        has = values.__contains__
    else:
        def get(key: str) -> object:
            return getattr(values, key)

        def has(key: str) -> bool:
            return hasattr(values, key)

    missing = [
        name
        for name in (
            "sample_rate",
            "clip_seconds",
            "window_size",
            "hop_size",
            "mel_bins",
            "fmin",
            "fmax",
        )
        if not has(name)
    ]
    if missing:
        raise KeyError(f"configuration is missing mel parameter(s): {missing}")
    return MelConfig(
        sample_rate=int(get("sample_rate")),  # type: ignore[arg-type]
        clip_seconds=int(get("clip_seconds")),  # type: ignore[arg-type]
        window_size=int(get("window_size")),  # type: ignore[arg-type]
        hop_size=int(get("hop_size")),  # type: ignore[arg-type]
        mel_bins=int(get("mel_bins")),  # type: ignore[arg-type]
        fmin=float(get("fmin")),  # type: ignore[arg-type]
        fmax=float(get("fmax")),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Step 2-3: filtering and normalisation
# ---------------------------------------------------------------------------


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    config: BandpassConfig = DEFAULT_BANDPASS,
) -> np.ndarray:
    """Apply the paper's causal 4th-order Butterworth bandpass.

    ``sosfilt`` (causal, single pass) is used deliberately: the models were
    trained on causally filtered audio, and swapping in the zero-phase
    ``sosfiltfilt`` changes the group delay and therefore the spectrogram. Fig.
    1 of the paper illustrates this same filter.
    """
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {audio.shape}")
    nyquist = sample_rate / 2.0
    if config.high >= nyquist:
        raise ValueError(
            f"bandpass high cutoff {config.high} Hz must be below the Nyquist "
            f"frequency {nyquist} Hz for sample_rate={sample_rate}"
        )
    sos = butter(
        config.order,
        [config.low, config.high],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return sosfilt(sos, audio)


def zscore_normalise(audio: np.ndarray) -> np.ndarray:
    """Z-score the whole recording; return zeros for a silent one.

    Normalisation is per recording, before windowing, so every window of a
    recording shares one scale.
    """
    audio = np.asarray(audio)
    std = float(np.std(audio))
    if std < SILENCE_STD:
        return np.zeros_like(audio)
    return (audio - np.mean(audio)) / std


# ---------------------------------------------------------------------------
# Step 4-5: windowing
# ---------------------------------------------------------------------------


def tile_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Repeat *audio* until it is exactly *length* samples long.

    Used instead of zero-padding so that a short recording still presents a
    continuous cardiac rhythm to the network. ``deployment/edge/infer.py``
    depends on this: zero-padding there produced inputs unlike anything seen in
    training.
    """
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("cannot tile an empty waveform")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    repeats = (length // audio.size) + 1
    return np.tile(audio, repeats)[:length]


def segment_waveform(audio: np.ndarray, clip_samples: int) -> tuple[np.ndarray, ...]:
    """Split a recording into non-overlapping windows of *clip_samples*.

    A recording no longer than one window becomes a single tiled window; a
    longer one is cut into consecutive windows and a short tail is tiled up to
    full length. This is the behaviour that produced the published segment
    counts, so the tail is kept rather than dropped.
    """
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("cannot segment an empty waveform")
    if audio.size <= clip_samples:
        return (tile_to_length(audio, clip_samples),)
    windows = [
        tile_to_length(chunk, clip_samples)
        for start in range(0, audio.size, clip_samples)
        if (chunk := audio[start : start + clip_samples]).size > 0
    ]
    return tuple(windows)


def concatenate_with_self(waveform: np.ndarray) -> np.ndarray:
    """Return ``[waveform, waveform]``, doubling its length.

    Reproduces the doubling performed by the training dataset wrapper. Every
    exported ONNX graph has a fixed 2001-frame time axis that follows from it,
    so it is part of the model contract, not a preprocessing choice.
    """
    waveform = np.asarray(waveform)
    if waveform.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {waveform.shape}")
    return np.concatenate((waveform, waveform))


# ---------------------------------------------------------------------------
# Step 6: log-mel spectrogram
# ---------------------------------------------------------------------------


def num_frames(num_samples: int, window_size: int, hop_size: int) -> int:
    """Frames produced by a centred STFT, matching ``librosa.stft``."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    padded = num_samples + 2 * (window_size // 2)
    if padded < window_size:
        raise ValueError(
            f"{num_samples} samples is too short for window_size={window_size}"
        )
    return 1 + (padded - window_size) // hop_size


@lru_cache(maxsize=8)
def _hann_window(size: int) -> np.ndarray:
    """Periodic (``fftbins=True``) Hann window, as used by ``librosa.stft``."""
    window = np.asarray(get_window("hann", size, fftbins=True), dtype=np.float64)
    window.flags.writeable = False
    return window


def _hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    """Slaney (Auditory Toolbox) Hz -> mel, as in ``librosa.hz_to_mel``."""
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64)).copy()
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    mels[log_region] = min_log_mel + np.log(frequencies[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Slaney mel -> Hz, as in ``librosa.mel_to_hz``."""
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freqs[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freqs


@lru_cache(maxsize=8)
def mel_filterbank(config: MelConfig = DEFAULT_MEL) -> np.ndarray:
    """Return the ``(mel_bins, 1 + window_size // 2)`` Slaney-normalised bank.

    Identical to ``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)`` with
    librosa's defaults ``htk=False``, ``norm='slaney'``, ``dtype=float32``. The
    returned array is read-only; the cache hands out one shared instance.
    """
    n_bins = 1 + config.window_size // 2
    weights = np.zeros((config.mel_bins, n_bins), dtype=np.float32)

    fft_freqs = np.fft.rfftfreq(n=config.window_size, d=1.0 / config.sample_rate)
    mel_limits = _hz_to_mel_slaney(np.asarray([config.fmin, config.fmax]))
    mel_edges = _mel_to_hz_slaney(
        np.linspace(mel_limits[0], mel_limits[1], config.mel_bins + 2)
    )

    edge_diff = np.diff(mel_edges)
    ramps = np.subtract.outer(mel_edges, fft_freqs)
    for i in range(config.mel_bins):
        lower = -ramps[i] / edge_diff[i]
        upper = ramps[i + 2] / edge_diff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalisation: each filter integrates to a constant energy.
    enorm = 2.0 / (mel_edges[2 : config.mel_bins + 2] - mel_edges[: config.mel_bins])
    weights *= enorm[:, np.newaxis]

    weights.flags.writeable = False
    return weights


def _frame(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Strided ``(frame_length, n_frames)`` view, as ``librosa.util.frame``."""
    if waveform.size < frame_length:
        raise ValueError(
            f"{waveform.size} samples is too short to frame at "
            f"frame_length={frame_length}"
        )
    n_frames = 1 + (waveform.size - frame_length) // hop_length
    stride = waveform.strides[-1]
    return np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_length, n_frames),
        strides=(stride, stride * hop_length),
        writeable=False,
    )


def _centred_frames(
    waveform: np.ndarray, window_size: int, hop_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce ``librosa.stft``'s centred framing exactly, quirks included.

    Returns the head, middle and tail frame blocks separately, because librosa
    transforms them separately and the split is observable (see below).

    A naive implementation reflect-pads the whole signal once. librosa instead
    pads the head and the tail separately to avoid copying the signal, and
    reflect-padding a 64-sample tail slice is *not* the same as reflect-padding
    the full signal: once the pad width exceeds the slice length, NumPy reflects
    a second time. For the paper's 240 000-sample input exactly one sample of the
    final frame differs between the two schemes.

    That single sample propagates into every published ONNX input tensor, so the
    split scheme is reproduced here rather than corrected. The head frames are
    unaffected. Behaviour is identical to librosa 0.10-0.11.
    """
    pad = window_size // 2
    length = waveform.size
    empty = np.empty((window_size, 0), dtype=waveform.dtype)

    first_middle_frame = -(-pad // hop_size)  # ceil(pad / hop_size)
    first_tail_frame = (length + pad - window_size) // hop_size + 1

    if first_tail_frame <= first_middle_frame:
        # Head and tail padding overlap: one full reflect-pad is unavoidable.
        padded = np.pad(waveform, (pad, pad), mode="reflect")
        return empty, _frame(padded, window_size, hop_size), empty

    head_source = waveform[: (first_middle_frame - 1) * hop_size - pad + window_size + 1]
    head = _frame(
        np.pad(head_source, (pad, 0), mode="reflect"), window_size, hop_size
    )[:, :first_middle_frame]

    middle_start = first_middle_frame * hop_size - pad
    middle = _frame(waveform[middle_start:], window_size, hop_size)

    if first_tail_frame * hop_size - pad + window_size <= length + pad:
        tail_source = waveform[first_tail_frame * hop_size - pad :]
        tail = _frame(np.pad(tail_source, (0, pad), mode="reflect"), window_size, hop_size)
    else:
        tail = empty

    return head, middle, tail


def _stft(waveform: np.ndarray, config: MelConfig) -> np.ndarray:
    """Centred STFT, assembled exactly as ``librosa.stft`` assembles it.

    Four details are reproduced deliberately, because each one is observable in
    the last bits of the result and every published ONNX input carries them:

    * the transform is :func:`scipy.fft.rfft`, which is librosa's default FFT
      backend when SciPy is installed. NumPy's ``rfft`` differs from it by a few
      units in the last place;
    * the complex output is allocated at the precision matching the real input
      (``complex64`` for ``float32`` audio) and the ``float64`` FFT result is
      assigned into it, so the down-cast rounding happens per block;
    * the middle frames are transformed in blocks of at most ``MAX_MEM_BLOCK``
      bytes, because an FFT of a 512-column slab is not always bit-identical to
      the same columns inside a 2001-column one;
    * the buffer is Fortran-ordered, which selects a different BLAS path in the
      mel projection that follows and can differ in the last bit.

    Dropping any of the four perturbs a handful of cells in 128 000 by up to
    about 1e-5 dB on measured data - immaterial to any prediction, but enough to
    break bit-exactness, and the point of this module is that there is exactly
    one definition of the paper's input.
    """
    window_size = config.window_size
    head, middle, tail = _centred_frames(waveform, window_size, config.hop_size)
    total = head.shape[1] + middle.shape[1] + tail.shape[1]
    expected = num_frames(waveform.size, window_size, config.hop_size)
    if total != expected:
        raise AssertionError(
            f"framing produced {total} frames but {expected} were predicted; the "
            "centred-STFT frame count and the framing scheme have diverged"
        )

    window = _hann_window(window_size)[:, np.newaxis]
    complex_dtype = np.complex64 if waveform.dtype == np.float32 else np.complex128
    spectrum = np.zeros((1 + window_size // 2, total), dtype=complex_dtype, order="F")

    offset = head.shape[1]
    if offset:
        spectrum[:, :offset] = rfft(window * head, axis=0)
    if tail.shape[1]:
        spectrum[:, -tail.shape[1] :] = rfft(window * tail, axis=0)

    block = max(1, MAX_MEM_BLOCK // (window_size * waveform.itemsize))
    for start in range(0, middle.shape[1], block):
        stop = min(start + block, middle.shape[1])
        spectrum[:, start + offset : stop + offset] = rfft(
            window * middle[:, start:stop], axis=0
        )
    return spectrum


def _power_spectrogram(waveform: np.ndarray, config: MelConfig) -> np.ndarray:
    """Power spectrogram ``|STFT|^2``, matching ``librosa`` bit for bit."""
    return np.abs(_stft(waveform, config)) ** 2


def log_mel_spectrogram(
    waveform: np.ndarray,
    config: MelConfig = DEFAULT_MEL,
) -> np.ndarray:
    """Log-mel spectrogram of an already-windowed waveform.

    The waveform is expected to be the post-self-concatenation signal; this
    function does no filtering, normalisation, windowing or doubling, so that
    ``deployment/generate_bench_input.py`` can feed it a synthetic signal and
    get exactly the tensor the Table 4 benchmark uses.

    Args:
        waveform: 1-D waveform. ``float32`` input reproduces the published
            values; ``float64`` is accepted and computed at full precision.
        config: spectrogram parameters.

    Returns:
        ``(1, time_steps, mel_bins)`` array of decibel-scaled mel energies,
        ``float32``. Channel-first with a single channel.
    """
    waveform = np.ascontiguousarray(waveform)
    if waveform.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {waveform.shape}")
    if not np.issubdtype(waveform.dtype, np.floating):
        raise TypeError(
            f"expected a floating-point waveform, got dtype {waveform.dtype}; "
            "convert integer PCM to float before calling"
        )

    power = _power_spectrogram(waveform, config)
    mel = mel_filterbank(config) @ power
    log_mel = 10.0 * np.log10(np.maximum(mel, LOG_AMIN))
    return log_mel.T[np.newaxis, :, :].astype(np.float32)


#: Historical name of :func:`log_mel_spectrogram`, kept because it is the name
#: the original ``export_onnx.py`` used and the one the analysis and calibration
#: code refers to. Same function, same output shape ``(1, time_steps, mel_bins)``.
waveform_to_mel_spectrogram = log_mel_spectrogram

#: Short name for :func:`zscore_normalise`.
zscore = zscore_normalise


def bandpass(
    audio: np.ndarray,
    sr: int = DEFAULT_MEL.sample_rate,
    low: float = DEFAULT_BANDPASS.low,
    high: float = DEFAULT_BANDPASS.high,
    order: int = DEFAULT_BANDPASS.order,
) -> np.ndarray:
    """Keyword-friendly view of :func:`bandpass_filter`.

    Convenient where the band is being varied for illustration or ablation, as
    in Fig. 1 and Table S10, without constructing a :class:`BandpassConfig`.
    """
    return bandpass_filter(audio, sr, BandpassConfig(low=low, high=high, order=order))


def waveform_to_model_input(
    waveform: np.ndarray,
    config: MelConfig = DEFAULT_MEL,
) -> np.ndarray:
    """Turn one 30 s window into the ONNX graphs' ``(1, 1, 2001, 64)`` input.

    Doubles the window (step 5) and then computes the log-mel spectrogram.
    """
    waveform = np.asarray(waveform)
    if waveform.size != config.clip_samples:
        raise ValueError(
            f"expected exactly {config.clip_samples} samples "
            f"({config.clip_seconds} s at {config.sample_rate} Hz), got {waveform.size}; "
            "use segment_waveform() to window a full recording first"
        )
    doubled = concatenate_with_self(waveform)
    return log_mel_spectrogram(doubled, config)[np.newaxis, ...]


def preprocess_recording(
    audio: np.ndarray,
    sample_rate: int,
    *,
    config: MelConfig = DEFAULT_MEL,
    bandpass: BandpassConfig | None = DEFAULT_BANDPASS,
) -> np.ndarray:
    """Run the complete chain on one recording.

    Args:
        audio: 1-D waveform already resampled to ``config.sample_rate``.
        sample_rate: the waveform's sample rate; must equal
            ``config.sample_rate``, checked rather than silently resampled
            because a mismatch would invalidate the filter and the mel scale.
        config: spectrogram parameters.
        bandpass: filter parameters, or ``None`` for the no-filter ablation.

    Returns:
        ``(n_windows, 1, time_steps, mel_bins)`` float32 array - one batch entry
        per non-overlapping 30 s window, in recording order. Feed it to the ONNX
        session as a batch and mean-pool the logits to reproduce the paper's
        recording-level decision rule.
    """
    if sample_rate != config.sample_rate:
        raise ValueError(
            f"audio is at {sample_rate} Hz but the pipeline is defined at "
            f"{config.sample_rate} Hz; resample before calling"
        )
    filtered = audio if bandpass is None else bandpass_filter(audio, sample_rate, bandpass)
    normalised = zscore_normalise(filtered)
    windows = segment_waveform(normalised.astype(np.float32), config.clip_samples)
    return np.concatenate(
        [waveform_to_model_input(window, config) for window in windows], axis=0
    )
