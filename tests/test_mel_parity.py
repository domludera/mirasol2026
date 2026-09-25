"""``mirasol.signal.mel`` must reproduce the pre-release preprocessing exactly.

Every published number - the five-fold tables, the ONNX exports, the Orange Pi
latencies of Table 4 - was produced through the log-mel front end that the
pre-release tree implemented three times over (in ``export_onnx.py``,
``inference_edge.py`` and ``make_figures.py``) on top of librosa.  The release
consolidates them into one librosa-free module, so these tests assert that the
consolidation did not move a single mel bin.

Three anchors, in decreasing order of availability:

1. ``deployment/edge/bench_input.npy``, committed, is the log-mel of a
   fixed-seed synthetic waveform and the only input to the Table 4 benchmark.
   Reproducing it bit-for-bit needs nothing but the repository.
2. librosa, if installed, gives the reference ``stft`` and ``filters.mel`` the
   original implementation called.
3. the pre-release working tree, if ``MIRASOL_LEGACY_SOURCE`` points at it,
   gives the original functions themselves, compiled out of their modules
   without importing torch.

Anchors 2 and 3 skip cleanly when unavailable, so a clone's CI run is green.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    BENCH_INPUT,
    BENCH_INPUT_ATOL_DB,
    bench_waveform,
    legacy_config,
    load_legacy_function,
    require_legacy_source,
    synthetic_waveform,
)

from mirasol.signal.mel import (
    DEFAULT_BANDPASS,
    DEFAULT_MEL,
    bandpass_filter,
    concatenate_with_self,
    log_mel_spectrogram,
    mel_filterbank,
    num_frames,
    segment_waveform,
    tile_to_length,
    waveform_to_model_input,
    zscore_normalise,
)

#: The tensor every exported graph consumes: one channel, 2001 frames of the
#: self-concatenated 60 s signal at 4 kHz with hop 120, 64 mel bins.
PUBLISHED_INPUT_SHAPE = (1, 1, 2001, 64)


# --------------------------------------------------------------------------
# Anchor 1: the committed benchmark tensor
# --------------------------------------------------------------------------
def test_published_configuration_is_the_paper_s() -> None:
    """4 kHz, 30 s, window 128, hop 120, 64 mel bins, 25-2000 Hz."""
    assert (DEFAULT_MEL.sample_rate, DEFAULT_MEL.clip_seconds) == (4000, 30)
    assert (DEFAULT_MEL.window_size, DEFAULT_MEL.hop_size) == (128, 120)
    assert (DEFAULT_MEL.mel_bins, DEFAULT_MEL.fmin, DEFAULT_MEL.fmax) == (
        64,
        25.0,
        2000.0,
    )
    assert DEFAULT_MEL.clip_samples == 120_000
    assert DEFAULT_MEL.model_samples == 240_000
    assert DEFAULT_MEL.time_steps == 2001
    assert DEFAULT_MEL.input_shape == PUBLISHED_INPUT_SHAPE
    assert (DEFAULT_BANDPASS.low, DEFAULT_BANDPASS.high, DEFAULT_BANDPASS.order) == (
        25.0,
        400.0,
        4,
    )


def test_reproduces_the_committed_benchmark_input() -> None:
    """``deployment/edge/bench_input.npy`` is the Table 4 benchmark's only input.

    Regenerated here from the same fixed-seed synthetic waveform the
    pre-release ``generate_bench_input.py`` used.  Agreement means the
    latency figures in Table 4 refer to this exact preprocessing chain.  The
    file was written on the author's workstation; FFT/SIMD code paths differ
    across CPUs, so the comparison allows float32 rounding (in dB) rather than
    byte equality.
    """
    if not BENCH_INPUT.is_file():
        pytest.fail(f"{BENCH_INPUT.name} is missing from deployment/edge/")
    expected = np.load(BENCH_INPUT)
    got = waveform_to_model_input(bench_waveform(DEFAULT_MEL.clip_samples))

    assert expected.shape == PUBLISHED_INPUT_SHAPE
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=0, atol=BENCH_INPUT_ATOL_DB)


def test_model_input_rejects_a_wrongly_sized_window() -> None:
    """A 30 s window is required; silent truncation would change the input."""
    with pytest.raises(ValueError):
        waveform_to_model_input(synthetic_waveform(119_999))


def test_log_mel_floor_matches_torchlibrosa() -> None:
    """``10*log10(max(mel, 1e-10))``, no top-dB clipping: silence is -100 dB."""
    silence = np.zeros(DEFAULT_MEL.model_samples, dtype=np.float32)
    assert np.allclose(log_mel_spectrogram(silence), -100.0)


def test_frame_count_follows_the_centred_stft_rule() -> None:
    assert num_frames(240_000, 128, 120) == 2001
    assert num_frames(120_000, 128, 120) == 1001


def test_bandpassed_energy_concentrates_in_the_low_mel_bins() -> None:
    """Murmur energy occupies the lowest ~20 of 64 bins (Fig. 1 caption)."""
    filtered = bandpass_filter(
        synthetic_waveform(DEFAULT_MEL.model_samples), DEFAULT_MEL.sample_rate
    )
    mel = log_mel_spectrogram(filtered.astype(np.float32))[0].mean(axis=0)
    assert mel[:20].mean() > mel[20:].mean() + 10.0


# --------------------------------------------------------------------------
# Anchor 2: librosa, the original reference implementation
# --------------------------------------------------------------------------
def test_mel_filterbank_matches_librosa() -> None:
    """Slaney scale, area normalisation, float32 - librosa's defaults."""
    librosa = pytest.importorskip("librosa")
    expected = librosa.filters.mel(
        sr=DEFAULT_MEL.sample_rate,
        n_fft=DEFAULT_MEL.window_size,
        n_mels=DEFAULT_MEL.mel_bins,
        fmin=DEFAULT_MEL.fmin,
        fmax=DEFAULT_MEL.fmax,
    )
    assert np.array_equal(mel_filterbank(DEFAULT_MEL), expected)


@pytest.mark.parametrize("n_samples", [240_000, 120_000, 28_000, 4_096])
def test_log_mel_matches_a_librosa_implementation(n_samples: int) -> None:
    """The librosa chain the original ``export_onnx.py`` used, verbatim."""
    librosa = pytest.importorskip("librosa")
    waveform = synthetic_waveform(n_samples)

    stft = librosa.stft(
        waveform,
        n_fft=DEFAULT_MEL.window_size,
        hop_length=DEFAULT_MEL.hop_size,
        win_length=DEFAULT_MEL.window_size,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    mel = librosa.filters.mel(
        sr=DEFAULT_MEL.sample_rate,
        n_fft=DEFAULT_MEL.window_size,
        n_mels=DEFAULT_MEL.mel_bins,
        fmin=DEFAULT_MEL.fmin,
        fmax=DEFAULT_MEL.fmax,
    ) @ (np.abs(stft) ** 2)
    expected = (10.0 * np.log10(np.maximum(mel, 1e-10))).T[np.newaxis, :, :].astype(
        np.float32
    )

    assert np.array_equal(log_mel_spectrogram(waveform), expected)


# --------------------------------------------------------------------------
# Anchor 3: the pre-release working tree
# --------------------------------------------------------------------------
def test_log_mel_matches_the_legacy_export_function() -> None:
    """``export_onnx.waveform_to_mel_spectrogram``, compiled without torch."""
    root = require_legacy_source()
    pytest.importorskip("librosa")
    config = legacy_config(root)
    legacy = load_legacy_function(
        root,
        "export_onnx.py",
        "waveform_to_mel_spectrogram",
        {"np": np, "config": config},
    )
    assert (config.window_size, config.hop_size) == (
        DEFAULT_MEL.window_size,
        DEFAULT_MEL.hop_size,
    )
    assert (config.mel_bins, config.fmin, config.fmax) == (
        DEFAULT_MEL.mel_bins,
        DEFAULT_MEL.fmin,
        DEFAULT_MEL.fmax,
    )
    # clip_samples is a derived expression in the legacy config, not a literal.
    clip_samples = config.sample_rate * config.clip_seconds
    assert clip_samples == DEFAULT_MEL.clip_samples

    waveform = concatenate_with_self(synthetic_waveform(clip_samples))
    expected = np.asarray(legacy(waveform))
    assert np.array_equal(log_mel_spectrogram(waveform), expected)


def test_bandpass_matches_the_legacy_preparation_filter() -> None:
    """Causal 4th-order Butterworth, 25-400 Hz, via ``sosfilt``."""
    root = require_legacy_source()
    from scipy import signal as scipy_signal

    legacy = load_legacy_function(
        root,
        "src/prepare_data.py",
        "bandpass_filter",
        {"np": np, "signal": scipy_signal},
    )
    waveform = synthetic_waveform(10 * DEFAULT_MEL.sample_rate)
    expected = legacy(
        waveform,
        DEFAULT_MEL.sample_rate,
        DEFAULT_BANDPASS.low,
        DEFAULT_BANDPASS.high,
        DEFAULT_BANDPASS.order,
    )
    got = bandpass_filter(waveform, DEFAULT_MEL.sample_rate)
    assert np.array_equal(got, np.asarray(expected))


def test_zscore_matches_the_legacy_normaliser() -> None:
    root = require_legacy_source()
    legacy = load_legacy_function(
        root, "src/prepare_data.py", "zscore_normalise", {"np": np}
    )
    waveform = synthetic_waveform(40_000)
    assert np.array_equal(zscore_normalise(waveform), np.asarray(legacy(waveform)))
    silence = np.zeros(1_000, dtype=np.float32)
    assert np.array_equal(zscore_normalise(silence), np.asarray(legacy(silence)))


@pytest.mark.parametrize("n_samples", [28_000, 120_000, 190_000, 240_000])
def test_windowing_matches_the_legacy_clip_splitter(n_samples: int) -> None:
    """Short recordings and short tails are repeat-tiled, never zero-padded.

    ``inference_edge.py`` zero-padded instead, which put inputs unlike anything
    seen in training through the deployed graphs; this is the regression guard
    for that fix.
    """
    root = require_legacy_source()
    legacy = load_legacy_function(
        root, "src/prepare_data.py", "window_to_clips", {"np": np}
    )
    waveform = synthetic_waveform(n_samples)
    expected = [np.asarray(clip) for clip in legacy(waveform, DEFAULT_MEL.clip_samples)]
    got = segment_waveform(waveform, DEFAULT_MEL.clip_samples)

    assert len(got) == len(expected)
    for clip, reference in zip(got, expected, strict=True):
        assert clip.shape == (DEFAULT_MEL.clip_samples,)
        assert np.array_equal(clip, reference)
    # A tiled tail is never silent; a zero-padded one would be.
    assert np.abs(got[-1][-DEFAULT_MEL.sample_rate :]).max() > 0.0


def test_tile_to_length_repeats_rather_than_pads() -> None:
    waveform = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert tile_to_length(waveform, 7).tolist() == [1, 2, 3, 1, 2, 3, 1]
    with pytest.raises(ValueError):
        tile_to_length(np.asarray([], dtype=np.float32), 4)
