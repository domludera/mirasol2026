"""``deployment/edge/mel.py`` must never drift from ``mirasol.signal.mel``.

The edge directory is deliberately self-contained: a field deployment is one
``scp`` of that folder onto the Orange Pi, with no package install, no PyTorch
and no librosa.  The price of vendoring is duplication - and the two files are
not even textually alike, since the edge copy reimplements the pieces of
librosa that the packaged module gets from SciPy.  If they diverge, the device
classifies a different input representation than every published number was
measured on, silently.

These tests therefore compare *behaviour*, function by function, on identical
inputs, and require exact equality: same constants, same filterbank, same
log-mel, same windowing, same end-to-end tensor.  They also assert the edge
copy stays free of torch, librosa and of any dependency on the installed
package, since the whole point of the directory is that it needs neither.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest
from conftest import BENCH_INPUT, EDGE_MEL, bench_waveform, synthetic_waveform

from mirasol.signal import mel as package
from mirasol.signal.mel import DEFAULT_BANDPASS, DEFAULT_MEL

#: The edge module may import these and nothing heavier.
ALLOWED_EDGE_IMPORTS = frozenset({"__future__", "functools", "typing", "numpy", "scipy"})

SHORT_LENGTHS = (4_096, 28_000, 120_000)


@pytest.fixture(scope="module")
def edge_tree() -> ast.Module:
    return ast.parse(EDGE_MEL.read_text(encoding="utf-8"), filename=str(EDGE_MEL))


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
def test_edge_constants_match_the_published_configuration(edge_mel) -> None:
    assert DEFAULT_MEL.sample_rate == edge_mel.SAMPLE_RATE
    assert DEFAULT_MEL.clip_seconds == edge_mel.CLIP_SECONDS
    assert DEFAULT_MEL.clip_samples == edge_mel.CLIP_SAMPLES
    assert DEFAULT_MEL.window_size == edge_mel.WINDOW_SIZE
    assert DEFAULT_MEL.hop_size == edge_mel.HOP_SIZE
    assert DEFAULT_MEL.mel_bins == edge_mel.MEL_BINS
    assert DEFAULT_MEL.fmin == edge_mel.FMIN
    assert DEFAULT_MEL.fmax == edge_mel.FMAX
    assert edge_mel.AMIN == package.LOG_AMIN
    assert DEFAULT_MEL.time_steps == edge_mel.TIME_STEPS
    assert DEFAULT_MEL.model_samples == edge_mel.CLIP_REPEATS * edge_mel.CLIP_SAMPLES


def test_edge_bandpass_constants_match(edge_mel) -> None:
    assert DEFAULT_BANDPASS.low == edge_mel.BANDPASS_LOW
    assert DEFAULT_BANDPASS.high == edge_mel.BANDPASS_HIGH
    assert DEFAULT_BANDPASS.order == edge_mel.BANDPASS_ORDER


# --------------------------------------------------------------------------
# Spectral front end
# --------------------------------------------------------------------------
def test_mel_filterbanks_are_identical(edge_mel) -> None:
    expected = package.mel_filterbank(DEFAULT_MEL)
    got = edge_mel.mel_filter_bank()
    assert got.shape == expected.shape == (64, 65)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("n_samples", (240_000, *SHORT_LENGTHS))
def test_log_mel_spectrograms_are_identical(edge_mel, n_samples: int) -> None:
    waveform = synthetic_waveform(n_samples)
    expected = package.log_mel_spectrogram(waveform, DEFAULT_MEL)
    got = edge_mel.log_mel_spectrogram(waveform)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype == np.float32
    assert np.array_equal(got, expected)


def test_model_input_tensors_are_identical(edge_mel) -> None:
    """The tensor handed to ``onnxruntime`` must be the same on both paths."""
    window = synthetic_waveform(DEFAULT_MEL.clip_samples)
    expected = package.waveform_to_model_input(window, DEFAULT_MEL)
    got = edge_mel.clip_to_model_input(window)
    assert got.shape == expected.shape == DEFAULT_MEL.input_shape
    assert np.array_equal(got, expected)


def test_both_paths_reproduce_the_committed_benchmark_input(edge_mel) -> None:
    """The Table 4 input tensor, regenerated through the edge module."""
    expected = np.load(BENCH_INPUT)
    got = edge_mel.clip_to_model_input(bench_waveform(edge_mel.CLIP_SAMPLES))
    assert np.array_equal(got, expected)


# --------------------------------------------------------------------------
# Waveform conditioning
# --------------------------------------------------------------------------
def test_bandpass_filters_are_identical(edge_mel) -> None:
    waveform = synthetic_waveform(10 * DEFAULT_MEL.sample_rate)
    expected = package.bandpass_filter(waveform, DEFAULT_MEL.sample_rate)
    got = edge_mel.bandpass_filter(waveform)
    assert np.array_equal(got, expected)


def test_zscore_normalisers_are_identical(edge_mel) -> None:
    waveform = synthetic_waveform(40_000)
    assert np.array_equal(
        edge_mel.zscore_normalise(waveform), package.zscore_normalise(waveform)
    )
    silence = np.zeros(1_000, dtype=np.float32)
    assert np.array_equal(
        edge_mel.zscore_normalise(silence), package.zscore_normalise(silence)
    )


@pytest.mark.parametrize("n_samples", [3, 4_096, 28_000, 120_000, 190_000, 240_001])
def test_tiling_is_identical(edge_mel, n_samples: int) -> None:
    waveform = synthetic_waveform(n_samples)
    assert np.array_equal(
        edge_mel.tile_to_length(waveform, DEFAULT_MEL.clip_samples),
        package.tile_to_length(waveform, DEFAULT_MEL.clip_samples),
    )


@pytest.mark.parametrize("n_samples", [28_000, 120_000, 190_000, 240_000, 305_000])
def test_windowing_is_identical(edge_mel, n_samples: int) -> None:
    waveform = synthetic_waveform(n_samples)
    expected = package.segment_waveform(waveform, DEFAULT_MEL.clip_samples)
    got = edge_mel.segment_waveform(waveform, DEFAULT_MEL.clip_samples)
    assert len(got) == len(expected)
    for clip, reference in zip(got, expected, strict=True):
        assert np.array_equal(clip, reference)


@pytest.mark.parametrize("bandpass", [True, False])
def test_full_recording_pipelines_are_identical(edge_mel, bandpass: bool) -> None:
    """Including the no-filter variant used for the Table S10 ablation."""
    waveform = synthetic_waveform(190_000)
    expected = package.preprocess_recording(
        waveform,
        DEFAULT_MEL.sample_rate,
        config=DEFAULT_MEL,
        bandpass=DEFAULT_BANDPASS if bandpass else None,
    )
    got = np.concatenate(
        edge_mel.preprocess_recording(
            waveform, DEFAULT_MEL.sample_rate, DEFAULT_MEL.clip_samples, bandpass
        )
    )
    assert got.shape == expected.shape
    assert got.shape[1:] == DEFAULT_MEL.input_shape[1:]
    assert np.array_equal(got, expected)


def test_both_modules_reject_a_wrong_sample_rate(edge_mel) -> None:
    """A resample would invalidate both the filter and the mel scale."""
    waveform = synthetic_waveform(8_000)
    with pytest.raises(ValueError):
        edge_mel.preprocess_recording(waveform, 8_000)
    with pytest.raises(ValueError):
        package.preprocess_recording(waveform, 8_000)


# --------------------------------------------------------------------------
# Import hygiene
# --------------------------------------------------------------------------
def _imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_edge_module_imports_nothing_the_pi_would_have_to_install(
    edge_tree: ast.Module,
) -> None:
    """``requirements-edge.txt`` is onnxruntime, numpy, scipy and soundfile."""
    unexpected = sorted(_imported_roots(edge_tree) - ALLOWED_EDGE_IMPORTS)
    assert unexpected == [], (
        f"deployment/edge/mel.py imports {unexpected}; the edge directory must "
        "run from requirements-edge.txt alone and must not need the package"
    )


def test_edge_module_declares_itself_a_vendored_copy(edge_tree: ast.Module) -> None:
    """A reader of the file must be told which module it duplicates."""
    docstring = ast.get_docstring(edge_tree) or ""
    assert "vendored" in docstring.lower()
    assert "mirasol/signal/mel.py" in docstring
