"""The PyTorch-free edge pipeline must reproduce the training preprocessing."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "edge"))
inference_edge = pytest.importorskip("inference_edge")

from export_onnx import waveform_to_mel_spectrogram
from src.prepare_data import bandpass_filter, window_to_clips, zscore_normalise

SR = inference_edge.SAMPLE_RATE


@pytest.mark.parametrize("seconds", [12, 30, 47])
def test_edge_mels_match_training_pipeline(seconds):
    rng = np.random.default_rng(0)
    waveform = rng.standard_normal(seconds * SR).astype(np.float32)

    audio = zscore_normalise(bandpass_filter(waveform, SR))
    expected = [waveform_to_mel_spectrogram(np.concatenate([c, c]).astype(np.float32))
                for c in window_to_clips(audio, inference_edge.CLIP_SAMPLES)]

    mels = inference_edge.preprocess_waveform(waveform)
    assert len(mels) == len(expected) == int(np.ceil(seconds / 30))
    for mel, exp in zip(mels, expected):
        np.testing.assert_allclose(mel[0], exp, rtol=1e-3, atol=1e-2)  # float32 vs float64 filtering
