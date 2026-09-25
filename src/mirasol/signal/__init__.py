"""Torch-free signal processing: the one definition of the paper's model input.

:mod:`mirasol.signal.mel` holds the complete preprocessing chain described in
Sec. 2.2 and drawn in Fig. 1 of the paper - 4th-order Butterworth bandpass,
per-recording z-score, 30 s tile-and-crop windowing, concatenate-with-self, and
the 64-bin log-mel spectrogram that every trained model and every exported ONNX
graph consumes.

The module depends only on NumPy and SciPy so that the identical code runs on
the Orange Pi Zero 2W, where neither PyTorch nor librosa is installed. It is
vendored verbatim as ``deployment/edge/mel.py`` and the two copies are checked
for parity in CI.
"""

from __future__ import annotations

__all__: list[str] = []
