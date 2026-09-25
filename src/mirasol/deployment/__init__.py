"""ONNX export and quantization-validation for the MIRASOL heart-murmur models.

Serves Sec. 3.5 (quantization) and Table 4 (on-device latency/memory) of
"A Low-Cost AI Stethoscope for Resource-Constrained Heart Murmur Screening".

* :mod:`mirasol.deployment.export` turns a training checkpoint into the FP32 /
  INT8 / FP16 ONNX graphs published as GitHub Release assets.
* :mod:`mirasol.deployment.validate` compares a quantized graph against its
  FP32 PyTorch baseline, at segment and patient level.
* :mod:`mirasol.deployment.wrappers` holds the traced modules (log-mel in,
  logits out) and carries the HTS-AT / Swin-Transformer attribution.
* :mod:`mirasol.deployment.calibration` supplies the real calibration segments
  and the constant-classifier guard that refuses to publish a graph which
  answers one class for everything.
* :mod:`mirasol.deployment.manifest` records which checkpoint each graph is.

The device-side half of the pipeline lives in ``deployment/edge/`` and is
deliberately outside this package: it must run on a board with no PyTorch.
"""

from __future__ import annotations

__all__ = ["calibration", "export", "manifest", "validate", "wrappers"]
