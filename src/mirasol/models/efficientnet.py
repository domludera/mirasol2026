# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""EfficientNet-B0 arm, built on torchvision rather than timm.

This is deliberately a separate builder from :mod:`mirasol.models.timm_models`: it
loads **torchvision**'s ``IMAGENET1K_V1`` weights, not timm's
``efficientnet_b0.ra_in1k``.  The two differ, and every released EfficientNet-B0
checkpoint and ONNX graph - including the INT8 build the abstract quotes at 210 ms /
4.3 MB - descends from the torchvision weights, so folding this file into the timm
builder would silently change the published artifacts.

The first convolution is rebuilt for one input channel with its pretrained kernels
averaged over RGB, and the classifier is replaced with a ``classes_num``-way head.
The log-mel front end is shared with the other five architectures via
:mod:`mirasol.models.frontend` (hence the HTS-AT attribution above).

Two fixes relative to the original implementation:

1. The ``try/except ImportError`` ``do_mixup`` fallback, which defined a different
   mixup algorithm (adjacent-pair rather than batch-flip), is deleted; the import from
   :mod:`mirasol.training.utils` is unconditional.
2. The weight download had no error path at all, so an offline machine raised a raw
   ``URLError`` from deep inside torchvision.  It now raises one actionable error, and
   ``allow_random_init`` makes the random-init path explicit rather than accidental.

Paper artifacts served: the EfficientNet-B0 rows of Table 1, Table 4, Sec. 3.5 and
Tables S7/S10.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision.models as models

from mirasol.models.base import BaseAudioModel, ModelConfig, zero_framewise_output
from mirasol.models.frontend import (
    apply_mel_frontend,
    build_mel_frontend,
    to_vision_layout,
)

# The exact torchvision weight enum every released EfficientNet-B0 artifact used.
WEIGHTS = models.EfficientNet_B0_Weights.IMAGENET1K_V1


def _weights_cache_hint() -> str:
    """Describe where torchvision would look for cached pretrained weights."""
    for var in ("TORCH_HOME", "XDG_CACHE_HOME"):
        value = os.environ.get(var)
        if value:
            return f"${var} = {value}"
    return os.path.join("~", ".cache", "torch", "hub", "checkpoints") + " (default)"


class EfficientNetAudio(BaseAudioModel):
    """torchvision EfficientNet-B0 driven by log-mel spectrograms.

    Attributes ``spectrogram_extractor``, ``logmel_extractor``, ``spec_augmenter``,
    ``bn0`` and ``backbone`` are part of the released checkpoints' state_dict keys and
    are read by name in :mod:`mirasol.deployment.export`; do not rename them.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config: Configuration object providing the spectrogram parameters and
                ``classes_num``.  ``allow_random_init`` is read optionally.
        """
        super().__init__()
        self.config = config

        frontend = build_mel_frontend(config)
        self.spectrogram_extractor = frontend.spectrogram_extractor
        self.logmel_extractor = frontend.logmel_extractor
        self.spec_augmenter = frontend.spec_augmenter
        self.bn0 = frontend.bn0

        allow_random_init = bool(getattr(config, "allow_random_init", False))
        weights = None if allow_random_init else WEIGHTS
        try:
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                "Could not load torchvision EfficientNet_B0_Weights.IMAGENET1K_V1. "
                f"Weights are cached in {_weights_cache_hint()}; a machine with no "
                "network access needs that file pre-populated. Every EfficientNet-B0 "
                "number in the paper depends on these weights, so this is a hard "
                "error. Pass --allow-random-init only for shape or latency smoke "
                "tests."
            ) from exc

        # Rebuild the stem for a single-channel mel-spectrogram, averaging the
        # pretrained RGB kernels so the ImageNet initialisation is not thrown away.
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone.features[0][0] = new_conv

        # Replace the 1000-way ImageNet head.
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_ftrs, config.classes_num)

    def forward(
        self,
        x: torch.Tensor,
        mix_lambda: torch.Tensor | None = None,
        infer_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Waveform -> log-mel -> EfficientNet-B0 -> class logits.

        Args:
            x: ``(batch_size, num_samples)`` waveform batch.
            mix_lambda: Mixup weights, applied only while training.
            infer_mode: Unused; EfficientNet-B0 is shape-agnostic and consumes all
                2001 mel frames of a 30 s clip directly.  Accepted so the whole
                registry shares one call signature.

        Returns:
            Dict with ``clipwise_output`` ``(batch_size, classes_num)`` logits and a
            zero ``framewise_output``.
        """
        del infer_mode  # no variable-length special case for this backbone
        x = apply_mel_frontend(self, x, mix_lambda)  # (B, 1, time_steps, mel_bins)
        x = to_vision_layout(x)  # (B, 1, mel_bins, time_steps)

        clipwise_output = self.backbone(x)  # (B, classes_num)
        return {
            'clipwise_output': clipwise_output,
            'framewise_output': zero_framewise_output(clipwise_output, x.size(3)),
        }
