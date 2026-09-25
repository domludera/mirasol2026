# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""timm-backbone arm: FastViT, TinyViT, MobileNetV3-S and ResNet-18.

Four of the six architectures in Table 1 are ImageNet-pretrained timm backbones with
``in_chans=1`` and a fresh ``classes_num``-way head, fed the shared log-mel front end
from :mod:`mirasol.models.frontend` (whose waveform -> log-mel -> ``bn0`` block is
derived from HTS-AT, hence the attribution above).

Two defects in the original implementation are fixed here, because both could change
a published number without leaving a trace:

1. A ``try: from src.utils import do_mixup / except ImportError:`` fallback defined a
   *different* mixup - adjacent-pair interpolation instead of the batch-flip mixup in
   ``mirasol.training.utils`` - so an import hiccup silently switched augmentation
   algorithms.  The import is now unconditional.
2. A bare ``except Exception`` around ``timm.create_model(..., pretrained=True)``
   printed a message and silently continued with a randomly initialised backbone.  An
   offline machine or a cold weights cache would therefore produce a full, plausible
   set of results for untrained models.  It now raises, naming the weights and the
   cache location; pass ``--allow-random-init`` (config key ``allow_random_init``) to
   opt in deliberately, e.g. for a shape-only smoke test.

Paper artifacts served: the MobileNetV3-S, FastViT, TinyViT and ResNet-18 rows of
Table 1, Table 4 and Tables S7/S10.
"""

from __future__ import annotations

import os

import timm
import torch

from mirasol.models.base import BaseAudioModel, ModelConfig, zero_framewise_output
from mirasol.models.frontend import (
    apply_mel_frontend,
    build_mel_frontend,
    to_vision_layout,
)

# Environment variables timm/Hugging Face consult for a local weights cache, in the
# order huggingface_hub itself resolves them.
_CACHE_ENV_VARS = ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME")


def _weights_cache_hint() -> str:
    """Describe where timm would look for cached pretrained weights.

    Returns:
        A human-readable location, taken from the first cache environment variable
        that is set, or the documented default.
    """
    for var in _CACHE_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return f"${var} = {value}"
    return os.path.join("~", ".cache", "huggingface", "hub") + " (default)"


class TimmAudioModel(BaseAudioModel):
    """A timm image backbone driven by log-mel spectrograms.

    Attributes ``spectrogram_extractor``, ``logmel_extractor``, ``spec_augmenter``,
    ``bn0`` and ``backbone`` are part of the released checkpoints' state_dict keys and
    are read by name in :mod:`mirasol.deployment.export`; do not rename them.
    """

    def __init__(self, config: ModelConfig, model_name: str) -> None:
        """
        Args:
            config: Configuration object providing the spectrogram parameters and
                ``classes_num``.  ``allow_random_init`` is read optionally.
            model_name: Fully qualified timm model name including the pretrained tag,
                e.g. ``"fastvit_t8.apple_in1k"``.  :mod:`mirasol.models.factory` owns
                the mapping from the paper's architecture names to these strings.
        """
        super().__init__()
        self.config = config
        self.model_name = model_name

        frontend = build_mel_frontend(config)
        self.spectrogram_extractor = frontend.spectrogram_extractor
        self.logmel_extractor = frontend.logmel_extractor
        self.spec_augmenter = frontend.spec_augmenter
        self.bn0 = frontend.bn0

        # in_chans=1 because a mel-spectrogram has one channel; timm adapts the first
        # conv layer's pretrained weights by summing over the RGB axis.
        classes_num = config.classes_num
        allow_random_init = bool(getattr(config, "allow_random_init", False))
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=not allow_random_init,
                in_chans=1,
                num_classes=classes_num,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"Could not create timm model {model_name!r} with pretrained="
                f"{not allow_random_init}. Pretrained weights are looked up in "
                f"{_weights_cache_hint()}; a machine with no network access needs them "
                "pre-populated (HF_HUB_OFFLINE=1 plus a warm cache). Every number in "
                "Table 1 depends on these ImageNet weights, so this is a hard error. "
                "Pass --allow-random-init only for shape or latency smoke tests."
            ) from exc

    def forward(
        self,
        x: torch.Tensor,
        mix_lambda: torch.Tensor | None = None,
        infer_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Waveform -> log-mel -> timm backbone -> class logits.

        Args:
            x: ``(batch_size, num_samples)`` waveform batch.
            mix_lambda: Mixup weights, applied only while training.
            infer_mode: Unused; these backbones are shape-agnostic and consume all
                2001 mel frames of a 30 s clip directly.  Accepted so the whole
                registry shares one call signature.

        Returns:
            Dict with ``clipwise_output`` ``(batch_size, classes_num)`` logits and a
            zero ``framewise_output``.
        """
        del infer_mode  # no variable-length special case for these backbones
        x = apply_mel_frontend(self, x, mix_lambda)  # (B, 1, time_steps, mel_bins)
        x = to_vision_layout(x)  # (B, 1, mel_bins, time_steps)

        clipwise_output = self.backbone(x)  # (B, classes_num)
        return {
            'clipwise_output': clipwise_output,
            'framewise_output': zero_framewise_output(clipwise_output, x.size(3)),
        }
