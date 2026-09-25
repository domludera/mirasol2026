# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""Shared log-mel front end for the CNN and ViT arms.

The waveform -> log-mel -> ``bn0`` block below is copied from
``model/htsat.py`` lines 453-463 of the HTS-AT reference implementation and was
duplicated verbatim in ``model/timm_models.py`` and ``model/efficientnet.py``.
It is defined once here so that all six architectures in Table 1 are guaranteed to
see the identical input representation; :mod:`mirasol.signal.mel` is the torch-free
mirror of the same computation used for ONNX export and on-device inference.

Two properties are load-bearing and must not change:

1. **Attribute names.** The builders return the modules but do not own them: each
   model assigns them to ``self.spectrogram_extractor``, ``self.logmel_extractor``,
   ``self.spec_augmenter`` and ``self.bn0``.  Those names appear in the state_dict of
   every released checkpoint (torchlibrosa's STFT and mel filter bank hold ~20.9k
   frozen parameters, ``bn0`` holds 128), and ``mirasol.deployment.export`` reads
   ``model.bn0`` and ``model.backbone`` directly.  Nesting them inside a container
   module would silently invalidate all 30 checkpoints.
2. **Log-mel parameters.** ``ref=1.0``, ``amin=1e-10``, ``top_db=None`` - i.e. plain
   ``10 * log10(max(mel, 1e-10))`` with no dynamic-range clamp.  ``top_db=None`` is
   what makes the numpy reimplementation in :mod:`mirasol.signal.mel` exact.

Paper artifacts served: the input representation behind Table 1, Table 2, Table 4
and Tables S1-S10.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from mirasol.models.base import ModelConfig
from mirasol.training.utils import do_mixup

# torchlibrosa STFT settings, fixed for every reported run.
_WINDOW = "hann"
_CENTER = True
_PAD_MODE = "reflect"

# LogmelFilterBank settings.  top_db=None disables the dynamic-range clamp, so the
# output is exactly 10 * log10(max(power_mel, amin)) and is reproducible in numpy.
_REF = 1.0
_AMIN = 1e-10
_TOP_DB: float | None = None

# SpecAugmentation settings, scaled for 64 mel bins (HTS-AT default was 128).
_TIME_DROP_WIDTH = 64
_TIME_STRIPES_NUM = 2
_FREQ_DROP_WIDTH = 4
_FREQ_STRIPES_NUM = 2


class MelFrontend(NamedTuple):
    """The four front-end submodules, in the order they are applied."""

    spectrogram_extractor: Spectrogram
    logmel_extractor: LogmelFilterBank
    spec_augmenter: SpecAugmentation
    bn0: nn.BatchNorm2d


class HasMelFrontend(Protocol):
    """Structural type of a model that owns a :class:`MelFrontend`."""

    training: bool
    spectrogram_extractor: Spectrogram
    logmel_extractor: LogmelFilterBank
    spec_augmenter: SpecAugmentation
    bn0: nn.BatchNorm2d


def build_mel_frontend(config: ModelConfig) -> MelFrontend:
    """Build the shared front-end submodules from ``config``.

    The caller must assign the four returned modules to the attribute names given by
    :class:`MelFrontend`, because those names are baked into the released
    checkpoints.

    Args:
        config: Configuration providing ``window_size``, ``hop_size``,
            ``sample_rate``, ``mel_bins``, ``fmin`` and ``fmax``.

    Returns:
        A :class:`MelFrontend` of freshly constructed modules.
    """
    spectrogram_extractor = Spectrogram(
        n_fft=config.window_size,
        hop_length=config.hop_size,
        win_length=config.window_size,
        window=_WINDOW,
        center=_CENTER,
        pad_mode=_PAD_MODE,
        freeze_parameters=True,
    )
    logmel_extractor = LogmelFilterBank(
        sr=config.sample_rate,
        n_fft=config.window_size,
        n_mels=config.mel_bins,
        fmin=config.fmin,
        fmax=config.fmax,
        ref=_REF,
        amin=_AMIN,
        top_db=_TOP_DB,
        freeze_parameters=True,
    )
    spec_augmenter = SpecAugmentation(
        time_drop_width=_TIME_DROP_WIDTH,
        time_stripes_num=_TIME_STRIPES_NUM,
        freq_drop_width=_FREQ_DROP_WIDTH,
        freq_stripes_num=_FREQ_STRIPES_NUM,
    )
    bn0 = nn.BatchNorm2d(config.mel_bins)
    return MelFrontend(
        spectrogram_extractor=spectrogram_extractor,
        logmel_extractor=logmel_extractor,
        spec_augmenter=spec_augmenter,
        bn0=bn0,
    )


def apply_mel_frontend(
    model: HasMelFrontend,
    x: torch.Tensor,
    mix_lambda: torch.Tensor | None = None,
) -> torch.Tensor:
    """Turn a waveform batch into a batch-normalised log-mel spectrogram.

    SpecAugment and mixup are applied only while ``model.training`` is true, which is
    why evaluation is deterministic even though ``deterministic=False`` during
    training.

    Args:
        model: The owning module; must expose the four :class:`MelFrontend`
            attributes and the standard ``training`` flag.
        x: Waveform batch of shape ``(batch_size, num_samples)``.
        mix_lambda: Per-sample mixup weights, or ``None``.

    Returns:
        Tensor of shape ``(batch_size, 1, time_steps, mel_bins)``.  For the released
        30 s clips at ``sample_rate=4000`` and ``hop_size=120`` this is
        ``(batch_size, 1, 2001, 64)``.
    """
    x = model.spectrogram_extractor(x)  # (B, 1, time_steps, freq_bins)
    x = model.logmel_extractor(x)  # (B, 1, time_steps, mel_bins)

    # bn0 normalises across mel bins, so the mel axis is moved to the channel axis.
    x = x.transpose(1, 3)  # (B, mel_bins, time_steps, 1)
    x = model.bn0(x)
    x = x.transpose(1, 3)  # (B, 1, time_steps, mel_bins)

    if model.training:
        x = model.spec_augmenter(x)
    if model.training and mix_lambda is not None:
        x = do_mixup(x, mix_lambda)
    return x


def to_vision_layout(x: torch.Tensor) -> torch.Tensor:
    """Reorder ``(B, 1, time_steps, mel_bins)`` to the layout ImageNet backbones want.

    timm and torchvision backbones read ``(batch, channels, height, width)``; the
    released models map ``height=mel_bins`` and ``width=time_steps``.  The same
    transpose is reproduced in ``mirasol.deployment.export``'s wrappers, so changing
    it here would invalidate every exported ONNX graph.

    Args:
        x: Tensor of shape ``(batch_size, 1, time_steps, mel_bins)``.

    Returns:
        Tensor of shape ``(batch_size, 1, mel_bins, time_steps)``.
    """
    return x.transpose(2, 3)
