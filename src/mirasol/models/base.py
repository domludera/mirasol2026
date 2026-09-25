# SPDX-License-Identifier: MIT
"""Common interface shared by the six architectures reported in the paper.

Every architecture in Table 1 (MobileNetV3-S, FastViT, EfficientNet-B0, TinyViT,
ResNet-18, HTS-AT) is an ``nn.Module`` that takes a raw waveform batch and returns
a dict with ``clipwise_output`` (the logits that ``results/predictions/*.data.csv``
records) and ``framewise_output``.  ``BaseAudioModel`` pins that contract, and
``ModelConfig`` documents exactly which configuration keys the models read, so a
typo in ``mirasol.config`` fails at review time rather than at epoch 40.

Paper artifacts served: Table 1, Table 4, Tables S7/S10 - i.e. every trained model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
import torch.nn as nn


class ModelConfig(Protocol):
    """Configuration keys read by the model package.

    ``mirasol.config.defaults`` (and any :class:`ResolvedConfig` derived from it)
    satisfies this protocol structurally; it is declared here only so the required
    key set is written down in one place.

    Signal front end (identical for all six architectures)::

        sample_rate = 4000      # Hz, CirCor native rate
        window_size = 128       # STFT n_fft and win_length
        hop_size    = 120       # 30 s -> 2001 frames
        mel_bins    = 64
        fmin        = 25        # Hz
        fmax        = 2000      # Hz

    Heads and losses::

        classes_num = 3         # Present=0, Unknown=1, Absent=2 (2 in the binary task)
        loss_type   = "weighted_clip_ce"

    HTS-AT only::

        enable_tscam, htsat_spec_size, htsat_patch_size, htsat_stride,
        htsat_window_size, htsat_dim, htsat_depth, htsat_num_head

    One further key is read with :func:`getattr` and therefore optional:
    ``allow_random_init`` (bool, default ``False``).  When true, the timm and
    torchvision builders may fall back to randomly initialised backbones instead of
    raising on a failed pretrained-weight download; see
    :mod:`mirasol.models.timm_models`.
    """

    # Signal front end, shared by all six architectures.
    sample_rate: int
    window_size: int
    hop_size: int
    mel_bins: int
    fmin: float
    fmax: float
    # Head, loss and registry key.
    classes_num: int
    loss_type: str
    model_type: str
    # HTS-AT arm only.
    enable_tscam: bool
    htsat_spec_size: int
    htsat_patch_size: int
    htsat_stride: tuple[int, int]
    htsat_window_size: int
    htsat_dim: int
    htsat_depth: list[int]
    htsat_num_head: list[int]


class BaseAudioModel(ABC, nn.Module):
    """Base class for all audio classification models.

    All models accept a raw waveform batch and return a dictionary with:

    - ``clipwise_output``: ``(batch_size, classes_num)`` unnormalised logits when
      ``loss_type`` is ``clip_ce``/``weighted_clip_ce`` (the released setting), and
      sigmoid activations otherwise.
    - ``framewise_output``: ``(batch_size, num_frames, classes_num)``.  The CNN and
      ViT arms have no frame-level head and return a zero tensor of the right shape
      so downstream code can stay uniform.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mix_lambda: torch.Tensor | None = None,
        infer_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the model on a waveform batch.

        Args:
            x: Input waveform tensor of shape ``(batch_size, num_samples)``.
            mix_lambda: Per-sample mixup weights, or ``None`` to disable mixup.
                Only applied while ``self.training`` is true.
            infer_mode: If true, handle variable-length audio input.  Only the
                HTS-AT arm distinguishes this case; the CNN and ViT arms are
                shape-agnostic and ignore it.

        Returns:
            Dict with keys ``clipwise_output`` and ``framewise_output`` (HTS-AT
            additionally returns ``latent_output``).
        """
        raise NotImplementedError


def zero_framewise_output(
    clipwise_output: torch.Tensor, time_steps: int
) -> torch.Tensor:
    """Return the all-zero ``framewise_output`` used by the non-HTS-AT arms.

    Args:
        clipwise_output: ``(batch_size, classes_num)`` logits, used for shape,
            device and dtype.
        time_steps: Number of mel frames in the input.

    Returns:
        A zero tensor of shape ``(batch_size, time_steps, classes_num)``.
    """
    batch_size, num_classes = clipwise_output.shape
    return torch.zeros(
        (batch_size, time_steps, num_classes),
        device=clipwise_output.device,
        dtype=clipwise_output.dtype,
    )
