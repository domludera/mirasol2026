# SPDX-License-Identifier: MIT
"""Single entry point for building the six architectures the paper reports.

:data:`MODEL_TYPES` is the authoritative model list for the whole repository: the
experiment driver's ``--models`` choices, the config's ``model_type`` validation, the
exporter and the analysis scripts all read it from here instead of hard-coding a
subset.  That subset drift is exactly why Table 1 needed two different aggregation
scripts (one covering four architectures, one covering the other two).

The order of :data:`MODEL_TYPES` is Table 1's column order, i.e. ascending trainable
parameter count: 1.52 / 3.26 / 4.01 / 5.07 / 11.17 / 27.57 M.  Regenerate those
headers with ``python -m mirasol.cli.count_params``.

Adding an architecture:
    1. Add a builder module under ``mirasol/models/`` whose class subclasses
       :class:`mirasol.models.base.BaseAudioModel`.
    2. Add one entry to :data:`_BUILDERS` (and to :data:`PAPER_MODEL_NAMES`).

Paper artifacts served: every trained model - Table 1, Table 3, Table 4, Tables
S7/S8/S10.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from mirasol.models.base import BaseAudioModel, ModelConfig
from mirasol.models.efficientnet import EfficientNetAudio
from mirasol.models.htsat import HTSAT_Swin_Transformer
from mirasol.models.timm_models import TimmAudioModel

# Exact timm model names, pretrained tag included, used for every reported run.
# Pinning them here (rather than behind a config override that no config key ever set)
# is what makes the Table 1 parameter counts reproducible.
TIMM_BACKBONES: Mapping[str, str] = MappingProxyType({
    "mobilenetv3": "mobilenetv3_small_100.lamb_in1k",
    "fastvit": "fastvit_t8.apple_in1k",
    "tinyvit": "tiny_vit_5m_224",
    "resnet18": "resnet18.a1_in1k",
})

# How each registry key is spelled in the paper.
PAPER_MODEL_NAMES: Mapping[str, str] = MappingProxyType({
    "mobilenetv3": "MobileNetV3-S",
    "fastvit": "FastViT",
    "efficientnet_b0": "EfficientNet-B0",
    "tinyvit": "TinyViT",
    "resnet18": "ResNet-18",
    "htsat": "HTS-AT",
})


def _build_htsat(config: ModelConfig) -> BaseAudioModel:
    """Build the HTS-AT Swin Transformer from config."""
    return HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head,
    )


def _build_efficientnet_b0(config: ModelConfig) -> BaseAudioModel:
    """Build the torchvision EfficientNet-B0 model from config."""
    return EfficientNetAudio(config)


def _build_timm(model_type: str) -> Callable[[ModelConfig], BaseAudioModel]:
    """Return a builder for the timm backbone registered under ``model_type``."""

    def builder(config: ModelConfig) -> BaseAudioModel:
        return TimmAudioModel(config, TIMM_BACKBONES[model_type])

    builder.__name__ = f"_build_{model_type}"
    builder.__doc__ = f"Build {PAPER_MODEL_NAMES[model_type]} from config."
    return builder


# Registry: model_type string -> builder taking a config and returning an nn.Module.
# Order is Table 1's column order (ascending trainable parameter count).
_BUILDERS: Mapping[str, Callable[[ModelConfig], BaseAudioModel]] = MappingProxyType({
    "mobilenetv3": _build_timm("mobilenetv3"),
    "fastvit": _build_timm("fastvit"),
    "efficientnet_b0": _build_efficientnet_b0,
    "tinyvit": _build_timm("tinyvit"),
    "resnet18": _build_timm("resnet18"),
    "htsat": _build_htsat,
})

#: The six architectures the paper reports, in Table 1 column order.
MODEL_TYPES: tuple[str, ...] = tuple(_BUILDERS)


def available_models() -> tuple[str, ...]:
    """Return :data:`MODEL_TYPES`, for callers that prefer a function."""
    return MODEL_TYPES


def create_model(config: ModelConfig) -> BaseAudioModel:
    """Instantiate the architecture named by ``config.model_type``.

    Args:
        config: Configuration object (``mirasol.config.defaults`` or a
            ``ResolvedConfig`` derived from it) with a ``model_type`` attribute.

    Returns:
        A :class:`~mirasol.models.base.BaseAudioModel` ready to wrap in
        ``mirasol.training.lightning_module.SEDWrapper``.

    Raises:
        ValueError: If ``config`` has no ``model_type``, or names an architecture that
            is not registered.
    """
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError(
            "config has no model_type; expected one of: " + ", ".join(MODEL_TYPES)
        )
    if model_type not in _BUILDERS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Available models: "
            + ", ".join(MODEL_TYPES)
        )
    return _BUILDERS[model_type](config)
