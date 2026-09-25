# SPDX-License-Identifier: MIT
"""The six heart-murmur classification architectures reported in the paper.

Import surface::

    from mirasol.models import MODEL_TYPES, create_model

:data:`~mirasol.models.factory.MODEL_TYPES` is the single source of truth for which
architectures exist, in Table 1 column order; drivers, configs, the exporter and the
analysis scripts must read it from here rather than hard-coding a subset.

Names are resolved lazily via :func:`__getattr__` so that ``import mirasol.models``
inside a package ``__init__`` chain does not eagerly pull in torch, timm and
torchvision; the first attribute access does.

Paper artifacts served: Table 1, Table 3, Table 4, Tables S7/S8/S10.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from mirasol.models.base import BaseAudioModel, ModelConfig
    from mirasol.models.factory import (
        MODEL_TYPES,
        PAPER_MODEL_NAMES,
        TIMM_BACKBONES,
        available_models,
        create_model,
    )

_LAZY: dict[str, str] = {
    "BaseAudioModel": "mirasol.models.base",
    "ModelConfig": "mirasol.models.base",
    "MODEL_TYPES": "mirasol.models.factory",
    "PAPER_MODEL_NAMES": "mirasol.models.factory",
    "TIMM_BACKBONES": "mirasol.models.factory",
    "available_models": "mirasol.models.factory",
    "create_model": "mirasol.models.factory",
}

# Listed explicitly rather than derived from _LAZY so static analysis and editors
# can see the public surface; tests assert the two stay in agreement.
__all__ = (
    "BaseAudioModel",
    "MODEL_TYPES",
    "ModelConfig",
    "PAPER_MODEL_NAMES",
    "TIMM_BACKBONES",
    "available_models",
    "create_model",
)


def __getattr__(name: str) -> Any:
    """Resolve the public names in :data:`__all__` on first use.

    Raises:
        AttributeError: If ``name`` is not part of the public surface.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
