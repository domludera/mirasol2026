# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Some Useful Common Methods
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""Live training helpers shared by the Lightning module, the CLI and QAT.

This is the surviving half of the upstream HTS-AT ``src/utils.py``: the mixup
primitives that ``mirasol.models`` calls, the framewise interpolation that
``mirasol.models.htsat`` calls, and the provenance helpers (``create_folder`` /
``dump_config``) that write ``runs/<exp>/<exp>.json`` for every run.

Paper artifacts served: every trained model behind Table 1, Table 2, Table S7
and Table S10 goes through ``dump_config``, which is the only machine-readable
record of the parameters a run actually used.

``build_config`` is the single adapter onto
:func:`mirasol.config.resolve.resolve_config`, so the CLI entry points, the
experiment drivers and the QAT library all obtain a configuration the same way.
Output locations are not resolved here: they come from
:class:`mirasol.paths.Paths` (``config.run_dir``, ``config.dataset_npy``), which
is the single path resolver for the whole repository.

Deleted relative to upstream, all of it dead on the CirCor path: ``import
museval`` (a music-source-separation package with no uses that was nonetheless
a hard install requirement), ``AsymmetricLoss``, ``process_idc``, ``clip_bce``,
``d_prime``, ``do_mixup_label``, ``pad_framewise_output``, ``float32_to_int16``,
``init_hier_head`` and ``prepprocess_audio`` (which additionally called the
removed positional ``librosa.resample`` signature). Loss selection moved to
:mod:`mirasol.training.losses`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from mirasol.config.resolve import ResolvedConfig
    from mirasol.paths import Paths

__all__ = [
    "build_config",
    "config_to_dict",
    "create_folder",
    "do_mixup",
    "dump_config",
    "get_mix_lambda",
    "int16_to_float32",
    "interpolate",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def build_config(
    config_path: str | os.PathLike[str] | None = None,
    overrides: Sequence[str] = (),
    paths: Paths | None = None,
) -> ResolvedConfig:
    """Build an immutable resolved configuration.

    Single adapter onto :mod:`mirasol.config.resolve`, so the CLI entry points,
    the experiment drivers and the QAT library all obtain a configuration
    through one call site.

    Args:
        config_path: optional flat YAML file of overrides (``--config``).
        overrides: ``key=value`` strings (``--set``). Keys that do not already
            exist in the defaults are rejected, so a typo fails at startup
            instead of silently doing nothing -- the failure mode of the seven
            scripts that used to regex-rewrite ``config/esc_config.py``.
        paths: filesystem locations; defaults to the environment and the
            repo-relative defaults in :mod:`mirasol.paths`.
    """
    # Imported here because mirasol.config imports nothing from mirasol.training;
    # keeping it local documents that this is the only coupling point.
    from mirasol.config.resolve import resolve_config

    return resolve_config(
        config_file=config_path, set_overrides=tuple(overrides), paths=paths
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    """Return a JSON-serialisable mapping of a configuration's parameters.

    Prefers the ``as_dict()`` of :class:`mirasol.config.resolve.ResolvedConfig`;
    falls back to scanning public attributes so a plain module or namespace also
    works (which is what the upstream helper assumed).
    """
    as_dict = getattr(config, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    serialisable: dict[str, Any] = {}
    for key in dir(config):
        if key.startswith("_"):
            continue
        value = getattr(config, key)
        if isinstance(value, (int, float, str, list, dict, bool, type(None))):
            serialisable[key] = value
    return serialisable


def create_folder(folder: str | os.PathLike[str]) -> Path:
    """Create ``folder`` (and parents) if it does not exist and return it."""
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_config(
    config: Any, filename: str | os.PathLike[str], include_time: bool = False
) -> Path:
    """Write the resolved configuration of a run to ``<filename>.json``.

    This is the provenance record for a run: the JSON next to the checkpoints
    is what lets a fold be traced back to the exact parameters it used. The
    ``.json`` suffix is appended by this function, matching the upstream
    behaviour that produced the run records shipped with the paper.
    """
    target = Path(filename)
    if include_time:
        target = target.with_name(
            target.name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )
    target = target.with_name(target.name + ".json")
    create_folder(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=4, sort_keys=True)
    return target


# --------------------------------------------------------------------------- #
# Signal / tensor helpers
# --------------------------------------------------------------------------- #
def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert int16 PCM samples to float32 in ``[-1, 1)``."""
    return (x / 32767.0).astype(np.float32)


def get_mix_lambda(mixup_alpha: float, batch_size: int) -> np.ndarray:
    """Draw one Beta(alpha, alpha) mixup coefficient per batch element."""
    mixup_lambdas: Iterable[float] = (
        float(np.random.beta(mixup_alpha, mixup_alpha, 1)[0]) for _ in range(batch_size)
    )
    return np.array(list(mixup_lambdas), dtype=np.float32)


def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor) -> torch.Tensor:
    """Mix each batch element with the batch reversed along dim 0.

    Args:
        x: ``(batch_size, ...)`` tensor.
        mixup_lambda: ``(batch_size,)`` mixing coefficients.

    Returns:
        A tensor with the same shape as ``x``.
    """
    return (
        x.transpose(0, -1) * mixup_lambda
        + torch.flip(x, dims=[0]).transpose(0, -1) * (1 - mixup_lambda)
    ).transpose(0, -1)


def interpolate(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """Repeat along time to undo a CNN's temporal downsampling.

    Args:
        x: ``(batch_size, time_steps, classes_num)``.
        ratio: integer upsampling factor.

    Returns:
        ``(batch_size, time_steps * ratio, classes_num)``.
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    return upsampled.reshape(batch_size, time_steps * ratio, classes_num)
