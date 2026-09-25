# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Some Useful Common Methods
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""Training objective for the murmur classifier.

Split out of the upstream HTS-AT ``src/utils.py`` so that the one loss the paper
uses is defined in one obvious place.

Paper artifact served: every number in Table 1, Table 2, Tables S1-S8 and
Table S10 comes from a model trained with ``weighted_clip_ce``, i.e. softmax
cross-entropy on the three-class head with the inverse-frequency class weights
``[4.0, 10.0, 1.0]`` for Present / Unknown / Absent reported in Sec. 2.2 (the
two-class runs behind Table S7 use ``[4.0, 1.0]``).

Two upstream defects are fixed here:

* the ``clip_ce`` branch raised ``NameError`` because that function was never
  ported, so selecting the loss type named in the config comment crashed;
* an unknown ``loss_type`` fell off the end of ``get_loss_func`` and returned
  ``None``, which surfaced much later as ``TypeError: 'NoneType' object is not
  callable`` inside the first training step. It now raises immediately.

``clip_bce`` and the Asymmetric Loss are not ported: neither has a consumer in
this repository, and ``clip_bce`` additionally assumed sigmoid probabilities
while every model here emits logits.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

__all__ = ["LossFunction", "SUPPORTED_LOSS_TYPES", "get_loss_func", "weighted_clip_ce"]

#: A loss takes ``(logits, target)`` and an optional per-class ``weight``.
LossFunction = Callable[..., torch.Tensor]


def weighted_clip_ce(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-weighted softmax cross-entropy over clip-level logits.

    Args:
        pred: ``(batch_size, classes_num)`` unnormalised logits.
        target: ``(batch_size,)`` integer class indices.
        weight: optional ``(classes_num,)`` per-class weights.

    Returns:
        Scalar mean loss.
    """
    return F.cross_entropy(pred, target, weight=weight)


#: The loss types this release supports. ``config.loss_type`` must name one.
SUPPORTED_LOSS_TYPES: tuple[str, ...] = ("weighted_clip_ce",)

_LOSS_FUNCTIONS: dict[str, LossFunction] = {
    "weighted_clip_ce": weighted_clip_ce,
}


def get_loss_func(loss_type: str) -> LossFunction:
    """Look up a loss function by the name used in the configuration.

    Raises:
        ValueError: if ``loss_type`` is not one of :data:`SUPPORTED_LOSS_TYPES`.
    """
    try:
        return _LOSS_FUNCTIONS[loss_type]
    except KeyError:
        supported = ", ".join(repr(name) for name in SUPPORTED_LOSS_TYPES)
        raise ValueError(
            f"unknown loss_type {loss_type!r}; supported loss types are {supported}"
        ) from None
