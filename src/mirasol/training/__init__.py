# SPDX-License-Identifier: MIT
"""The training loop, its loss, its prediction dump and quantization-aware training.

Import surface::

    from mirasol.training import SEDWrapper, PredictionWriter, get_loss_func

* :mod:`mirasol.training.lightning_module` is the wrapper that trains all six
  architectures under identical conditions (Sec. 2.2).
* :mod:`mirasol.training.losses` holds the one objective the paper uses,
  class-weighted cross-entropy.
* :mod:`mirasol.training.predictions` writes ``runs/<exp>/predictions.csv`` exactly
  once per run - the file every patient-level table is aggregated from.
* :mod:`mirasol.training.quantization` decides which parts of a model get
  quantized; :mod:`mirasol.training.qat` runs quantization-aware training on top of
  it and refuses to report a fake-quant score as an INT8 score.
* :mod:`mirasol.training.utils` keeps the surviving HTS-AT helpers (mixup,
  interpolation, the parameter record) and the single ``build_config`` adapter.

Names are resolved lazily so that importing this package does not eagerly pull in
torch, Lightning and scikit-learn; the first attribute access does.

Paper artifacts served: every checkpoint and prediction dump behind Table 1,
Table 2, Tables S1-S8 and Table S10, and the QAT sentence of Sec. 3.5.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from mirasol.training.lightning_module import SEDWrapper
    from mirasol.training.losses import SUPPORTED_LOSS_TYPES, get_loss_func
    from mirasol.training.predictions import DEFAULT_PATIENT_RANGE, PredictionWriter
    from mirasol.training.qat import QATSettings, run_qat
    from mirasol.training.quantization import QAT_BACKENDS, QAT_MODES
    from mirasol.training.utils import build_config, dump_config

_LAZY: dict[str, str] = {
    "DEFAULT_PATIENT_RANGE": "mirasol.training.predictions",
    "PredictionWriter": "mirasol.training.predictions",
    "QATSettings": "mirasol.training.qat",
    "QAT_BACKENDS": "mirasol.training.quantization",
    "QAT_MODES": "mirasol.training.quantization",
    "SEDWrapper": "mirasol.training.lightning_module",
    "SUPPORTED_LOSS_TYPES": "mirasol.training.losses",
    "build_config": "mirasol.training.utils",
    "dump_config": "mirasol.training.utils",
    "get_loss_func": "mirasol.training.losses",
    "run_qat": "mirasol.training.qat",
}

# Listed explicitly rather than derived from _LAZY so static analysis and editors
# can see the public surface; tests assert the two stay in agreement.
__all__ = (
    "DEFAULT_PATIENT_RANGE",
    "PredictionWriter",
    "QATSettings",
    "QAT_BACKENDS",
    "QAT_MODES",
    "SEDWrapper",
    "SUPPORTED_LOSS_TYPES",
    "build_config",
    "dump_config",
    "get_loss_func",
    "run_qat",
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
