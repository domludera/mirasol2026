# SPDX-License-Identifier: MIT
"""Dataset preparation, the torch dataset, and the canonical fold recipe.

Three modules, one per concern:

- :mod:`mirasol.data.prepare` turns the CirCor download into the prepared fold file
  (``--num-classes 2|3`` plus the Table S10 ablation flags).
- :mod:`mirasol.data.dataset` is the ``torch.utils.data.Dataset`` the training loop
  reads, including the 30 s self-concatenation that yields ``(1, 2001, 64)``.
- :mod:`mirasol.data.splits` is the ``sorted -> Random(42).shuffle -> i % 5`` recipe
  and the writer/verifier of ``data/splits/folds.csv``.

``mirasol.data.splits`` is deliberately free of torch and librosa, so the split can be
recomputed and checked with the torch-free ``[analysis]`` install.  Names are resolved
lazily here so that importing the recipe does not pull in torch via
:mod:`mirasol.data.dataset`.

Paper artifacts served: ``data/splits/folds.csv``, the input to every trained model,
and Table S10's dataset variants.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from mirasol.data.dataset import ESC_Dataset
    from mirasol.data.prepare import CLASS_NAMES, surviving_patient_ids
    from mirasol.data.splits import (
        EXPECTED_FOLD_SIZES,
        EXPECTED_PATIENT_COUNT,
        N_FOLDS,
        SPLIT_SEED,
        fold_assignment,
        fold_counts,
        observed_partition,
        patients_from_predictions,
        read_folds_csv,
        shuffled_patients,
        write_folds_csv,
    )

_LAZY: dict[str, str] = {
    "CLASS_NAMES": "mirasol.data.prepare",
    "ESC_Dataset": "mirasol.data.dataset",
    "EXPECTED_FOLD_SIZES": "mirasol.data.splits",
    "EXPECTED_PATIENT_COUNT": "mirasol.data.splits",
    "N_FOLDS": "mirasol.data.splits",
    "SPLIT_SEED": "mirasol.data.splits",
    "fold_assignment": "mirasol.data.splits",
    "fold_counts": "mirasol.data.splits",
    "observed_partition": "mirasol.data.splits",
    "patients_from_predictions": "mirasol.data.splits",
    "read_folds_csv": "mirasol.data.splits",
    "shuffled_patients": "mirasol.data.splits",
    "surviving_patient_ids": "mirasol.data.prepare",
    "write_folds_csv": "mirasol.data.splits",
}

# Listed explicitly rather than derived from _LAZY so static analysis and editors
# can see the public surface; tests assert the two stay in agreement.
__all__ = (
    "CLASS_NAMES",
    "ESC_Dataset",
    "EXPECTED_FOLD_SIZES",
    "EXPECTED_PATIENT_COUNT",
    "N_FOLDS",
    "SPLIT_SEED",
    "fold_assignment",
    "fold_counts",
    "observed_partition",
    "patients_from_predictions",
    "read_folds_csv",
    "shuffled_patients",
    "surviving_patient_ids",
    "write_folds_csv",
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
