"""Segment-level prediction dumps, written exactly once per test run.

Every patient-level number in the paper -- Table 1, Table 2, Tables S1-S8 and
Table S10 -- is aggregated from the CSV this module writes: one row per
evaluated 30 s segment, holding the model's clip-level logits, the integer
murmur target and the recording filename. ``results/predictions/*.data.csv`` in
this repository are files of exactly this shape.

Why this module exists rather than the original helper: the original opened the
CSV with ``mode='a'`` while its own comment promised the previous file was
removed. Re-running ``test`` for an existing experiment name therefore appended
a second copy of every row, and the patient-level mean pool then averaged each
recording with itself -- silently, and in a way that changes Unknown-class
metrics. Three of the four drivers had no skip-if-exists guard, and one of them
reused the experiment name that the Table S10 baseline row depends on.

The writer here is a Lightning callback with three properties that make that
class of corruption impossible:

* it opens the file once, in ``on_test_start``, in truncating mode, and refuses
  to clobber an existing file unless ``overwrite=True``;
* it writes to ``<name>.partial`` and renames onto the final path only after the
  integrity checks pass, so a crashed or short run can never leave a file that
  looks complete;
* it asserts the row count equals the number of evaluation segments and that the
  patient count falls in the expected range (188-189 patients per fold for the
  941-patient five-fold split), then prints both.

Schema of the emitted CSV, headerless, comma separated, matching the shipped
prediction dumps byte-for-byte in layout:

    logit_0, ..., logit_{classes_num-1}, target, audio_name

``target`` is written as a float (``2.0``) exactly as the shipped files do;
``audio_name`` is the recording filename, and the patient identifier is the
substring before the first underscore.

Patient-level pooling of these rows is *not* implemented here: it has exactly one
definition, :func:`mirasol.metrics.pool_by_patient`, so the training-time check
and the table scripts cannot drift apart.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pytorch_lightning as pl
import torch

from mirasol.metrics import patient_id_from_filename

__all__ = [
    "AUDIO_NAME_KEY",
    "DEFAULT_PATIENT_RANGE",
    "LOGITS_KEY",
    "PredictionWriter",
    "TARGET_KEY",
]

#: Keys the model's ``test_step`` must return for the writer to consume.
LOGITS_KEY = "logits"
TARGET_KEY = "target"
AUDIO_NAME_KEY = "audio_name"

#: Patients per fold for the paper's five-fold split of 941 patients
#: (measured: 189 / 188 / 188 / 188 / 188). The lower bound is deliberately one
#: below the measured minimum so a legitimately reduced extract still passes
#: while a doubled or truncated dump does not.
DEFAULT_PATIENT_RANGE: tuple[int, int] = (187, 189)


class PredictionWriter(pl.Callback):
    """Write one CSV of segment-level predictions per test run.

    Args:
        output_path: destination CSV, conventionally
            ``runs/<exp_name>/predictions.csv``.
        expected_segments: number of segments the evaluation set contains. The
            caller knows this (``len(eval_dataset)``); when omitted the writer
            recovers it from the trainer's test dataloader and fails if it
            cannot, because an unchecked row count is what allowed the original
            append-mode bug to go unnoticed.
        patient_range: inclusive ``(low, high)`` bounds on the number of distinct
            patients in the dump, or ``None`` to skip that check (ablations that
            keep a single auscultation location per patient still have one
            patient per group, but a custom extract may not).
        overwrite: allow replacing an existing ``output_path``.
    """

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        expected_segments: int | None = None,
        patient_range: tuple[int, int] | None = DEFAULT_PATIENT_RANGE,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        if expected_segments is not None and expected_segments <= 0:
            raise ValueError(f"expected_segments must be positive, got {expected_segments}")
        if patient_range is not None and patient_range[0] > patient_range[1]:
            raise ValueError(f"patient_range must be (low, high), got {patient_range}")
        self._output_path = Path(output_path)
        self._partial_path = self._output_path.with_name(
            self._output_path.name + ".partial"
        )
        self._expected_segments = expected_segments
        self._patient_range = patient_range
        self._overwrite = overwrite
        self._handle: TextIO | None = None
        self._writer: Any = None
        self._rows = 0
        self._patient_targets: dict[str, int] = {}

    # -- properties ------------------------------------------------------- #
    @property
    def output_path(self) -> Path:
        """Path the completed CSV is renamed onto."""
        return self._output_path

    @property
    def rows_written(self) -> int:
        """Segment rows written so far."""
        return self._rows

    @property
    def patient_ids(self) -> tuple[str, ...]:
        """Distinct patient identifiers seen so far, sorted."""
        return tuple(sorted(self._patient_targets))

    # -- Lightning hooks -------------------------------------------------- #
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Open the dump once, truncating, after refusing unsafe situations."""
        if trainer.world_size > 1:
            raise RuntimeError(
                "prediction dumping requires a single process: with "
                f"world_size={trainer.world_size} every rank would write the same "
                "file and the patient-level pool would silently average duplicated "
                "rows. Re-run the test stage with --devices 1."
            )
        if self._output_path.exists() and not self._overwrite:
            raise FileExistsError(
                f"{self._output_path} already exists. Re-running a test into an "
                "existing dump is how patient-level numbers get corrupted; pass "
                "--overwrite to replace it deliberately."
            )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = 0
        self._patient_targets = {}
        self._handle = self._partial_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Append one row per segment in the batch."""
        if self._writer is None:
            raise RuntimeError(
                "PredictionWriter.on_test_batch_end called before on_test_start"
            )
        logits, targets, names = _unpack_step_output(outputs, batch_idx)
        for row, target, name in zip(logits, targets, names, strict=True):
            self._writer.writerow([*(float(v) for v in row), float(target), name])
            self._rows += 1
            pid = patient_id_from_filename(name)
            label = int(target)
            previous = self._patient_targets.setdefault(pid, label)
            if previous != label:
                raise ValueError(
                    f"patient {pid} appears with two targets ({previous} and "
                    f"{label}); the murmur label is defined per patient"
                )

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Close the dump, verify it, then publish it under its final name."""
        self._close()
        expected = self._resolve_expected_segments(trainer)
        n_patients = len(self._patient_targets)
        if self._rows != expected:
            raise ValueError(
                f"{self._partial_path} holds {self._rows} rows but the evaluation "
                f"set has {expected} segments. The dump is not a faithful record "
                "of this run and has been left in place for inspection."
            )
        if self._patient_range is not None:
            low, high = self._patient_range
            if not low <= n_patients <= high:
                raise ValueError(
                    f"{self._partial_path} covers {n_patients} patients, outside the "
                    f"expected range {low}-{high}. Either the split changed or the "
                    "dump is duplicated/truncated; it has been left in place for "
                    "inspection."
                )
        os.replace(self._partial_path, self._output_path)
        distribution = _target_distribution(self._patient_targets)
        print(
            f"predictions: wrote {self._rows} segment rows for {n_patients} patients "
            f"to {self._output_path} (patient targets {distribution})",
            flush=True,
        )

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Close the handle if the test stage raised before ``on_test_end``."""
        self._close()

    # -- internals -------------------------------------------------------- #
    def _close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def _resolve_expected_segments(self, trainer: pl.Trainer) -> int:
        if self._expected_segments is not None:
            return self._expected_segments
        discovered = _dataloader_dataset_length(trainer)
        if discovered is None:
            raise ValueError(
                "cannot verify the row count: the number of evaluation segments was "
                "not supplied and could not be read from the test dataloader. "
                "Construct PredictionWriter with expected_segments=len(eval_dataset)."
            )
        return discovered


def _unpack_step_output(
    outputs: Any, batch_idx: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate and convert one ``test_step`` return value."""
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"test batch {batch_idx} returned {type(outputs).__name__}; "
            f"PredictionWriter needs a mapping with keys {LOGITS_KEY!r}, "
            f"{TARGET_KEY!r} and {AUDIO_NAME_KEY!r}"
        )
    missing = [
        key for key in (LOGITS_KEY, TARGET_KEY, AUDIO_NAME_KEY) if key not in outputs
    ]
    if missing:
        raise KeyError(f"test batch {batch_idx} output is missing keys {missing}")

    logits = _to_numpy(outputs[LOGITS_KEY])
    if logits.ndim != 2:
        raise ValueError(
            f"test batch {batch_idx} logits must be (batch, classes_num), "
            f"got shape {logits.shape}"
        )
    targets = _to_numpy(outputs[TARGET_KEY]).reshape(-1)
    names = tuple(str(name) for name in outputs[AUDIO_NAME_KEY])
    if not logits.shape[0] == targets.shape[0] == len(names):
        raise ValueError(
            f"test batch {batch_idx} is ragged: {logits.shape[0]} logit rows, "
            f"{targets.shape[0]} targets, {len(names)} names"
        )
    return logits, targets, names


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _dataloader_dataset_length(trainer: pl.Trainer) -> int | None:
    dataloaders = getattr(trainer, "test_dataloaders", None)
    if dataloaders is None:
        return None
    if not isinstance(dataloaders, (list, tuple)):
        dataloaders = [dataloaders]
    if len(dataloaders) != 1:
        return None
    dataset = getattr(dataloaders[0], "dataset", None)
    if dataset is None:
        return None
    try:
        return len(dataset)
    except TypeError:
        return None


def _target_distribution(patient_targets: Mapping[str, int]) -> dict[int, int]:
    distribution: dict[int, int] = {}
    for label in patient_targets.values():
        distribution[label] = distribution.get(label, 0) + 1
    return dict(sorted(distribution.items()))
