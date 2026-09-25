# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The configuration for training the model
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""Read-only training and evaluation defaults for every published run.

This module is the descendant of HTS-AT's ``config/esc_config.py``. In the
original workflow it was a *mutable* file that seven driver scripts rewrote in
place with regular expressions, so a checkout recorded whatever the last
experiment happened to leave behind and two folds could not run concurrently.

Here it holds neutral, publishable defaults only:

* ``exp_name``/``model_type`` name the reference configuration
  (EfficientNet-B0, three-class, fold 0), not the last sweep that ran;
* ``resume_checkpoint`` and ``model_checkpoint`` are ``None``, so a fresh clone
  trains instead of failing on a checkpoint that was never distributed;
* keys with no consumer in the released code are gone.

Nothing may write to this module. All parameter changes go through
:mod:`mirasol.config.resolve`, which layers an optional YAML file and repeatable
``--set key=value`` overrides on top and rejects any key not defined here, so a
misspelt parameter fails at startup rather than being silently ignored.

Paper artifacts served: the five-fold cross-validation runs behind Table 1 and
Tables S1-S8, the two-class configuration of Table S7, the class-weight sweep of
Sec. 3.6, and the preprocessing parameters that define the model input in
Fig. 1. Every run's effective values are recorded in
``runs/<exp_name>/resolved_config.json``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "CLASS_NAMES",
    "DEFAULTS",
    "MODEL_TYPES",
    "as_mapping",
]

# ── Experiment identity ──────────────────────────────────────────────────────

#: Directory name under ``runs/`` and stem of the prediction dump. The reference
#: configuration; the drivers set it per model/task/fold.
exp_name: str = "effnetb0_3class_fold0"

#: The six architectures benchmarked in Table 1, in the order the table lists
#: them (smallest to largest, with the audio transformer last).
MODEL_TYPES: Final[tuple[str, ...]] = (
    "mobilenetv3",
    "fastvit",
    "efficientnet_b0",
    "tinyvit",
    "resnet18",
    "htsat",
)

#: One of :data:`MODEL_TYPES`.
model_type: str = "efficientnet_b0"

#: Repo-relative root for run outputs. Authoritative resolution, including the
#: ``MIRASOL_RUNS_DIR`` override, lives in :func:`mirasol.paths.resolve_paths`;
#: this value only supplies the default that resolver uses.
workspace: str = "runs"

#: Repo-relative CirCor download root; see :mod:`mirasol.paths` and
#: ``data/README.md``. Nothing from CirCor is redistributed with this repository.
dataset_path: str = "data/circor"

#: Write the per-segment logits/target/filename dump used by every table script.
save_preds: bool = True

# ── Dataset ──────────────────────────────────────────────────────────────────

#: Selects the ESC-style fold-list dataset wrapper. The only supported value;
#: HTS-AT's "audioset" and "scv2" paths are not part of this release.
dataset_type: str = "esc-50"

#: Draw training batches by cycling over classes rather than uniformly over
#: segments. Active on the ESC path and used for every published run: with 179
#: Present, 68 Unknown and 695 Absent patients, uniform sampling starves the
#: Unknown class. Evaluation is never resampled.
balanced_data: bool = True

#: Prepared fold array produced by ``mirasol.data.prepare``. The three-class
#: 30 s file is the main protocol; the two-class Table S7 runs use
#: ``train-circor-data-30s-binary.npy`` and the Table S10 ablations use the
#: ``ablation-*.npy`` files. The driver always sets this together with
#: ``classes_num`` and ``loss_weight`` - never inherited.
dataset_npy_name: str = "train-circor-data-30s.npy"

#: Held-out fold index, 0-4. Patient-level folds; see ``docs/splits.md``.
esc_fold: int = 0

# ── Objective ────────────────────────────────────────────────────────────────

#: ``"weighted_clip_ce"`` (used for every published run) or ``"clip_ce"``.
loss_type: str = "weighted_clip_ce"

#: Cross-entropy class weights, in :data:`CLASS_NAMES` order. Approximately
#: inverse class frequency; Sec. 3.6 reports the sweep that selected it. Must
#: have ``classes_num`` entries - the two-class configuration uses ``(4.0, 1.0)``.
loss_weight: tuple[float, ...] = (4.0, 10.0, 1.0)

#: Number of output logits. Present=0, Unknown=1, Absent=2 - the order produced
#: by ``mirasol.data.prepare`` and assumed by every exported ONNX graph and by
#: ``deployment/edge/infer.py``. The two-class configuration merges Present and
#: Unknown into class 0 (referable) and Absent becomes class 1.
classes_num: int = 3

#: Human-readable labels for the three-class configuration. Exported so that no
#: script has to restate them; ``mirasol.metrics.CLASS_NAMES`` is the same tuple.
CLASS_NAMES: Final[tuple[str, str, str]] = ("Present", "Unknown", "Absent")

# ── Reproducibility ──────────────────────────────────────────────────────────

#: TRAINING seed: seeds PyTorch, NumPy and Python for weight initialisation and
#: batch sampling. It is *not* the split seed - patients are assigned to folds
#: with :data:`split_seed` in a separate, offline step, so changing this value
#: cannot move a patient between folds.
random_seed: int = 123777

#: SPLIT seed, used once by ``mirasol.data.splits`` to partition patients into
#: five folds. Fixed at 42 for every published number; ``data/splits/folds.csv``
#: is the authoritative record. See ``docs/splits.md``.
split_seed: int = 42

#: Extra diagnostics and a truncated run; never enabled for a published run.
debug: bool = False

# ── Signal processing ────────────────────────────────────────────────────────
# These values define the model input and must stay in step with
# mirasol.signal.mel.MelConfig, which is the implementation. CI asserts equality.

#: CirCor's native rate; no resampling is applied to the published data.
sample_rate: int = 4000

#: Analysis window length in seconds. 30 s was selected in Table S10
#: (10 s < 15 s < 30 s monotonically).
clip_seconds: int = 30

#: Samples per analysis window. Derived - do not override independently.
clip_samples: int = sample_rate * clip_seconds

#: STFT window length in samples (32 ms at 4 kHz).
window_size: int = 128

#: STFT hop in samples. With the 60 s self-concatenated input this yields the
#: 2001-frame time axis baked into every exported ONNX graph.
hop_size: int = 120

#: Mel filterbank size.
mel_bins: int = 64

#: Mel filterbank range in Hz. Wider than the 25-400 Hz bandpass so the
#: spectrogram still shows the roll-off the filter imposes.
fmin: int = 25
fmax: int = 2000

# ── Optimisation ─────────────────────────────────────────────────────────────

batch_size: int = 16
learning_rate: float = 5e-4
max_epoch: int = 150

#: Dataloader workers. Capped at 8 so a default clone does not oversubscribe a
#: large shared machine; the original hard-coded 16.
num_workers: int = min(8, os.cpu_count() or 1)

#: Checkpoints retained per run. Table S8 bounds the optimism of selecting the
#: best of them by held-out accuracy; ``results/checkpoint_accuracies.csv``
#: records the accuracies instead of the 26 GB of weights.
save_top_k: int = 2

#: Validate every N epochs.
val_check_interval: int = 1

#: Step schedule: epoch boundaries and the learning-rate multiplier applied
#: after each. Same length is not required; see the scheduler in
#: ``mirasol.training.lightning_module``.
lr_scheduler_epoch: tuple[int, ...] = (10, 20, 30)
lr_rate: tuple[float, ...] = (0.05, 0.1, 0.2)

#: Lightning accelerator. ``"auto"`` selects CUDA when available and CPU
#: otherwise, so a CPU-only smoke test runs - important for a release aimed at
#: low-resource settings. ``docs/setup.md`` records the GPU used for the
#: published runs.
accelerator: str = "auto"

#: Lightning devices; ``"auto"`` uses every visible device.
devices: str = "auto"

# ── Checkpoints ──────────────────────────────────────────────────────────────

#: Resume training from this checkpoint path, or ``None`` to start from the
#: pretrained backbone. ``None`` in the release: the original value pointed at a
#: local, never-distributed ``.ckpt`` and made a fresh clone fail immediately.
resume_checkpoint: str | None = None

#: Checkpoint to evaluate in ``test`` mode, or ``None`` to require ``--ckpt`` on
#: the command line. Never a committed default, for the same reason.
model_checkpoint: str | None = None

#: Optional Swin-Transformer ImageNet checkpoint for HTS-AT. ``None`` for every
#: published run: HTS-AT was trained from its AudioSet-pretrained weights.
swin_pretrain_path: str | None = None

# ── Model geometry ───────────────────────────────────────────────────────────

#: Emit the token-semantic class-activation map head. True for every published
#: HTS-AT run.
enable_tscam: bool = True

#: HTS-AT / Swin geometry, unchanged from the upstream ESC-50 configuration.
htsat_window_size: int = 8
htsat_spec_size: int = 256
htsat_patch_size: int = 4
htsat_stride: tuple[int, int] = (4, 4)
htsat_num_head: tuple[int, ...] = (4, 8, 16, 32)
htsat_dim: int = 96
htsat_depth: tuple[int, ...] = (2, 2, 6, 2)

#: Framewise (sound-event-detection) outputs. False: this is a clip-level
#: classification task.
fl_local: bool = False


def _collect_defaults() -> Mapping[str, Any]:
    """Snapshot the module's parameter names into an immutable mapping.

    Only lower-case module-level names are parameters; upper-case names are
    fixed vocabularies (:data:`MODEL_TYPES`, :data:`CLASS_NAMES`) that an
    experiment may not override.
    """
    snapshot = {
        name: value
        for name, value in sorted(globals().items())
        if name.islower()
        and not name.startswith("_")
        and isinstance(value, (bool, int, float, str, tuple, type(None)))
    }
    return MappingProxyType(snapshot)


#: The parameter set, immutable. :mod:`mirasol.config.resolve` treats its keys as
#: the complete list of legal override names.
DEFAULTS: Final[Mapping[str, Any]] = _collect_defaults()


def as_mapping() -> Mapping[str, Any]:
    """Return the immutable default parameter mapping."""
    return DEFAULTS
