# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The main code for training and evaluating HTSAT
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""Train or evaluate one architecture on one fold.

This is the process that produced every checkpoint and every prediction dump
behind Table 1, Table 2, Tables S1-S8 and Table S10. Architecture, fold and
labelling scheme all come from the resolved configuration, so
``experiments/run_experiment.py`` only has to build an argv -- no script rewrites
a tracked configuration file any more.

Usage::

    python -m mirasol.cli.train train --set model_type=efficientnet_b0 \\
        --set esc_fold=0 --set exp_name=efficientnet_b0_3class_fold0
    python -m mirasol.cli.train test --set model_type=efficientnet_b0 \\
        --set esc_fold=0 --set exp_name=efficientnet_b0_3class_fold0 \\
        --set resume_checkpoint=runs/.../l-epoch=97-acc=0.871.ckpt

A CPU-only smoke test is supported and is the cheapest way to check an install::

    python -m mirasol.cli.train test --accelerator cpu --devices 1 \\
        --set num_workers=0 --no-expect-patients

Outputs, all under the untracked ``runs/<exp_name>/``: ``resolved_config.json``
(the full provenance record), ``<exp_name>.json`` (the flat parameter dump, in
the same shape as the records the published folds carry), ``checkpoint/`` and
``predictions.csv``.

Changes relative to the upstream ``main_moody.py``:

* ``device_num = max(1, torch.cuda.device_count())`` and ``accelerator = 'gpu' if
  torch.cuda.is_available() else 'cpu'``, both overridable by
  ``--devices``/``--accelerator``. Upstream divided the batch size by the CUDA
  device count and hardcoded ``accelerator='gpu'``, so nothing ran at all on a
  machine without CUDA.
* ``save_idc``, ``weight_average``, ``esm_test`` (which read a config key that
  does not exist and therefore raised ``AttributeError``), the AudioSet branch of
  ``test`` and their three subparsers are gone.
* prediction dumps are written exactly once, in truncating mode, through
  :class:`mirasol.training.predictions.PredictionWriter`.
* a checkpoint that does not load strictly is an error rather than a warning
  followed by a partial load; ``--allow-partial-load`` opts into the old
  behaviour deliberately.
* the parameter record is written for ``test`` as well as ``train``.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mirasol.paths import PHYSIONET_URL, resolve_paths
from mirasol.training.predictions import DEFAULT_PATIENT_RANGE, PredictionWriter
from mirasol.training.utils import build_config, dump_config

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from mirasol.config.resolve import ResolvedConfig

__all__ = [
    "ACCELERATORS",
    "DataModule",
    "RuntimeSettings",
    "build_parser",
    "load_fold_dataset",
    "main",
    "resolve_accelerator",
    "resolve_device_num",
    "run_test",
    "run_train",
]

#: The only dataset layout this release supports: the five-fold object array
#: written by ``mirasol.data.prepare``. The upstream AudioSet/SCV2 paths and the
#: HDF5 index they needed are gone.
SUPPORTED_DATASET_TYPE = "esc-50"

#: Accelerators accepted on the command line, matching what the configuration
#: layer validates.
ACCELERATORS: tuple[str, ...] = ("auto", "cpu", "gpu", "cuda", "mps")

#: Accelerators that can expose more than one device.
_MULTI_DEVICE_ACCELERATORS = frozenset({"gpu", "cuda"})


@dataclass(frozen=True)
class RuntimeSettings:
    """Machine-level settings that are deliberately not part of the config.

    These say *where* a run happens, not *what* it computes, so they stay out of
    the parameter record, whose job is to describe the experiment.
    """

    accelerator: str
    device_num: int
    save_predictions: bool = True
    predictions_path: Path | None = None
    overwrite: bool = False
    patient_range: tuple[int, int] | None = DEFAULT_PATIENT_RANGE
    allow_partial_load: bool = False


class DataModule(pl.LightningDataModule):
    """Wrap one fold's train/eval datasets in dataloaders.

    Args:
        train_dataset: the four training folds.
        eval_dataset: the held-out fold, used for validation and for the dump.
        config: resolved configuration (``batch_size``, ``num_workers``).
        device_num: number of devices; more than one adds a ``DistributedSampler``.
    """

    def __init__(
        self,
        train_dataset: Any,
        eval_dataset: Any,
        config: ResolvedConfig,
        device_num: int,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.device_num = device_num

    def _loader(self, dataset: Any) -> DataLoader:
        sampler = (
            DistributedSampler(dataset, shuffle=False) if self.device_num > 1 else None
        )
        return DataLoader(
            dataset=dataset,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size // self.device_num,
            shuffle=False,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        """Loader over the training folds; the dataset owns the sampling order."""
        return self._loader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        """Loader over the held-out fold, source of the monitored ``acc``."""
        return self._loader(self.eval_dataset)

    def test_dataloader(self) -> DataLoader:
        """Loader over the held-out fold, source of the prediction dump."""
        return self._loader(self.eval_dataset)


# --------------------------------------------------------------------------- #
# Runtime resolution
# --------------------------------------------------------------------------- #
def resolve_accelerator(requested: str | None = None) -> str:
    """Return the Lightning accelerator to use.

    ``None`` or ``'auto'`` selects ``'gpu'`` when CUDA is available and ``'cpu'``
    otherwise, which is what makes a CPU-only smoke test possible.
    """
    value = requested or "auto"
    if value not in ACCELERATORS:
        raise ValueError(f"accelerator {value!r} is not one of {', '.join(ACCELERATORS)}")
    if value == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return value


def resolve_device_num(requested: str | int | None, accelerator: str) -> int:
    """Return the number of devices, never zero.

    ``'auto'`` means every visible CUDA device (at least one) on a GPU
    accelerator, and one device otherwise. Upstream used
    ``torch.cuda.device_count()`` directly as the batch-size divisor, so a
    CPU-only machine raised ``ZeroDivisionError`` before training started.

    Raises:
        ValueError: if ``requested`` is neither ``'auto'`` nor a positive integer.
    """
    value: str | int = "auto" if requested is None else requested
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            if accelerator in _MULTI_DEVICE_ACCELERATORS:
                return max(1, torch.cuda.device_count())
            return 1
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                f"--devices must be 'auto' or a positive integer, got {requested!r}"
            ) from None
    if value < 1:
        raise ValueError(f"--devices must be >= 1, got {value}")
    return int(value)


def load_fold_dataset(config: ResolvedConfig) -> np.ndarray:
    """Load the prepared five-fold dataset this configuration names.

    The location comes from ``config.dataset_npy`` (``--dataset-path`` /
    ``$CIRCOR_ROOT`` / the repo-relative default), so no path is hardcoded.

    Returns:
        A length-5 object array; element ``i`` holds fold ``i``'s segment dicts
        (``name``, ``target``, ``waveform``).

    Raises:
        ValueError: if ``config.dataset_type`` is unsupported, the array does not
            have five folds, or ``config.esc_fold`` is out of range.
        FileNotFoundError: if the prepared dataset has not been built.
    """
    if config.dataset_type != SUPPORTED_DATASET_TYPE:
        raise ValueError(
            f"config.dataset_type is {config.dataset_type!r}; this release supports "
            f"only {SUPPORTED_DATASET_TYPE!r} (the prepared five-fold CirCor array)"
        )
    npy_path = Path(config.dataset_npy)
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"prepared dataset {npy_path} not found. Download CirCor DigiScope from "
            f"{PHYSIONET_URL} and build it with "
            f"'python -m mirasol.data.prepare --num-classes {config.classes_num}', "
            "or point --dataset-path at an existing build."
        )
    # allow_pickle is required: each element is a list of dicts. This file is
    # produced locally by mirasol.data.prepare and is never downloaded.
    full_dataset = np.load(npy_path, allow_pickle=True)
    if len(full_dataset) != 5:
        raise ValueError(
            f"{npy_path} has {len(full_dataset)} folds; the five-fold patient-level "
            "protocol of the paper needs exactly 5"
        )
    fold = int(config.esc_fold)
    if not 0 <= fold < 5:
        raise ValueError(f"config.esc_fold must be in 0..4, got {fold}")
    return full_dataset


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def run_train(config: ResolvedConfig, runtime: RuntimeSettings) -> None:
    """Train one architecture on four folds, validating on the held-out fold."""
    from mirasol.data.dataset import ESC_Dataset
    from mirasol.models.factory import create_model
    from mirasol.training.lightning_module import SEDWrapper

    _check_batch_size(config, runtime)
    full_dataset = load_fold_dataset(config)
    checkpoint_dir = config.run_dir / "checkpoint"
    if not config.debug:
        _write_run_record(config, checkpoint_dir, config.exp_name)

    train_dataset = ESC_Dataset(dataset=full_dataset, config=config, eval_mode=False)
    eval_dataset = ESC_Dataset(dataset=full_dataset, config=config, eval_mode=True)
    data_module = DataModule(train_dataset, eval_dataset, config, runtime.device_num)

    # 'acc' is segment-level accuracy on the held-out fold. The Challenge Score
    # the paper reports is a patient-level quantity computed post hoc from the
    # prediction dump, so it cannot be monitored here; Table S8 quantifies how
    # much the choice of selection metric changes the published numbers.
    checkpoint_callback_acc = ModelCheckpoint(
        monitor="acc",
        filename="l-{epoch:d}-{acc:.3f}",
        save_top_k=config.save_top_k,
        mode="max",
    )
    checkpoint_callback_loss = ModelCheckpoint(
        monitor="loss",
        filename="l-{epoch:d}-{acc:.3f}",
        save_top_k=config.save_top_k,
        mode="min",
    )
    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir=str(checkpoint_dir),
        devices=runtime.device_num,
        accelerator=runtime.accelerator,
        check_val_every_n_epoch=config.val_check_interval,
        max_epochs=config.max_epoch,
        sync_batchnorm=runtime.device_num > 1,
        callbacks=[checkpoint_callback_acc, checkpoint_callback_loss],
        strategy="ddp" if runtime.device_num > 1 else "auto",
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        precision=32,
    )
    model = SEDWrapper(
        sed_model=create_model(config), config=config, dataset=train_dataset
    )
    _load_initial_weights(model, config)
    trainer.fit(model, data_module)


def run_test(config: ResolvedConfig, runtime: RuntimeSettings) -> None:
    """Evaluate a checkpoint on the held-out fold and dump segment predictions."""
    from mirasol.data.dataset import ESC_Dataset
    from mirasol.models.factory import create_model
    from mirasol.training.lightning_module import SEDWrapper

    _check_batch_size(config, runtime)
    full_dataset = load_fold_dataset(config)
    if not config.debug:
        _write_run_record(config, config.run_dir, f"{config.exp_name}_test")

    eval_dataset = ESC_Dataset(dataset=full_dataset, config=config, eval_mode=True)
    data_module = DataModule(eval_dataset, eval_dataset, config, runtime.device_num)

    callbacks: list[pl.Callback] = []
    # save_preds defaults to True: producing the segment-level dump is the point
    # of the test stage, and --no-save-predictions is the explicit way out.
    if runtime.save_predictions and config.get("save_preds", True):
        predictions_path = runtime.predictions_path or (
            config.run_dir / "predictions.csv"
        )
        # Refuses a path under version control, so a run cannot overwrite the
        # committed results/predictions/ dumps the published tables were built on.
        config.paths.assert_writable(predictions_path)
        callbacks.append(
            PredictionWriter(
                output_path=predictions_path,
                expected_segments=len(eval_dataset),
                patient_range=runtime.patient_range,
                overwrite=runtime.overwrite,
            )
        )

    trainer = pl.Trainer(
        deterministic=False,
        devices=runtime.device_num,
        accelerator=runtime.accelerator,
        max_epochs=config.max_epoch,
        sync_batchnorm=runtime.device_num > 1,
        enable_checkpointing=False,
        callbacks=callbacks,
        strategy="ddp" if runtime.device_num > 1 else "auto",
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        precision=32,
    )
    model = SEDWrapper(sed_model=create_model(config), config=config, dataset=None)
    checkpoint = config.resume_checkpoint
    if checkpoint is None:
        print(
            "WARNING: resume_checkpoint is None, so this evaluates randomly "
            "initialised weights. That is a smoke test, not a result.",
            flush=True,
        )
    else:
        _load_for_evaluation(model, Path(checkpoint), runtime)
    trainer.test(model, datamodule=data_module)


# --------------------------------------------------------------------------- #
# Provenance and weight loading
# --------------------------------------------------------------------------- #
def _write_run_record(config: ResolvedConfig, directory: Path, stem: str) -> None:
    """Create ``directory`` and write both parameter records for this run.

    ``resolved_config.json`` carries the value *and the origin* of every
    parameter; ``<stem>.json`` is the flat dump in the same shape as the records
    the published folds carry, so old and new runs stay comparable.
    """
    config.paths.ensure(directory)
    config.write()
    dump_config(config, config.run_dir / stem, False)


def _load_for_evaluation(
    model: pl.LightningModule, checkpoint: Path, runtime: RuntimeSettings
) -> None:
    """Load a checkpoint for evaluation, strictly unless told otherwise."""
    state_dict = _load_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        if not runtime.allow_partial_load:
            raise RuntimeError(
                f"{checkpoint} does not match the configured model exactly: {exc}. "
                "Evaluating a partially loaded model reports numbers for weights "
                "that were never trained; check model_type and classes_num, or pass "
                "--allow-partial-load if a partial load is really intended."
            ) from exc
        print(
            f"WARNING: --allow-partial-load: loading {checkpoint} non-strictly ({exc})",
            flush=True,
        )
        model.load_state_dict(state_dict, strict=False)


def _load_initial_weights(model: pl.LightningModule, config: ResolvedConfig) -> None:
    """Optionally warm-start training from a checkpoint or a Swin pretrain file."""
    resume = config.resume_checkpoint
    if resume is not None:
        state_dict = _load_state_dict(Path(resume))
        # The classifier head and the HTS-AT token-semantic conv are dropped so a
        # checkpoint trained for a different class count can still warm-start.
        for key in (
            "sed_model.head.weight",
            "sed_model.head.bias",
            "sed_model.tscam_conv.weight",
            "sed_model.tscam_conv.bias",
        ):
            state_dict.pop(key, None)
        model.load_state_dict(state_dict, strict=False)
        print(f"warm-started from {resume}", flush=True)
        return
    if config.swin_pretrain_path is not None:
        _load_swin_pretrain(model, Path(config.swin_pretrain_path))


def _load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Read the ``state_dict`` out of a Lightning checkpoint."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint {checkpoint} not found")
    # weights_only=False is explicit: a Lightning checkpoint carries pickled
    # hyperparameters, which the torch >= 2.6 default would refuse to unpickle.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise KeyError(f"{checkpoint} has no 'state_dict' key; keys: {sorted(payload)}")
    return dict(payload["state_dict"])


def _load_swin_pretrain(model: pl.LightningModule, checkpoint: Path) -> None:
    """Load Microsoft Swin-Transformer ImageNet weights into the HTS-AT arm."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"swin_pretrain_path {checkpoint} not found")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt = payload["model"]
    found: list[str] = []
    unfound: list[str] = []
    model_params = dict(model.state_dict())
    for key in model_params:
        source_key = key.replace("sed_model.", "")
        if source_key not in ckpt:
            unfound.append(key)
            continue
        if source_key == "patch_embed.proj.weight":
            # RGB -> single mel channel.
            ckpt[source_key] = torch.mean(ckpt[source_key], dim=1, keepdim=True)
        if source_key in ("head.weight", "head.bias"):
            ckpt.pop(source_key)
            unfound.append(key)
            continue
        if model_params[key].shape != ckpt[source_key].shape:
            raise ValueError(
                f"{key} shape mismatch: model {tuple(model_params[key].shape)} vs "
                f"pretrain {tuple(ckpt[source_key].shape)}"
            )
        found.append(key)
        ckpt[key] = ckpt.pop(source_key)
    print(
        f"swin pretrain: matched {len(found)} tensors, {len(unfound)} left at init",
        flush=True,
    )
    model.load_state_dict(ckpt, strict=False)


def _check_batch_size(config: ResolvedConfig, runtime: RuntimeSettings) -> None:
    """Refuse a device count that would give an empty per-device batch."""
    if config.batch_size < runtime.device_num:
        raise ValueError(
            f"batch_size {config.batch_size} is smaller than the device count "
            f"{runtime.device_num}, which would give an empty per-device batch"
        )


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the ``train`` / ``test`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mirasol-train",
        description=(
            "Train or evaluate one architecture on one fold of the CirCor "
            "patient-level five-fold split."
        ),
    )
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="flat YAML file of parameter overrides",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override one parameter; repeatable. Unknown keys are rejected.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        choices=ACCELERATORS,
        help="default: gpu when CUDA is available, otherwise cpu",
    )
    parser.add_argument(
        "--devices",
        default=None,
        metavar="N|auto",
        help="number of devices; default max(1, torch.cuda.device_count())",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="root for runs/<exp_name>/ (default $MIRASOL_RUNS_DIR or <repo>/runs)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="directory holding the prepared .npy datasets (default $CIRCOR_ROOT)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        dest="predictions_path",
        metavar="PATH",
        help="prediction CSV path (default runs/<exp_name>/predictions.csv)",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_false",
        dest="save_predictions",
        help="evaluate without writing a prediction dump",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing prediction dump instead of refusing to start",
    )
    parser.add_argument(
        "--expect-patients",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=list(DEFAULT_PATIENT_RANGE),
        help="inclusive bounds on the patient count of the dump",
    )
    parser.add_argument(
        "--no-expect-patients",
        action="store_true",
        help="skip the patient-count check (custom or reduced extracts)",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="permit a non-strict checkpoint load when evaluating",
    )
    return parser


def _runtime_from_args(
    args: argparse.Namespace, config: ResolvedConfig
) -> RuntimeSettings:
    """Combine the machine flags with their configuration defaults."""
    accelerator = resolve_accelerator(args.accelerator or config.accelerator)
    devices = args.devices if args.devices is not None else config.devices
    patient_range: tuple[int, int] | None = None
    if not args.no_expect_patients:
        low, high = args.expect_patients
        patient_range = (int(low), int(high))
    return RuntimeSettings(
        accelerator=accelerator,
        device_num=resolve_device_num(devices, accelerator),
        save_predictions=args.save_predictions,
        predictions_path=args.predictions_path,
        overwrite=args.overwrite,
        patient_range=patient_range,
        allow_partial_load=args.allow_partial_load,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning, module="torchlibrosa")

    config = build_config(
        config_path=args.config,
        overrides=args.overrides,
        paths=resolve_paths(circor_root=args.dataset_path, runs_dir=args.runs_dir),
    )
    runtime = _runtime_from_args(args, config)
    pl.seed_everything(seed=config.random_seed, workers=True)
    torch.set_float32_matmul_precision("high")
    print(
        f"{args.mode}: model_type={config.model_type} exp_name={config.exp_name} "
        f"fold={config.esc_fold} classes_num={config.classes_num} "
        f"accelerator={runtime.accelerator} devices={runtime.device_num}",
        flush=True,
    )
    if args.mode == "train":
        run_train(config, runtime)
    else:
        run_test(config, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
