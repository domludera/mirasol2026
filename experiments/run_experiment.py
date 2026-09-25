#!/usr/bin/env python3
"""Unified experiment driver for the MIRASOL heart-murmur screening study.

This one script replaces the six near-duplicate ``run_*.py`` drivers and the two
``automate_*`` export wrappers used to produce the paper.  It serves these paper
artifacts:

* ``cv``        -- Table 1 (six architectures, five folds, three-class task) and
                   Table S7 (the two-class murmur configuration).
* ``ablation``  -- the preprocessing block of Table S10.
* ``sweep``     -- the class-weight sensitivity block of Table S10.
* ``export``    -- the ONNX/quantized graphs behind Table 4 and Sec. 3.5.

Design rules, each of which fixes a defect in the scripts this file replaces:

1. **No tracked file is ever modified.**  The retired drivers reconfigured runs by
   ``re.sub``-ing ``config/esc_config.py`` in place, so results depended on
   invocation order, a crash left an arbitrary state, two folds could not run
   concurrently, and a missed substitution failed silently.  This driver only
   builds ``argv`` and hands ``--set key=value`` pairs to ``mirasol.cli.train``,
   which resolves them through ``mirasol.config.resolve`` into an immutable
   config and records ``runs/<exp>/resolved_config.json`` as the provenance.
2. **A task is a complete parameter set.**  ``run_5fold.py`` patched
   ``model_type``/``exp_name``/``esc_fold``/``dataset_npy_name`` but never
   ``classes_num`` or ``loss_weight``, so the two-class experiment trained a
   three-output head with a three-element loss weight on two-class data and
   Table S7 was not reproducible.  Every parameter that distinguishes one
   experiment from another is written here (see ``TASKS``), never inherited.
   The same applies to ``ablation``, whose retired driver never set
   ``esc_fold``, ``classes_num`` or ``loss_weight`` at all.
3. **One checkpoint selector.**  ``select_checkpoint`` replaces five copies of an
   accuracy-parsing selector and three ``getmtime`` selectors, and orders
   ``version_*`` directories numerically rather than lexicographically.
4. **Failures are loud.**  Per-unit status is collected and the process exits
   non-zero if any requested unit produced no predictions.
5. **Portable.**  ``sys.executable`` instead of a hardcoded venv interpreter, all
   paths derived from ``mirasol.paths``, ASCII-only output.

Usage (see experiments/README.md for the full migration table)::

    python experiments/run_experiment.py cv --task 3class --models all
    python experiments/run_experiment.py cv --task 2class --models tinyvit
    python experiments/run_experiment.py ablation --model tinyvit --fold 0
    python experiments/run_experiment.py sweep --preset class-weight --model tinyvit
    python experiments/run_experiment.py export --models all --formats fp32,int8-dynamic
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

TRAIN_MODULE = "mirasol.cli.train"
PREPARE_MODULE = "mirasol.data.prepare"
EXPORT_MODULE = "mirasol.deployment.export"
VALIDATE_MODULE = "mirasol.deployment.validate"

OK_STATUSES = frozenset({"ok", "skipped", "dry-run"})


class DriverError(RuntimeError):
    """A configuration or pre-flight error that must stop the driver."""


# ---------------------------------------------------------------------------
# Task presets: COMPLETE parameter sets, never partial patches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """A complete, self-contained parameter set for one labelling scheme.

    ``exp_tag`` reproduces the experiment names in the shipped prediction dumps
    (``<model>_3class_fold<k>`` and ``<model>_binary_fold<k>``), so the CSVs in
    ``results/predictions/`` can be traced back to the command that made them.
    """

    exp_tag: str
    classes_num: int
    loss_weight: tuple[float, ...]
    dataset_npy_name: str
    clip_seconds: int
    prepare_num_classes: int
    description: str

    def params(self) -> dict[str, Any]:
        """Return the config overrides this task requires, as a fresh dict.

        ``loss_weight`` stays a tuple to match the type in
        ``mirasol.config.defaults``, so the resolver needs no coercion.
        """
        return {
            "classes_num": self.classes_num,
            "loss_weight": self.loss_weight,
            "loss_type": "weighted_clip_ce",
            "dataset_npy_name": self.dataset_npy_name,
            "clip_seconds": self.clip_seconds,
        }


TASKS: Mapping[str, Task] = {
    "3class": Task(
        exp_tag="3class",
        classes_num=3,
        loss_weight=(4.0, 10.0, 1.0),
        dataset_npy_name="train-circor-data-30s.npy",
        clip_seconds=30,
        prepare_num_classes=3,
        description="Present=0 / Unknown=1 / Absent=2 (Table 1, Table 2, S1-S6)",
    ),
    "2class": Task(
        exp_tag="binary",
        classes_num=2,
        loss_weight=(4.0, 1.0),
        dataset_npy_name="train-circor-data-30s-binary.npy",
        clip_seconds=30,
        prepare_num_classes=2,
        description="Present|Unknown=0 / Absent=1 (Table S7)",
    ),
}


# ---------------------------------------------------------------------------
# Ablation conditions (Table S10, preprocessing block)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ablation:
    """One preprocessing ablation condition and the dataset it needs."""

    name: str
    group: str
    description: str
    dataset_npy: str
    clip_seconds: int
    prepare_flags: tuple[str, ...]


ABLATIONS: tuple[Ablation, ...] = (
    Ablation("single_location", "Location restructuring", "single location per patient",
             "ablation-single-loc-30s.npy", 30, ("--single_location",)),
    Ablation("window_10s", "Window length", "10-second clips",
             "ablation-10s.npy", 10, ()),
    Ablation("window_15s", "Window length", "15-second clips",
             "ablation-15s.npy", 15, ()),
    Ablation("no_bandpass", "Bandpass filter", "no bandpass filter",
             "ablation-nofilter-30s.npy", 30, ("--no_bandpass",)),
    Ablation("wide_bandpass", "Bandpass filter", "25-1000 Hz bandpass",
             "ablation-wide-30s.npy", 30, ("--bandpass_high", "1000")),
)

# Class-weight sensitivity (Table S10, lower block).  The paper's baseline row is
# the main three-class fold-0 run; the preset retrains it under its own name so
# all three rows come from identically-driven runs.
SWEEP_PRESETS: Mapping[str, tuple[str, tuple[tuple[str, Any], ...]]] = {
    "class-weight": (
        "loss_weight",
        (
            ("cw_2_5_1", (2.0, 5.0, 1.0)),
            ("cw_4_10_1", (4.0, 10.0, 1.0)),
            ("cw_6_15_1", (6.0, 15.0, 1.0)),
        ),
    ),
}

# Export formats -> (extra export flags, emitted ONNX filename).
EXPORT_FORMATS: Mapping[str, tuple[tuple[str, ...], str]] = {
    "fp32": ((), "model_fp32.onnx"),
    "int8-dynamic": ((), "model_int8.onnx"),
    "int8-static": (("--static",), "model_int8_static.onnx"),
    "fp16": (("--fp16",), "model_fp16.onnx"),
}


# ---------------------------------------------------------------------------
# Package interfaces (imported lazily so --help works without torch installed)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def default_config_keys() -> frozenset[str]:
    """Return every settable key in ``mirasol.config.defaults``.

    Used for pre-flight validation so a typo in ``--set`` fails before a
    multi-hour sweep starts rather than in the middle of it.  ``defaults.DEFAULTS``
    is the authoritative mapping; the introspective branch is a fallback.
    """
    from mirasol.config import defaults

    mapping = getattr(defaults, "DEFAULTS", None)
    if mapping is not None:
        return frozenset(mapping)
    return frozenset(
        name
        for name, value in vars(defaults).items()
        if not name.startswith("_")
        and not inspect.ismodule(value)
        and not callable(value)
    )


@lru_cache(maxsize=1)
def default_sample_rate() -> int:
    """Return the sample rate from the defaults module."""
    from mirasol.config import defaults

    return int(defaults.sample_rate)


@lru_cache(maxsize=1)
def registered_models() -> tuple[str, ...]:
    """Return the architecture names registered in ``mirasol.models.factory``.

    Read from the factory in its declared order (which is Table 1 column order),
    so the driver can never disagree with the model list the training code
    actually supports.  The retired drivers hardcoded four of the six
    architectures in five different places.
    """
    from mirasol.models import factory

    for attribute in ("MODEL_TYPES", "available_models", "_BUILDERS"):
        registry = getattr(factory, attribute, None)
        if registry is None:
            continue
        return tuple(registry() if callable(registry) else registry)
    raise DriverError(
        "mirasol.models.factory exposes no model registry; expected MODEL_TYPES, "
        "available_models() or _BUILDERS."
    )


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root as resolved by ``mirasol.paths``."""
    from mirasol.paths import resolve_paths

    return resolve_paths().repo_root


def runs_dir(override: str | None) -> Path:
    """Return the directory holding per-experiment run outputs.

    Delegates to ``mirasol.paths.resolve_paths``, so the resolution order is
    ``--runs-dir`` > ``$MIRASOL_RUNS_DIR`` > ``<repo root>/runs``.
    """
    from mirasol.paths import resolve_paths

    return resolve_paths(runs_dir=override).runs_dir


def exports_dir(override: str | None) -> Path:
    """Return the directory ONNX exports are written to.

    ``--output-dir`` > ``$MIRASOL_EXPORTS_DIR`` > ``<repo root>/exports``.
    """
    from mirasol.paths import resolve_paths

    return resolve_paths(exports_dir=override).exports_dir


# ---------------------------------------------------------------------------
# Override formatting and validation
# ---------------------------------------------------------------------------


def format_value(value: Any) -> str:
    """Render a config value so ``ast.literal_eval`` recovers it exactly.

    Strings are emitted bare (``exp_name=tinyvit_3class_fold0``) because
    ``literal_eval`` fails on them and the resolver falls back to ``str``.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        # A list literal is emitted for both, because it is what a user would
        # type and it parses under both ast.literal_eval and json.loads. Keys
        # whose default is a tuple are coerced back by the resolver.
        return "[" + ",".join(format_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return repr(value)
    raise DriverError(f"cannot serialise config value {value!r} of type {type(value)}")


def set_args(params: Mapping[str, Any]) -> list[str]:
    """Turn a parameter mapping into repeated ``--set key=value`` arguments."""
    args: list[str] = []
    for key in sorted(params):
        args += ["--set", f"{key}={format_value(params[key])}"]
    return args


def parse_set_overrides(raw: Sequence[str]) -> dict[str, str]:
    """Parse user ``--set key=value`` strings into a mapping of raw strings."""
    parsed: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise DriverError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise DriverError(f"--set expects a non-empty key, got {item!r}")
        parsed[key] = value
    return parsed


def validate_keys(keys: Iterable[str]) -> None:
    """Fail fast if any override key does not exist in the defaults module."""
    known = default_config_keys()
    unknown = sorted(set(keys) - known)
    if unknown:
        raise DriverError(
            "unknown config key(s): "
            + ", ".join(unknown)
            + ". Every override must already exist in mirasol.config.defaults."
        )


def with_derived_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Add parameters that are functions of others, so none can go stale.

    ``clip_samples`` is ``sample_rate * clip_seconds`` in the defaults module.
    The retired ablation driver rewrote both by regex; here the derived value is
    recomputed from whatever ``clip_seconds``/``sample_rate`` this run uses.
    """
    resolved = dict(params)
    if "clip_seconds" in resolved and "clip_samples" not in resolved:
        sample_rate = int(resolved.get("sample_rate", default_sample_rate()))
        resolved["clip_samples"] = sample_rate * int(resolved["clip_seconds"])
    return resolved


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

_ACC_RE = re.compile(r"acc=(\d+(?:\.\d+)?)")
_VERSION_RE = re.compile(r"version_(\d+)")


@dataclass(frozen=True)
class Checkpoint:
    """A Lightning checkpoint with the metadata used to rank it."""

    path: Path
    accuracy: float | None
    version: int
    mtime: float


def find_checkpoints(exp_dir: Path) -> tuple[Checkpoint, ...]:
    """Return every ``.ckpt`` under an experiment directory, newest last."""
    if not exp_dir.is_dir():
        return ()
    found: list[Checkpoint] = []
    for path in sorted(exp_dir.rglob("*.ckpt")):
        acc_match = _ACC_RE.search(path.name)
        version = -1
        for part in path.parts:
            version_match = _VERSION_RE.fullmatch(part)
            if version_match:
                version = max(version, int(version_match.group(1)))
        found.append(
            Checkpoint(
                path=path,
                accuracy=float(acc_match.group(1)) if acc_match else None,
                version=version,
                mtime=path.stat().st_mtime,
            )
        )
    return tuple(found)


def select_checkpoint(exp_dir: Path, strategy: str = "acc") -> Checkpoint:
    """Pick one checkpoint for testing or export.

    ``acc`` ranks by the accuracy encoded in the filename, breaking ties by
    numeric ``version_*`` index then modification time; ``latest`` ranks by
    version then modification time.  ``version_*`` directories are ordered
    numerically, so ``version_10`` no longer sorts before ``version_2``.

    Raises:
        DriverError: if no checkpoint exists, or if ``acc`` is requested and no
            filename carries a parsable accuracy.  The retired selectors scored
            such files 0.0 and silently returned an arbitrary one.
    """
    candidates = find_checkpoints(exp_dir)
    if not candidates:
        raise DriverError(f"no .ckpt found under {exp_dir}")
    if strategy == "acc":
        scored = [c for c in candidates if c.accuracy is not None]
        if not scored:
            raise DriverError(
                f"no checkpoint under {exp_dir} has an 'acc=' value in its "
                "filename; re-run with --checkpoint-strategy latest"
            )
        return max(scored, key=lambda c: (c.accuracy, c.version, c.mtime))
    if strategy == "latest":
        return max(candidates, key=lambda c: (c.version, c.mtime))
    raise DriverError(f"unknown checkpoint strategy {strategy!r}")


def predictions_path(base: Path, exp_name: str) -> Path:
    """Return the per-run prediction dump written by the test callback."""
    return base / exp_name / "predictions.csv"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunUnit:
    """One train-then-test unit: an experiment name plus its complete config."""

    label: str
    exp_name: str
    params: Mapping[str, Any]
    prepare_cmd: tuple[str, ...] = ()
    note: str = ""


@dataclass
class RunResult:
    """The outcome of a single unit, collected for the summary and the report."""

    label: str
    exp_name: str
    status: str
    params: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    predictions: str | None = None
    seconds: float = 0.0


def run_cmd(cmd: Sequence[str], *, dry_run: bool, cwd: Path) -> int:
    """Print and run a command, returning its exit status (0 when dry-running).

    The printed form is shell-quoted so it can be copied out of a log and re-run
    verbatim; the subprocess itself receives the unquoted argv.
    """
    print("  >>> " + shlex.join(str(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(list(cmd), cwd=str(cwd), check=False).returncode


def trainer_flags(args: argparse.Namespace) -> list[str]:
    """Return the non-``--set`` flags to forward to ``mirasol.cli.train``."""
    flags: list[str] = []
    if args.config:
        flags += ["--config", args.config]
    if args.accelerator:
        flags += ["--accelerator", args.accelerator]
    if args.devices is not None:
        flags += ["--devices", str(args.devices)]
    if args.runs_dir:
        flags += ["--runs-dir", args.runs_dir]
    if args.overwrite:
        # mirasol.cli.train refuses to replace an existing prediction dump
        # unless it is told to; the decision is made once, here.
        flags.append("--overwrite")
    if getattr(args, "dataset_dir", None):
        flags += ["--dataset-path", args.dataset_dir]
    return flags


def preflight(units: Sequence[RunUnit], base: Path, args: argparse.Namespace) -> None:
    """Validate every unit before any of them starts.

    Refuses to run if a prediction dump already exists, unless ``--overwrite``
    (redo it) or ``--skip-existing`` (leave it alone) says what to do.  The
    retired drivers either appended to the existing CSV, doubling every row and
    corrupting Tables 1, 2 and S1-S8, or silently skipped the fold.
    """
    seen: dict[str, str] = {}
    for unit in units:
        validate_keys(unit.params)
        if unit.exp_name in seen:
            raise DriverError(
                f"experiment name {unit.exp_name!r} is used by both "
                f"{seen[unit.exp_name]!r} and {unit.label!r}"
            )
        seen[unit.exp_name] = unit.label
    if args.overwrite or args.skip_existing:
        return
    existing = [
        str(predictions_path(base, unit.exp_name))
        for unit in units
        if predictions_path(base, unit.exp_name).exists()
    ]
    if existing:
        raise DriverError(
            "predictions already exist for "
            + str(len(existing))
            + " requested run(s), first: "
            + existing[0]
            + ". Pass --overwrite to regenerate them or --skip-existing to keep them."
        )


def execute_unit(unit: RunUnit, base: Path, args: argparse.Namespace) -> RunResult:
    """Run dataset preparation (if any), training and testing for one unit."""
    target = predictions_path(base, unit.exp_name)
    if args.skip_existing and not args.overwrite and target.exists():
        print(f"  SKIP: {unit.exp_name} already has {target}")
        return RunResult(unit.label, unit.exp_name, "skipped", unit.params,
                         predictions=str(target))

    started = time.time()
    root = repo_root()
    print(f"\n{'=' * 72}\n{unit.label}  ->  exp_name={unit.exp_name}\n{'=' * 72}")
    if unit.note:
        print(f"  note: {unit.note}")
    for key in sorted(unit.params):
        print(f"    {key} = {unit.params[key]!r}")

    if unit.prepare_cmd and run_cmd(unit.prepare_cmd, dry_run=args.dry_run, cwd=root) != 0:
        return RunResult(unit.label, unit.exp_name, "prepare-failed", unit.params)

    train_params = dict(unit.params)
    train_params["exp_name"] = unit.exp_name
    train_params["resume_checkpoint"] = None
    train_cmd = [sys.executable, "-m", TRAIN_MODULE, "train"]
    train_cmd += set_args(with_derived_params(train_params)) + trainer_flags(args)
    if run_cmd(train_cmd, dry_run=args.dry_run, cwd=root) != 0:
        return RunResult(unit.label, unit.exp_name, "train-failed", unit.params)

    if args.dry_run:
        test_params = dict(train_params)
        test_params["resume_checkpoint"] = "<selected-after-training>"
        test_cmd = [sys.executable, "-m", TRAIN_MODULE, "test"]
        test_cmd += set_args(with_derived_params(test_params)) + trainer_flags(args)
        run_cmd(test_cmd, dry_run=True, cwd=root)
        return RunResult(unit.label, unit.exp_name, "dry-run", unit.params)

    try:
        checkpoint = select_checkpoint(base / unit.exp_name, args.checkpoint_strategy)
    except DriverError as error:
        print(f"  ERROR: {error}")
        return RunResult(unit.label, unit.exp_name, "no-checkpoint", unit.params)
    print(f"  selected checkpoint: {checkpoint.path} (acc={checkpoint.accuracy})")

    test_params = dict(train_params)
    test_params["resume_checkpoint"] = str(checkpoint.path)
    test_cmd = [sys.executable, "-m", TRAIN_MODULE, "test"]
    test_cmd += set_args(with_derived_params(test_params)) + trainer_flags(args)
    status = "ok"
    if run_cmd(test_cmd, dry_run=False, cwd=root) != 0:
        status = "test-failed"
    elif not target.exists() or target.stat().st_size == 0:
        status = "no-predictions"

    elapsed = time.time() - started
    print(f"  {unit.exp_name}: {status} in {elapsed / 60:.1f} min")
    return RunResult(
        label=unit.label,
        exp_name=unit.exp_name,
        status=status,
        params=unit.params,
        checkpoint=str(checkpoint.path),
        predictions=str(target) if target.exists() else None,
        seconds=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Subcommand unit builders
# ---------------------------------------------------------------------------


def parse_list(raw: Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    """Flatten repeated and comma-separated option values."""
    if not raw:
        return tuple(default)
    items: list[str] = []
    for chunk in raw:
        items += [part.strip() for part in chunk.split(",") if part.strip()]
    return tuple(dict.fromkeys(items))


def resolve_models(
    raw: Sequence[str] | None, default: Sequence[str] = ("all",)
) -> tuple[str, ...]:
    """Expand ``--models`` into architecture names, validating against the factory."""
    available = registered_models()
    requested = parse_list(raw, default)
    if "all" in requested:
        return available
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise DriverError(
            "unknown model(s): "
            + ", ".join(unknown)
            + ". Registered architectures: "
            + ", ".join(available)
        )
    return requested


def resolve_folds(raw: Sequence[str] | None) -> tuple[int, ...]:
    """Expand ``--folds`` into fold indices."""
    folds: list[int] = []
    for item in parse_list(raw, ("0", "1", "2", "3", "4")):
        try:
            folds.append(int(item))
        except ValueError as error:
            raise DriverError(f"--folds expects integers, got {item!r}") from error
    return tuple(folds)


def build_cv_units(args: argparse.Namespace, extra: Mapping[str, Any]) -> list[RunUnit]:
    """Build one unit per (architecture, fold) for a cross-validation sweep."""
    task = TASKS[args.task]
    units: list[RunUnit] = []
    for model in resolve_models(args.models):
        for fold in resolve_folds(args.folds):
            params = task.params()
            params["model_type"] = model
            params["esc_fold"] = fold
            params.update(extra)
            exp_name = args.exp_template.format(
                model=model, task=args.task, tag=task.exp_tag, fold=fold, label=""
            )
            units.append(
                RunUnit(
                    label=f"cv {args.task} {model} fold{fold}",
                    exp_name=exp_name,
                    params=params,
                )
            )
    return units


def build_ablation_units(
    args: argparse.Namespace, extra: Mapping[str, Any]
) -> list[RunUnit]:
    """Build one unit per preprocessing ablation condition (Table S10)."""
    task = TASKS[args.task]
    selected = parse_list(args.conditions, tuple(a.name for a in ABLATIONS))
    known = {ablation.name: ablation for ablation in ABLATIONS}
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise DriverError(
            "unknown ablation condition(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(known)
        )
    units: list[RunUnit] = []
    for name in selected:
        ablation = known[name]
        params = task.params()
        params["model_type"] = args.model
        params["esc_fold"] = args.fold
        params["dataset_npy_name"] = ablation.dataset_npy
        params["clip_seconds"] = ablation.clip_seconds
        params.update(extra)
        prepare_cmd: tuple[str, ...] = ()
        if not args.skip_prepare:
            prepare = [
                sys.executable,
                "-m",
                PREPARE_MODULE,
                "--num-classes",
                str(task.prepare_num_classes),
                "--clip_seconds",
                str(ablation.clip_seconds),
                "--output_name",
                ablation.dataset_npy,
            ]
            if args.dataset_dir:
                prepare += ["--output_dir", args.dataset_dir]
            prepare += list(ablation.prepare_flags)
            prepare_cmd = tuple(prepare)
        note = ablation.group + ": " + ablation.description
        if name == "single_location":
            note += (
                " (this condition drops recordings, so the surviving patient set "
                "and therefore the fold membership differ from the baseline)"
            )
        units.append(
            RunUnit(
                label=f"ablation {name}",
                exp_name=args.exp_template.format(
                    model=args.model,
                    task=args.task,
                    tag=task.exp_tag,
                    fold=args.fold,
                    label=name,
                ),
                params=params,
                prepare_cmd=prepare_cmd,
                note=note,
            )
        )
    return units


def build_sweep_units(
    args: argparse.Namespace, extra: Mapping[str, Any]
) -> list[RunUnit]:
    """Build one unit per value of a swept config key (Table S10 lower block)."""
    task = TASKS[args.task]
    if args.preset:
        key, values = SWEEP_PRESETS[args.preset]
    else:
        if not args.key or not args.value:
            raise DriverError(
                "sweep needs either --preset or both --key and at least one "
                "--value label=literal"
            )
        key = args.key
        values = tuple(_parse_sweep_value(item) for item in args.value)
    validate_keys([key])
    units: list[RunUnit] = []
    for label, value in values:
        params = task.params()
        params["model_type"] = args.model
        params["esc_fold"] = args.fold
        params[key] = value
        params.update(extra)
        units.append(
            RunUnit(
                label=f"sweep {key}={value}",
                exp_name=args.exp_template.format(
                    model=args.model,
                    task=args.task,
                    tag=task.exp_tag,
                    fold=args.fold,
                    label=label,
                ),
                params=params,
                note=f"{key} = {value}",
            )
        )
    return units


def _parse_sweep_value(item: str) -> tuple[str, Any]:
    """Parse a ``label=literal`` sweep value into (label, python value)."""
    import ast

    if "=" not in item:
        raise DriverError(f"--value expects label=literal, got {item!r}")
    label, raw = item.split("=", 1)
    label = label.strip()
    if not label:
        raise DriverError(f"--value expects a non-empty label, got {item!r}")
    try:
        value: Any = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return label, value


# ---------------------------------------------------------------------------
# Export subcommand
# ---------------------------------------------------------------------------


def run_export(args: argparse.Namespace) -> list[RunResult]:
    """Export (and optionally validate) ONNX graphs for the requested models.

    Replaces automate_export_onnx.py and automate_static_fp16.py, both of which
    rewrote the tracked config to pick a model and then selected a checkpoint by
    modification time from a hardcoded ``version_0`` directory.
    """
    task = TASKS[args.task]
    formats = parse_list(args.formats, ("fp32", "int8-dynamic"))
    unknown = [name for name in formats if name not in EXPORT_FORMATS]
    if unknown:
        raise DriverError(
            "unknown format(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(EXPORT_FORMATS)
        )
    base = runs_dir(args.runs_dir)
    out_base = exports_dir(args.output_dir)
    root = repo_root()
    flags: list[str] = []
    for name in formats:
        flags += [flag for flag in EXPORT_FORMATS[name][0] if flag not in flags]
    if args.per_tensor:
        flags.append("--per-tensor")

    results: list[RunResult] = []
    for model in resolve_models(args.models):
        exp_name = args.exp_template.format(
            model=model, task=args.task, tag=task.exp_tag, fold=args.fold, label=""
        )
        print(f"\n{'=' * 72}\nexport {model} from {exp_name}\n{'=' * 72}")
        try:
            checkpoint = select_checkpoint(base / exp_name, args.checkpoint_strategy)
        except DriverError as error:
            print(f"  ERROR: {error}")
            results.append(RunResult(f"export {model}", exp_name, "no-checkpoint"))
            continue
        print(f"  selected checkpoint: {checkpoint.path} (acc={checkpoint.accuracy})")
        model_out = out_base / model
        cmd = [
            sys.executable,
            "-m",
            EXPORT_MODULE,
            "--model_type",
            model,
            "--checkpoint",
            str(checkpoint.path),
            "--output_dir",
            str(out_base),
            "--num_calib",
            str(args.num_calib),
            *flags,
        ]
        if run_cmd(cmd, dry_run=args.dry_run, cwd=root) != 0:
            results.append(
                RunResult(f"export {model}", exp_name, "export-failed",
                          checkpoint=str(checkpoint.path))
            )
            continue

        status = "dry-run" if args.dry_run else "ok"
        for name in formats:
            onnx_path = model_out / EXPORT_FORMATS[name][1]
            if not args.dry_run and not onnx_path.exists():
                print(f"  ERROR: expected {onnx_path} was not produced")
                status = "missing-artifact"
                continue
            if not args.validate:
                continue
            validate_cmd = [
                sys.executable,
                "-m",
                VALIDATE_MODULE,
                "--model_type",
                model,
                "--checkpoint",
                str(checkpoint.path),
                "--onnx_model",
                str(onnx_path),
                "--patient-level",
            ]
            if run_cmd(validate_cmd, dry_run=args.dry_run, cwd=root) != 0:
                status = "validate-failed"
        results.append(
            RunResult(
                label=f"export {model}",
                exp_name=exp_name,
                status=status,
                params={"formats": list(formats), "per_tensor": bool(args.per_tensor)},
                checkpoint=str(checkpoint.path),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(subcommand: str, results: Sequence[RunResult]) -> int:
    """Print an ASCII status table and return the process exit code."""
    print(f"\n{'=' * 72}\nSUMMARY: {subcommand}\n{'=' * 72}")
    width = max([len(r.label) for r in results] + [10])
    print(f"{'unit'.ljust(width)}  {'status':<16} experiment")
    print("-" * (width + 40))
    for result in results:
        print(f"{result.label.ljust(width)}  {result.status:<16} {result.exp_name}")
    failed = [r for r in results if r.status not in OK_STATUSES]
    if failed:
        print(f"\n{len(failed)} of {len(results)} unit(s) did not complete:")
        for result in failed:
            print(f"  {result.label}: {result.status}")
        return 1
    print(f"\nAll {len(results)} unit(s) completed.")
    return 0


def write_report(path: Path, subcommand: str, results: Sequence[RunResult]) -> None:
    """Write a machine-readable status report for the driver invocation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subcommand": subcommand,
        "argv": sys.argv[1:],
        "python": sys.version.split()[0],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "units": [
            {
                "label": r.label,
                "exp_name": r.exp_name,
                "status": r.status,
                "params": {k: r.params[k] for k in sorted(r.params)},
                "checkpoint": r.checkpoint,
                "predictions": r.predictions,
                "seconds": r.seconds,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nStatus report written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser) -> None:
    """Add the options every subcommand shares."""
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="3class",
        help="labelling scheme; each is a complete parameter set (see TASKS)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra config override, repeatable; validated against the defaults",
    )
    parser.add_argument("--config", help="YAML file of config overrides")
    parser.add_argument("--runs-dir", help="root for runs/<exp>/ (default: <repo>/runs)")
    parser.add_argument(
        "--checkpoint-strategy",
        choices=("acc", "latest"),
        default="acc",
        help="how to pick the checkpoint used for testing/export",
    )
    parser.add_argument("--accelerator", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--devices", type=int)
    parser.add_argument(
        "--overwrite", action="store_true", help="regenerate existing predictions"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave runs whose predictions already exist untouched",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print every command without running it"
    )
    parser.add_argument("--report", help="write a JSON status report to this path")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for all four subcommands."""
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the architectures registered in mirasol.models.factory and exit",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    cv = subparsers.add_parser("cv", help="five-fold cross-validation (Table 1, S7)")
    add_common(cv)
    cv.add_argument("--models", action="append", help="'all' or comma-separated names")
    cv.add_argument("--folds", action="append", help="comma-separated fold indices")
    cv.add_argument("--exp-template", default="{model}_{tag}_fold{fold}")

    ablation = subparsers.add_parser(
        "ablation", help="preprocessing ablation (Table S10, upper block)"
    )
    add_common(ablation)
    ablation.add_argument("--model", default="tinyvit")
    ablation.add_argument("--fold", type=int, default=0)
    ablation.add_argument(
        "--conditions", action="append", help="comma-separated condition names"
    )
    ablation.add_argument("--exp-template", default="ablation_{label}")
    ablation.add_argument(
        "--dataset-dir",
        help="directory for the ablation .npy files; forwarded to both "
        "mirasol.data.prepare (--output_dir) and mirasol.cli.train (--dataset-path)",
    )
    ablation.add_argument(
        "--skip-prepare",
        action="store_true",
        help="assume the ablation .npy files already exist",
    )

    sweep = subparsers.add_parser(
        "sweep", help="single-key sensitivity sweep (Table S10, lower block)"
    )
    add_common(sweep)
    sweep.add_argument("--model", default="tinyvit")
    sweep.add_argument("--fold", type=int, default=0)
    sweep.add_argument("--preset", choices=sorted(SWEEP_PRESETS))
    sweep.add_argument("--key", help="config key to sweep")
    sweep.add_argument(
        "--value", action="append", metavar="LABEL=LITERAL", help="one swept value"
    )
    sweep.add_argument("--exp-template", default="{model}_{label}_fold{fold}")

    export = subparsers.add_parser("export", help="ONNX export (Table 4, Sec. 3.5)")
    add_common(export)
    export.add_argument("--models", action="append")
    export.add_argument("--fold", type=int, default=0)
    export.add_argument("--formats", action="append", help="comma-separated formats")
    export.add_argument("--exp-template", default="{model}_{tag}_fold{fold}")
    export.add_argument("--output-dir", help="export root (default: <repo>/exports)")
    export.add_argument("--num_calib", type=int, default=50)
    export.add_argument(
        "--per-tensor",
        action="store_true",
        help="reproduce the paper's per-tensor static INT8 recipe",
    )
    export.add_argument(
        "--validate", action="store_true", help="run mirasol.deployment.validate after"
    )
    return parser


BUILDERS = {
    "cv": build_cv_units,
    "ablation": build_ablation_units,
    "sweep": build_sweep_units,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the requested units and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print("\n".join(registered_models()))
        return 0
    if not args.subcommand:
        parser.print_help()
        return 2

    extra = parse_set_overrides(args.overrides)
    validate_keys(extra)

    if args.subcommand == "export":
        results = run_export(args)
    else:
        units = BUILDERS[args.subcommand](args, extra)
        if not units:
            raise DriverError("no runs selected")
        base = runs_dir(args.runs_dir)
        preflight(units, base, args)
        print(f"Planned {len(units)} unit(s); runs directory {base}")
        results = [execute_unit(unit, base, args) for unit in units]

    exit_code = print_summary(args.subcommand, results)
    if args.report:
        write_report(Path(args.report), args.subcommand, results)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DriverError as driver_error:
        print(f"ERROR: {driver_error}", file=sys.stderr)
        sys.exit(2)
