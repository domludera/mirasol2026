#!/usr/bin/env python3
"""Quantization-aware training driver (paper Sec. 3.5, docs/errata.md).

Thin CLI over :mod:`mirasol.training.qat`, which owns all of the numerical work:
inserting fake-quantization observers on the backbone, fine-tuning, converting to
a real INT8 graph and scoring it at patient level. This file only

1. assembles a **complete** parameter set from the shared task presets in
   ``run_experiment.py`` and resolves it through ``mirasol.config.resolve``, and
2. selects the FP32 starting checkpoint with the one shared selector, because
   ``mirasol.training.qat`` deliberately contains no checkpoint globbing.

It stayed a separate script rather than becoming a ``run_experiment.py``
subcommand because it is the only in-process driver: everything else launches
``mirasol.cli.train`` as a subprocess.

What changed relative to the script that produced the published QAT figure:

* configuration is no longer applied by rewriting the tracked
  ``config/esc_config.py``, so a QAT run cannot leave that file in an arbitrary
  state or inherit the previous run's fold, class count or loss weights;
* ``--qbackend`` defaults to ``qnnpack``, the ARM backend matching the Cortex-A53
  deployment target. An audit of the pre-release driver reported it used the x86
  ``fbgemm`` backend; that driver is not shipped here and was not re-run;
* a failure in ``torch.ao.quantization.convert`` propagates. The old code caught
  it and printed the last *fake-quantized* evaluation under the heading "QAT INT8
  Challenge Score"; the library now reports fake-quant and converted-INT8 as
  separate fields and raises if the conversion fails;
* PTQ reference scores are read from a measured artifact via ``--ptq-reference``
  rather than hardcoded.

``docs/errata.md`` records the status of the QAT number in the paper.

Usage::

    python experiments/run_qat.py --model-type tinyvit
    python experiments/run_qat.py --model-type efficientnet_b0 --fold 2 \\
        --qbackend qnnpack --epochs 10 \\
        --ptq-reference results/benchmarks/ptq_patient_level.json
    python experiments/run_qat.py --model-type tinyvit \\
        --checkpoint runs/tinyvit_3class_fold0/.../l-epoch=135-acc=0.880.ckpt

With ``--checkpoint`` omitted the best checkpoint of ``runs/<exp>/`` is selected,
where ``<exp>`` comes from ``--exp-name`` or ``--exp-template``.

One architecture per invocation, matching the library. Loop in the shell for
several, and the per-run exit code tells you which ones failed::

    for m in tinyvit efficientnet_b0; do
        python experiments/run_qat.py --model-type "$m" || echo "FAILED: $m"
    done
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_experiment import (  # noqa: E402  (the path shim must precede this import)
    TASKS,
    DriverError,
    format_value,
    parse_set_overrides,
    registered_models,
    runs_dir,
    select_checkpoint,
    validate_keys,
    with_derived_params,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the parser: the library's QAT flags plus the driver's own.

    ``mirasol.training.qat.add_qat_arguments`` owns ``--model-type``,
    ``--checkpoint``, ``--fold``, ``--epochs``, ``--lr``, ``--batch-size``,
    ``--num-workers``, ``--qbackend``, ``--eval-every``, ``--ptq-reference``,
    ``--output`` and ``--device``, so those cannot drift from
    :class:`~mirasol.training.qat.QATSettings`.
    """
    from mirasol.training.qat import add_qat_arguments

    parser = argparse.ArgumentParser(
        prog="run_qat.py",
        description="Quantization-aware training for one MIRASOL heart-sound model.",
    )
    add_qat_arguments(parser)
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="3class",
        help="labelling scheme; a complete parameter set (see run_experiment.py)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra config override, repeatable; validated against the defaults",
    )
    parser.add_argument("--config", help="YAML file of configuration overrides")
    parser.add_argument(
        "--exp-name",
        help="experiment whose checkpoint to start from (default: --exp-template)",
    )
    parser.add_argument(
        "--exp-template",
        default="{model}_{tag}_fold{fold}",
        help="how to derive the experiment name from model, task tag and fold",
    )
    parser.add_argument("--runs-dir", help="root for runs/<exp>/ (default <repo>/runs)")
    parser.add_argument(
        "--checkpoint-strategy",
        choices=("acc", "latest"),
        default="acc",
        help="how to pick the starting checkpoint when --checkpoint is omitted",
    )
    return parser


def relax_required(parser: argparse.ArgumentParser, *dests: str) -> None:
    """Make the named library arguments optional.

    ``add_qat_arguments`` marks ``--checkpoint`` required because the library
    refuses to guess one. This driver *does* have a checkpoint selector, so the
    flag becomes optional here and is filled in from ``runs/<exp>/`` when absent.
    Nothing else about the library's flag definitions is touched.
    """
    wanted = set(dests)
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public API
        if action.dest in wanted:
            action.required = False


def experiment_name(args: argparse.Namespace) -> str:
    """Return the experiment whose checkpoint this run starts from."""
    if args.exp_name:
        return args.exp_name
    return args.exp_template.format(
        model=args.model_type,
        task=args.task,
        tag=TASKS[args.task].exp_tag,
        fold=args.fold,
        label="",
    )


def resolve_checkpoint(args: argparse.Namespace, exp_name: str) -> Path:
    """Return the FP32 checkpoint to start from, explicit or selected."""
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser()
        if not checkpoint.is_file():
            raise DriverError(f"--checkpoint not found: {checkpoint}")
        return checkpoint
    exp_dir = runs_dir(args.runs_dir) / exp_name
    selected = select_checkpoint(exp_dir, args.checkpoint_strategy)
    print(
        f"  selected checkpoint: {selected.path} "
        f"(acc={selected.accuracy}, version={selected.version})"
    )
    return selected.path


def build_overrides(
    args: argparse.Namespace, exp_name: str, extra: Mapping[str, str]
) -> tuple[str, ...]:
    """Assemble the complete ``key=value`` override set for this QAT run.

    Every parameter the run depends on is written here, so nothing is inherited:
    the task preset fixes ``classes_num``, ``loss_weight``, ``loss_type`` and the
    dataset, and ``model_type``/``esc_fold`` are cross-checked by
    ``mirasol.training.qat.run_qat`` against the CLI flags.
    """
    params: dict[str, Any] = TASKS[args.task].params()
    params["model_type"] = args.model_type
    params["esc_fold"] = args.fold
    params["exp_name"] = exp_name
    params["batch_size"] = args.batch_size
    # The starting checkpoint is loaded by the QAT library from --checkpoint;
    # the config must not also try to resume from whatever the defaults name.
    params["resume_checkpoint"] = None
    params.update(extra)
    validate_keys(params)
    resolved = with_derived_params(params)
    return tuple(f"{key}={format_value(resolved[key])}" for key in sorted(resolved))


def default_output(args: argparse.Namespace, exp_name: str) -> Path | None:
    """Return an explicit result path when ``--runs-dir`` moved the run root.

    The library defaults to ``runs/<exp>/qat_<model>_fold<n>.json`` resolved from
    the environment; if the caller redirected the run root on the command line
    the result must follow it.
    """
    if args.output is not None or not args.runs_dir:
        return None
    return (
        runs_dir(args.runs_dir)
        / exp_name
        / f"qat_{args.model_type}_fold{args.fold}.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run QAT for one architecture and return an exit code."""
    from mirasol.training.qat import run_qat, settings_from_args
    from mirasol.training.utils import build_config

    parser = build_parser()
    relax_required(parser, "checkpoint")
    args = parser.parse_args(argv)

    available = registered_models()
    if args.model_type not in available:
        raise DriverError(
            f"unknown model {args.model_type!r}. Registered architectures: "
            + ", ".join(available)
        )
    extra = parse_set_overrides(args.overrides)
    validate_keys(extra)

    exp_name = experiment_name(args)
    print(f"{'=' * 72}")
    print(f"QAT: {args.model_type}  task={args.task}  fold={args.fold}  exp={exp_name}")
    print(f"{'=' * 72}")

    args.checkpoint = resolve_checkpoint(args, exp_name)
    overrides = build_overrides(args, exp_name, extra)
    for item in overrides:
        print(f"    {item}")

    redirected = default_output(args, exp_name)
    if redirected is not None:
        args.output = redirected

    config = build_config(config_path=args.config, overrides=overrides)
    settings = settings_from_args(args)
    result = run_qat(config, settings)

    print(
        "\n  The INT8 score above is measured on a converted quantized graph. "
        "The\n  library raises rather than substituting a fake-quantized "
        "measurement, which\n  is how the published figure was produced - see "
        "docs/errata.md."
    )
    return 0 if result.int8 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DriverError as driver_error:
        print(f"ERROR: {driver_error}", file=sys.stderr)
        sys.exit(2)
