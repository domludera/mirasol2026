"""Regenerate Table S8: how much optimism the checkpoint-selection rule can carry.

Paper artifact
--------------
Supplementary Table S8, and the Sec. 2.3 sentence that quotes its range ("the
peak retained checkpoint exceeds the mean of the four retained checkpoints by
0.38-1.44 pp, mean 0.79 pp"), which is compared there against the 2.1 pp spread
of Challenge Scores across the six architectures.

Nothing in the original code base regenerated this table: it was produced by
globbing 26 GB of Lightning checkpoints, so an empty glob printed ``nan`` and the
result depended on how many times a fold happened to be re-run. This script reads
``results/checkpoint_accuracies.csv`` instead -- a few hundred bytes carrying one
row per retained checkpoint -- so the table is reproducible from a clean clone.

Input schema
------------
``results/checkpoint_accuracies.csv``, one row per retained checkpoint file::

    model,fold,version,checkpoint,monitor,epoch,accuracy

``model`` is the architecture key (``mobilenetv3``, ``fastvit``,
``efficientnet_b0``, ``tinyvit``, ``resnet18``, ``htsat``); ``version`` is the
Lightning ``version_N`` index as an integer; ``checkpoint`` is the checkpoint
filename, which is what makes the ``-v1`` duplicates visible; ``accuracy`` is the
held-out accuracy parsed from that filename. A file carrying ``exp_name`` instead
of ``model``/``fold`` is also accepted and parsed as ``{model}_3class_fold{N}``.

The two conventions that change the answer
------------------------------------------
1. ``--selection``. Some folds have several ``version_*`` directories because the
   run was restarted; only one of them is the completed 150-epoch run (for
   example ``mobilenetv3`` fold 0 has ten, of which only ``version_9`` trained to
   completion). ``pooled`` takes the best four accuracies over *all* version
   directories of a fold and reproduces the published table; ``per-run`` first
   keeps only the version that reached the highest epoch.
2. ``--duplicates``. Because two monitors are active, a checkpoint is sometimes
   written twice under the same epoch and accuracy with a ``-v1`` suffix; four of
   the thirty three-class folds contain such a pair. ``count`` treats the two
   files as two retained checkpoints and reproduces the published table;
   ``dedup`` counts the pair once.

The defaults (``pooled`` / ``count``) are the conventions behind the printed
Table S8. Every run prints all four combinations so the sensitivity of the table
to these two choices is visible rather than argued about.

Usage
-----
    python -m analysis.checkpoint_selection
    python -m analysis.checkpoint_selection --selection per-run --duplicates dedup
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analysis._common import (
    ARCH_KEYS,
    FOLDS,
    d3,
    print_header,
    print_latex_comment,
    resolve_results_dir,
)

#: Each Lightning run retains this many checkpoints (``save_top_k=2`` on two
#: monitors), and Table S8 compares the best of them against their mean.
RETAINED_PER_RUN: Final[int] = 4

#: Lightning's suffix for a checkpoint written to an already-existing filename.
DUPLICATE_SUFFIX: Final[re.Pattern[str]] = re.compile(r"-v\d+(?=\.ckpt$)")

_ACCURACY_ALIASES: Final[tuple[str, ...]] = ("accuracy", "acc", "held_out_accuracy")
_CHECKPOINT_ALIASES: Final[tuple[str, ...]] = ("checkpoint", "filename", "ckpt")
_EXP_ALIASES: Final[tuple[str, ...]] = ("exp_name", "experiment", "exp")

_EXP_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<model>.+?)_3class_fold(?P<fold>\d+)$")

SELECTIONS: Final[tuple[str, ...]] = ("pooled", "per-run")
DUPLICATE_POLICIES: Final[tuple[str, ...]] = ("count", "dedup")


@dataclass(frozen=True)
class FoldSummary:
    """Peak and mean-of-retained held-out accuracy for one (model, fold)."""

    model: str
    fold: int
    accuracies: tuple[float, ...]

    @property
    def peak(self) -> float:
        return max(self.accuracies)

    @property
    def mean_retained(self) -> float:
        return float(np.mean(self.accuracies))


def _pick_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in frame.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def load_checkpoints(path: Path) -> pd.DataFrame:
    """Read the checkpoint inventory and normalise it to a known column set.

    Returns a frame with columns ``model``, ``fold``, ``version``, ``checkpoint``,
    ``epoch`` and ``accuracy``. Any deviation from the documented schema is an
    error naming the columns that were found.
    """
    frame = pd.read_csv(path)
    accuracy_column = _pick_column(frame, _ACCURACY_ALIASES)
    if accuracy_column is None:
        raise SystemExit(
            f"error: '{path.name}' has no accuracy column "
            f"(looked for {list(_ACCURACY_ALIASES)}); found {list(frame.columns)}."
        )
    normalised = pd.DataFrame({"accuracy": frame[accuracy_column].astype(float)})

    if "model" in frame.columns and "fold" in frame.columns:
        normalised["model"] = frame["model"].astype(str)
        normalised["fold"] = frame["fold"].astype(int)
    else:
        exp_column = _pick_column(frame, _EXP_ALIASES)
        if exp_column is None:
            raise SystemExit(
                f"error: '{path.name}' needs either 'model' and 'fold' columns or an "
                f"experiment-name column; found {list(frame.columns)}."
            )
        parsed = frame[exp_column].astype(str).str.extract(_EXP_PATTERN)
        if parsed["model"].isna().any():
            bad = frame.loc[parsed["model"].isna(), exp_column].unique()[:3]
            raise SystemExit(
                f"error: '{path.name}' has experiment names that do not match "
                f"'{{model}}_3class_fold{{N}}', e.g. {list(bad)}."
            )
        normalised["model"] = parsed["model"]
        normalised["fold"] = parsed["fold"].astype(int)

    checkpoint_column = _pick_column(frame, _CHECKPOINT_ALIASES)
    normalised["checkpoint"] = (
        frame[checkpoint_column].astype(str) if checkpoint_column else ""
    )
    normalised["version"] = (
        frame["version"].astype(int) if "version" in frame.columns else 0
    )
    normalised["epoch"] = frame["epoch"].astype(int) if "epoch" in frame.columns else -1

    # Carry the experiment name through when the source has one. summarise()
    # prefers it over (model, fold) because the table also holds the single-fold
    # runs that predate the five-fold protocol, and those share a model and a
    # fold number with the protocol runs without belonging to them.
    exp_column = _pick_column(frame, _EXP_ALIASES)
    if exp_column is not None:
        normalised["experiment"] = frame[exp_column].astype(str)
    return normalised


def _completed_run(group: pd.DataFrame) -> pd.DataFrame:
    """Keep only the version directory that reached the highest epoch."""
    if (group["epoch"] < 0).all():
        raise SystemExit(
            "error: --selection per-run needs an 'epoch' column to tell a completed "
            "run from a restarted one."
        )
    best_version = int(group.loc[group["epoch"].idxmax(), "version"])
    return group[group["version"] == best_version]


def summarise(
    frame: pd.DataFrame, *, selection: str, duplicates: str, top_k: int = RETAINED_PER_RUN
) -> dict[str, list[FoldSummary]]:
    """Per-fold peak and mean-of-retained accuracy for every architecture."""
    if selection not in SELECTIONS:
        raise ValueError(f"unknown selection strategy {selection!r}")
    if duplicates not in DUPLICATE_POLICIES:
        raise ValueError(f"unknown duplicate policy {duplicates!r}")

    summaries: dict[str, list[FoldSummary]] = {}
    for display, model in ARCH_KEYS.items():
        per_fold: list[FoldSummary] = []
        for fold in FOLDS:
            # Key on the exact experiment name, not on (model, fold). The
            # checkpoint table also contains the single-fold runs that predate the
            # five-fold protocol -- experiments named `<model>_3class` with no
            # `_fold<n>` suffix, which were evaluated on fold 0. Selecting by
            # (model, fold) silently merges those into fold 0 for FastViT and
            # HTS-AT and shifts their peak accuracy, so Table S8 stops
            # reproducing. It also folds in the two-class, ablation and
            # class-weight-sweep checkpoints, which are different experiments
            # entirely.
            experiment = f"{model}_3class_fold{fold}"
            if "experiment" in frame.columns:
                group = frame[frame["experiment"] == experiment]
            else:
                group = frame[(frame["model"] == model) & (frame["fold"] == fold)]
            if group.empty:
                raise SystemExit(
                    f"error: no checkpoint rows for experiment {experiment!r}; "
                    "Table S8 needs all five folds of all six architectures."
                )
            if selection == "per-run":
                group = _completed_run(group)
            if duplicates == "dedup":
                group = group[~group["checkpoint"].str.contains(DUPLICATE_SUFFIX)]
            accuracies = tuple(
                sorted(group["accuracy"].tolist(), reverse=True)[:top_k]
            )
            if not accuracies:
                raise SystemExit(
                    f"error: {model} fold {fold} has no checkpoints left after the "
                    f"'{duplicates}' duplicate policy."
                )
            per_fold.append(FoldSummary(model=model, fold=fold, accuracies=accuracies))
        summaries[display] = per_fold
    return summaries


def aggregate(per_fold: Sequence[FoldSummary]) -> tuple[float, float, float]:
    """Mean peak, mean of the retained means, and their gap in percentage points."""
    peak = float(np.mean([s.peak for s in per_fold]))
    retained = float(np.mean([s.mean_retained for s in per_fold]))
    return peak, retained, 100.0 * (peak - retained)


def latex_table(summaries: Mapping[str, Sequence[FoldSummary]]) -> str:
    """The Table S8 tabular body."""
    lines = []
    for display, per_fold in summaries.items():
        peak, retained, gap = aggregate(per_fold)
        lines.append(f"{display} & {d3(peak)} & {d3(retained)} & {gap:.2f} \\\\")
    return "\n".join(lines)


def print_variants(frame: pd.DataFrame) -> None:
    """Print the gap under all four selection / duplicate conventions."""
    print()
    print_header("Sensitivity of Table S8 to the two counting conventions (gap, pp)")
    header = f"{'Model':<18}" + "".join(
        f"{s + '/' + d:>16}" for s in SELECTIONS for d in DUPLICATE_POLICIES
    )
    print(header)
    print("-" * len(header))
    columns = {
        (selection, duplicates): summarise(
            frame, selection=selection, duplicates=duplicates
        )
        for selection in SELECTIONS
        for duplicates in DUPLICATE_POLICIES
    }
    for display in ARCH_KEYS:
        line = f"{display:<18}"
        for selection in SELECTIONS:
            for duplicates in DUPLICATE_POLICIES:
                _, _, gap = aggregate(columns[(selection, duplicates)][display])
                line += f"{gap:>16.2f}"
        print(line)
    print(
        "\n  pooled/count is the convention behind the printed Table S8. per-run\n"
        "  restricts each fold to the version directory that reached the highest\n"
        "  epoch; dedup drops Lightning's '-vN' duplicate of an identical checkpoint."
    )


def print_detail(summaries: Mapping[str, Sequence[FoldSummary]]) -> None:
    """Per-fold retained accuracies, so every input to the table is visible."""
    print()
    print_header("Retained checkpoint accuracies per fold")
    for display, per_fold in summaries.items():
        for summary in per_fold:
            values = "  ".join(f"{a:.3f}" for a in summary.accuracies)
            print(
                f"  {display:<16} fold {summary.fold}: {values}"
                f"   peak={summary.peak:.3f} mean={summary.mean_retained:.3f}"
            )
        peak, retained, gap = aggregate(per_fold)
        print(
            f"  {display:<16} over folds: peak={peak:.4f} "
            f"mean_retained={retained:.4f} gap={gap:.2f} pp"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.checkpoint_selection",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help="checkpoint inventory (default: results/checkpoint_accuracies.csv)",
    )
    parser.add_argument(
        "--selection",
        choices=SELECTIONS,
        default="pooled",
        help="pool all version_* directories of a fold, or keep the completed run",
    )
    parser.add_argument(
        "--duplicates",
        choices=DUPLICATE_POLICIES,
        default="count",
        help="count Lightning's '-vN' duplicate checkpoints, or drop them",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="also write the LaTeX table body to this file",
    )
    return parser


def resolve_checkpoints_csv(cli: Path | None) -> Path:
    """Resolve the checkpoint inventory, defaulting into ``results/``."""
    if cli is not None:
        if not cli.is_file():
            raise SystemExit(f"error: checkpoint inventory not found at '{cli.name}'.")
        return cli
    path = resolve_results_dir(None) / "checkpoint_accuracies.csv"
    if not path.is_file():
        raise SystemExit(
            "error: results/checkpoint_accuracies.csv not found. It ships with the "
            "repository; pass --checkpoints-csv to use your own inventory."
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_checkpoints(resolve_checkpoints_csv(args.checkpoints_csv))

    summaries = summarise(
        frame, selection=args.selection, duplicates=args.duplicates
    )
    print_detail(summaries)
    print_variants(frame)

    gaps = [aggregate(per_fold)[2] for per_fold in summaries.values()]
    print(
        f"\nselection={args.selection} duplicates={args.duplicates}: "
        f"gap {min(gaps):.2f}-{max(gaps):.2f} pp, mean {np.mean(gaps):.2f} pp"
    )

    print_latex_comment("TABLE S8 -- checkpoint-selection sensitivity analysis")
    body = latex_table(summaries)
    print(body)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body + "\n")
        print(f"\nwrote {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
