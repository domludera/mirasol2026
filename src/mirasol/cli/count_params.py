"""Regenerate the parameter counts printed in Table 1's column headers.

Paper artifact served: Table 1 of the camera-ready, whose column headers read
``(1.52M) (3.26M) (4.01M) (5.07M) (11.2M) (27.6M)`` for MobileNetV3-S, FastViT,
EfficientNet-B0, TinyViT, ResNet-18 and HTS-AT, and the "18x parameter range"
claim in the abstract and Sec. 2.2. Nothing in the original tree regenerated
these numbers.

The counting convention, measured with this script:

* the reported figure is **trainable** parameters for the single-channel,
  three-class configuration;
* the mel front end contributes 20,928 parameters, of which 20,800 are frozen
  (torchlibrosa's STFT ``conv_real``/``conv_imag`` bases, 8,320 each, and the
  4,160-weight mel filterbank ``melW``) and are therefore *excluded*, while
  ``bn0``'s 128 affine parameters are trainable and *included*;
* HTS-AT is the one exception: the paper's 27.57M is its **total** parameter
  count (27,569,127); its trainable count is 27,548,327, i.e. 27.55M. The script
  prints both columns and states which convention reproduces each published
  figure rather than picking one silently.

Usage::

    python -m mirasol.cli.count_params
    python -m mirasol.cli.count_params --models efficientnet_b0 htsat --json out.json
    python -m mirasol.cli.count_params --check     # non-zero exit on a mismatch

Building a model goes through :func:`mirasol.models.factory.create_model`, which
loads the ImageNet weights each architecture was initialised from, so this needs
either network access or a populated timm/torchvision cache. Parameter counts do
not depend on the weight *values*, only on the shapes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from mirasol.training.utils import build_config

__all__ = [
    "FRONT_END_PREFIXES",
    "ParameterCount",
    "TABLE1_MODELS",
    "count_parameters",
    "main",
]


@dataclass(frozen=True)
class TableColumn:
    """One Table 1 column: architecture key, display name, published figure."""

    model_type: str
    display_name: str
    published_millions: float


#: Table 1 column order, left to right, with the figure printed in the header.
TABLE1_MODELS: tuple[TableColumn, ...] = (
    TableColumn("mobilenetv3", "MobileNetV3-S", 1.52),
    TableColumn("fastvit", "FastViT", 3.26),
    TableColumn("efficientnet_b0", "EfficientNet-B0", 4.01),
    TableColumn("tinyvit", "TinyViT", 5.07),
    TableColumn("resnet18", "ResNet-18", 11.17),
    TableColumn("htsat", "HTS-AT", 27.57),
)

#: Parameter-name prefixes belonging to the shared mel front end rather than to
#: the architecture under comparison.
FRONT_END_PREFIXES: tuple[str, ...] = (
    "spectrogram_extractor",
    "logmel_extractor",
    "spec_augmenter",
    "bn0",
)

#: Default tolerance when checking a measured count against the published one,
#: in millions of parameters (5,000 parameters).
DEFAULT_TOLERANCE_M = 0.005

#: Class weights that satisfy the configuration layer's length invariant for each
#: supported class count. They do not affect a parameter count; only the head
#: width does.
_LOSS_WEIGHTS: dict[int, str] = {3: "4.0,10.0,1.0", 2: "4.0,1.0"}


@dataclass(frozen=True)
class ParameterCount:
    """Measured parameter counts for one architecture."""

    model_type: str
    display_name: str
    trainable: int
    total: int
    frozen: int
    front_end: int
    front_end_frozen: int
    published_millions: float
    matching_convention: str

    @property
    def trainable_millions(self) -> float:
        """Trainable parameters in millions."""
        return self.trainable / 1e6

    @property
    def total_millions(self) -> float:
        """Total parameters in millions."""
        return self.total / 1e6


def count_parameters(
    model: nn.Module,
    column: TableColumn,
    tolerance_millions: float = DEFAULT_TOLERANCE_M,
) -> ParameterCount:
    """Count the parameters of one built model and match them to the paper.

    Args:
        model: an instantiated architecture.
        column: the Table 1 column it corresponds to.
        tolerance_millions: how close a measured figure must be to the published
            one to count as reproducing it.

    Returns:
        A :class:`ParameterCount`. ``matching_convention`` is ``'trainable'``,
        ``'total'`` or ``'none'``.
    """
    total = 0
    trainable = 0
    frozen = 0
    front_end = 0
    front_end_frozen = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count
        if name.startswith(FRONT_END_PREFIXES):
            front_end += count
            if not parameter.requires_grad:
                front_end_frozen += count

    published = column.published_millions
    if abs(published - trainable / 1e6) <= tolerance_millions:
        convention = "trainable"
    elif abs(published - total / 1e6) <= tolerance_millions:
        convention = "total"
    else:
        convention = "none"

    return ParameterCount(
        model_type=column.model_type,
        display_name=column.display_name,
        trainable=trainable,
        total=total,
        frozen=frozen,
        front_end=front_end,
        front_end_frozen=front_end_frozen,
        published_millions=published,
        matching_convention=convention,
    )


def _build_and_count(
    column: TableColumn,
    classes_num: int,
    config_path: Path | None,
    overrides: Sequence[str],
    tolerance_millions: float,
) -> ParameterCount:
    """Build one architecture from the resolved configuration and count it."""
    from mirasol.models.factory import create_model

    # loss_weight is set alongside classes_num because the configuration layer
    # rejects a weight vector whose length disagrees with the class count - the
    # invariant whose violation made Table S7 unreproducible.
    config = build_config(
        config_path=config_path,
        overrides=(
            *overrides,
            f"model_type={column.model_type}",
            f"classes_num={classes_num}",
            f"loss_weight={_LOSS_WEIGHTS[classes_num]}",
        ),
    )
    model = create_model(config)
    return count_parameters(model, column, tolerance_millions)


def _print_table(counts: Sequence[ParameterCount]) -> None:
    header = (
        f"{'Architecture':<18}{'Trainable':>12}{'Total':>12}{'Frozen':>9}"
        f"{'Front end':>11}{'Paper':>9}{'Reproduced by':>16}"
    )
    print(header)
    print("-" * len(header))
    for count in counts:
        print(
            f"{count.display_name:<18}{count.trainable:>12,}{count.total:>12,}"
            f"{count.frozen:>9,}{count.front_end:>11,}"
            f"{count.published_millions:>8.2f}M{count.matching_convention:>16}"
        )


def _print_latex_header(counts: Sequence[ParameterCount]) -> None:
    cells = " & ".join(
        rf"\textbf{{({count.trainable_millions:.2f}M)}}" for count in counts
    )
    print("\nTable 1 header row, trainable parameters:")
    print(f" & {cells} \\\\")


def _print_notes(counts: Sequence[ParameterCount]) -> None:
    front_end = {count.front_end for count in counts}
    front_end_frozen = {count.front_end_frozen for count in counts}
    print("\nConvention:")
    print(
        "  Reported figures are TRAINABLE parameters for the single-channel, "
        "three-class configuration."
    )
    if len(front_end) == 1 and len(front_end_frozen) == 1:
        total_front_end = front_end.pop()
        frozen_front_end = front_end_frozen.pop()
        print(
            f"  The shared mel front end holds {total_front_end:,} parameters, of "
            f"which {frozen_front_end:,} are frozen (torchlibrosa STFT bases and mel "
            f"filterbank) and excluded; the remaining "
            f"{total_front_end - frozen_front_end:,} are bn0's trainable affine "
            "parameters and are included."
        )
    for count in counts:
        if count.matching_convention == "total":
            print(
                f"  {count.display_name}: the published {count.published_millions:.2f}M "
                f"is the TOTAL parameter count ({count.total:,}); its trainable count "
                f"is {count.trainable:,} ({count.trainable_millions:.3f}M)."
            )
        elif count.matching_convention == "none":
            print(
                f"  {count.display_name}: MISMATCH -- published "
                f"{count.published_millions:.2f}M matches neither trainable "
                f"({count.trainable_millions:.3f}M) nor total "
                f"({count.total_millions:.3f}M)."
            )
    ratios = [count.trainable_millions for count in counts]
    if len(ratios) > 1:
        print(
            f"  Parameter range: {min(ratios):.2f}M to {max(ratios):.2f}M, a factor of "
            f"{max(ratios) / min(ratios):.1f}."
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``count_params`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mirasol-count-params",
        description="Regenerate Table 1's parameter-count column headers.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[column.model_type for column in TABLE1_MODELS],
        default=[column.model_type for column in TABLE1_MODELS],
        help="architectures to count, default all six in Table 1 order",
    )
    parser.add_argument(
        "--classes-num",
        type=int,
        default=3,
        choices=sorted(_LOSS_WEIGHTS),
        help="classifier outputs; 3 is the configuration Table 1 reports",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML file of configuration overrides"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override one configuration key (repeatable)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_M,
        metavar="MILLIONS",
        help=f"match tolerance in millions of parameters (default {DEFAULT_TOLERANCE_M})",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the counts to this JSON file"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if a published figure matches neither convention",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    requested = [
        column for column in TABLE1_MODELS if column.model_type in set(args.models)
    ]
    counts = [
        _build_and_count(
            column, args.classes_num, args.config, args.overrides, args.tolerance
        )
        for column in requested
    ]
    _print_table(counts)
    _print_latex_header(counts)
    _print_notes(counts)

    if args.json is not None:
        payload: dict[str, Any] = {
            "classes_num": args.classes_num,
            "tolerance_millions": args.tolerance,
            "models": [asdict(count) for count in counts],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")

    mismatched = [c.display_name for c in counts if c.matching_convention == "none"]
    if args.check and mismatched:
        print(
            f"\nFAIL: {', '.join(mismatched)} reproduce neither the trainable nor the "
            "total convention within the tolerance."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
