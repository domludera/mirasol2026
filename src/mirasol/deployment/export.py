# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
#
# Scope of the derivation: this module re-exports the export wrappers defined in
# mirasol/deployment/wrappers.py, one of which (HTSATExportWrapper.forward) is a
# re-derivation of HTS_AT_Swin_Transformer.reshape_wav2img, whose spatial
# reshaping in turn follows Microsoft's Swin-Transformer. wrappers.py carries the
# same notice. Everything else in this module is new work.
"""Export trained checkpoints to FP32 / INT8 / FP16 ONNX for edge deployment.

Paper artifacts: the graphs measured in Table 4 (on-device latency, memory and
size) and evaluated in Sec. 3.5 (quantization).  The graphs themselves are
published as GitHub Release assets, not committed; this module is how they are
produced, and it writes an ``export_manifest.json`` recording exactly which
checkpoint each one came from.

Quantization recipes
--------------------
``--dynamic`` (``model_int8.onnx``)
    ``quantize_dynamic(weight_type=QInt8)``.  Weight-only; activations stay
    FP32.  Sec. 3.5(1).

``--static`` (``model_int8_static.onnx``)
    ``quantize_static(quant_format=QDQ, weight_type=QInt8,
    activation_type=QInt8, calibrate_method=MinMax, reduce_range=False)`` over
    50 real calibration segments, and **per_channel=True by default**.

``--fp16`` (``model_fp16.onnx``)
    ``onnxconverter_common.float16.convert_float_to_float16``.  Sec. 3.5(2).

Why per-channel is the default, and what the flag reproduces
------------------------------------------------------------
The recipe that produced the paper's published static-INT8 graphs left
``per_channel`` unset, i.e. per-tensor weight scales.  Measured on 200 fold-0
segments / 126 patients, patient-level majority vote, scored with the project's
Challenge Score:

===============  =====  ==============  ==============  ==================
Architecture     FP32   per-tensor      per-channel     delta vs FP32
===============  =====  ==============  ==============  ==================
EfficientNet-B0  0.883  0.786 constant  **0.862**       -2.4%
MobileNetV3-S    0.887  0.786 constant  0.812           -8.5%
TinyViT          0.872  0.786 constant  0.746           -14.5%, still poor
FastViT          0.879  0.314           0.264           still broken
===============  =====  ==============  ==============  ==================

0.786 is exactly the weighted score of an all-Absent classifier: with
per-tensor scales three of the four graphs collapse to a constant class, and
the control reproduces the published 0.784.  With per-channel weight scales
both CNNs recover and neither transformer does, which supports the paper's
*mechanism* claim (attention layers are more sensitive to reduced precision)
more cleanly than the published numbers did, while making its *magnitude*
claim -- the abstract's 12-65% -- specific to the per-tensor export.

So: ``per_channel=True`` is the default because it is the recipe that yields
deployable CNN graphs, and ``--per-tensor`` reproduces the published graphs
bit-for-bit.  Two caveats a reader needs:

* per-channel INT8 latency on the Cortex-A53 was **not measured** (the Orange
  Pi was unavailable); per-channel can be slower, and Table 4's 210 ms /
  "fastest and smallest" figures describe the per-tensor build.  The
  deployment recommendation in the paper is unchanged by this module.
* graphs exported with ``--per-tensor`` are reproduction artifacts, not
  deployable models.  The constant-classifier guard in
  :mod:`mirasol.deployment.calibration` refuses to write them unless
  ``--allow-degenerate`` is passed, and QUInt8 activations, percentile
  calibration and 5x calibration data were each measured to change nothing on
  their own.

Verified end to end on the fold-0 EfficientNet-B0 checkpoint, over 60
class-stratified hold-out segments (20 Present / 20 Unknown / 20 Absent) on
which the FP32 graph is 60/60 correct:

* ``--static`` (per-channel, the default) -> 8 Present / 5 Unknown / 47 Absent,
  segment-level Challenge Score 0.817 vs 0.830 FP32;
* ``--static --per-tensor`` -> 0 / 0 / 60, i.e. exactly the published graph's
  collapse, and the guard blocks it;
* ``--dynamic`` -> 0 / 0 / 60 for this architecture, so the guard blocks it too.
  The dynamic-INT8 bundle published with the paper is evidence for Sec. 3.5(1)
  (where EfficientNet-B0 loses 82.8% prediction agreement), not a recommended
  build, so regenerating it needs ``--allow-degenerate``.  HTS-AT's dynamic
  INT8, by contrast, matches its FP32 graph.

Usage
-----
    python -m mirasol.deployment.export --model-type efficientnet_b0 \\
        --checkpoint runs/effnetb0_3class_fold0/.../l-epoch=142-acc=0.855.ckpt \\
        --static --fp16
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import numpy as np
import torch
import torch.nn as nn

from mirasol.deployment.calibration import (
    DEFAULT_CALIBRATION_SEGMENTS,
    DEFAULT_HOLDOUT_SEGMENTS,
    CalibrationSet,
    assert_not_constant,
    build_ort_reader,
    constant_prediction_report,
    load_calibration_set,
)
from mirasol.deployment.manifest import (
    CheckpointProvenance,
    manifest_entry,
    write_manifest,
)

# Re-exported so that `from mirasol.deployment.export import get_export_wrapper`
# keeps working; the HTS-AT/Swin-derived code itself lives in wrappers.py, which
# carries the upstream notice.
from mirasol.deployment.wrappers import (  # noqa: F401
    MODEL_TYPES,
    EfficientNetExportWrapper,
    HTSATExportWrapper,
    TimmExportWrapper,
    get_export_wrapper,
)
from mirasol.paths import REPO_ROOT

try:
    from mirasol.signal.mel import waveform_to_mel_spectrogram
except ImportError as error:  # pragma: no cover - packaging contract
    raise ImportError(
        "mirasol.signal.mel must provide "
        "waveform_to_mel_spectrogram(waveform) -> (1, time_steps, mel_bins) "
        "float32; deployment/edge/mel.py is the vendored reference "
        "implementation of the same computation."
    ) from error

# ONNX opset 17 with the TorchScript exporter (dynamo=False). The dynamo path
# emits a graph that onnxruntime.quantization cannot consume, so it is pinned.
ONNX_OPSET: Final[int] = 17
INPUT_NAME: Final[str] = "mel_spectrogram"
OUTPUT_NAME: Final[str] = "logits"

# The 30 s clip is concatenated with itself before the STFT, as at training time.
CLIP_REPEATS: Final[int] = 2


# ═════════════════════════════════════════════════════════════════════════════
# Configuration and checkpoint loading
# ═════════════════════════════════════════════════════════════════════════════

def config_snapshot(**overrides: object) -> SimpleNamespace:
    """Immutable-by-convention copy of ``mirasol.config.defaults`` + overrides.

    The original export script assigned to ``config.model_type`` and restored
    it afterwards, so an exception mid-export left the shared defaults module
    describing a different architecture.  Nothing here mutates the module.

    Raises:
        KeyError: an override names a key the defaults module does not define,
            which is always a typo.
    """
    from mirasol.config import defaults

    values = {
        name: getattr(defaults, name)
        for name in dir(defaults)
        if not name.startswith("_")
        and not callable(getattr(defaults, name))
        and not inspect.ismodule(getattr(defaults, name))
    }
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise KeyError(
            f"unknown config key(s) {unknown}; mirasol.config.defaults defines "
            f"{len(values)} keys"
        )
    values.update(overrides)
    return SimpleNamespace(**values)


def load_sed_model(model_type: str, checkpoint_path: Path, classes_num: int) -> nn.Module:
    """Rebuild the architecture and load a Lightning checkpoint into it.

    The Lightning ``state_dict`` prefixes every model tensor with
    ``sed_model.`` and adds a top-level ``loss_weight`` buffer; the prefix is
    stripped and the model is loaded with ``strict=True``.  The original script
    fell back to ``strict=False`` when the strict load raised, which silently
    exported a partially-initialised network -- that fallback is gone: a
    checkpoint that does not fit the architecture is an error.

    Raises:
        FileNotFoundError: the checkpoint is missing.
        RuntimeError: the checkpoint does not match the architecture.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    from mirasol.models.factory import create_model

    config = config_snapshot(model_type=model_type, classes_num=classes_num)
    sed_model = create_model(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    prefix = "sed_model."
    model_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not model_state:
        raise RuntimeError(
            f"{checkpoint_path} has no '{prefix}*' tensors; it does not look "
            "like a mirasol training checkpoint"
        )
    sed_model.load_state_dict(model_state, strict=True)
    sed_model.eval()
    return sed_model

# ═════════════════════════════════════════════════════════════════════════════
# Export steps
# ═════════════════════════════════════════════════════════════════════════════

def reference_input(config: SimpleNamespace) -> torch.Tensor:
    """A real-shaped dummy input, produced by the deployed front end.

    Shape only: the values come from a silent waveform, which is fine for
    tracing but useless for validating a quantized graph -- hence the separate
    hold-out check in :mod:`mirasol.deployment.calibration`.
    """
    silence = np.zeros(config.clip_samples * CLIP_REPEATS, dtype=np.float32)
    mel = waveform_to_mel_spectrogram(silence)
    return torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)).unsqueeze(0)


def export_fp32(
    export_model: nn.Module, dummy_input: torch.Tensor, output_path: Path
) -> Path:
    """Trace the wrapper to ONNX with a dynamic batch axis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=ONNX_OPSET,
        dynamo=False,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch_size"}, OUTPUT_NAME: {0: "batch_size"}},
    )
    return output_path


def quantize_int8_dynamic(fp32_path: Path, output_path: Path) -> Path:
    """Weight-only INT8. Sec. 3.5(1)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


def quantize_int8_static(
    fp32_path: Path,
    output_path: Path,
    calibration: CalibrationSet,
    per_channel: bool = True,
) -> Path:
    """Static INT8 (QDQ, QInt8 weights and activations, MinMax calibration).

    Args:
        fp32_path: the traced FP32 graph.
        output_path: destination ``model_int8_static.onnx``.
        calibration: real training-fold segments.
        per_channel: per-channel weight scales.  True is the default and the
            recipe that keeps the CNN graphs usable; False reproduces the
            paper's published graphs.  See the module docstring.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantType, quantize_static

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(output_path),
        calibration_data_reader=build_ort_reader(calibration.calibration, INPUT_NAME),
        quant_format=ort.quantization.QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        reduce_range=False,
    )
    return output_path


def convert_fp16(fp32_path: Path, output_path: Path) -> Path:
    """FP16 conversion via onnxconverter-common. Sec. 3.5(2)."""
    import onnx
    from onnxconverter_common import float16

    onnx.save(
        float16.convert_float_to_float16(onnx.load(str(fp32_path))), str(output_path)
    )
    return output_path

# ═════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExportRequest:
    """Everything one export run needs; built by :func:`build_parser`."""

    model_type: str
    checkpoint: Path
    output_dir: Path
    dataset: Path
    classes_num: int
    eval_fold: int
    static: bool
    dynamic: bool
    fp16: bool
    per_channel: bool
    n_calibration: int
    n_holdout: int
    allow_degenerate: bool
    exp_name: str | None
    fold: int | None


def export_model(request: ExportRequest) -> tuple[Path, ...]:
    """Export one checkpoint to every requested format.

    Returns:
        The graphs written, in the order they were produced.

    Raises:
        DegenerateModelError: a quantized graph is a constant classifier and
            ``allow_degenerate`` is False.
    """
    print(f"\n{'=' * 72}")
    print(f"Exporting {request.model_type} from {request.checkpoint.name}")
    print(f"{'=' * 72}")

    config = config_snapshot(
        model_type=request.model_type, classes_num=request.classes_num
    )
    sed_model = load_sed_model(
        request.model_type, request.checkpoint, request.classes_num
    )
    export_wrapper = get_export_wrapper(request.model_type, sed_model).eval()

    dummy_input = reference_input(config)
    print(f"  input shape: {tuple(dummy_input.shape)}")
    with torch.no_grad():
        traced_output = export_wrapper(dummy_input)
    if traced_output.shape[-1] != request.classes_num:
        raise RuntimeError(
            f"wrapper produced {traced_output.shape[-1]} logits but "
            f"classes_num={request.classes_num}"
        )

    output_dir = request.output_dir / request.model_type
    provenance = CheckpointProvenance.from_path(
        request.checkpoint, request.exp_name, request.fold
    )
    print(
        f"  provenance: exp={provenance.exp_name} fold={provenance.fold} "
        f"epoch={provenance.epoch} acc={provenance.monitored_accuracy}"
    )

    fp32_path = export_fp32(export_wrapper, dummy_input, output_dir / "model_fp32.onnx")
    print(f"  wrote {fp32_path} ({fp32_path.stat().st_size / 1024:.1f} KiB)")
    written: list[Path] = [fp32_path]
    entries: list[dict[str, object]] = [
        manifest_entry(
            request.model_type, "fp32", fp32_path, provenance, ONNX_OPSET
        )
    ]

    # Both quantized paths need real segments: static for calibration, dynamic
    # for the constant-classifier guard.  One draw serves both.
    calibration: CalibrationSet | None = None
    fp32_report = None
    if request.static or request.dynamic:
        calibration = load_calibration_set(
            request.dataset,
            request.eval_fold,
            request.n_calibration,
            request.n_holdout,
        )
        print(f"  calibration: {calibration.describe()}")
        # The FP32 spread over the same hold-out is the baseline every quantized
        # spread is judged against, and it is recorded in the manifest so the
        # delta is auditable after the fact.
        fp32_report = constant_prediction_report(fp32_path, calibration.holdout)
        print(f"  FP32 hold-out spread: {fp32_report.describe()}")
        entries[0]["holdout_prediction_counts"] = list(fp32_report.counts)

    if request.dynamic and calibration is not None:
        path = quantize_int8_dynamic(fp32_path, output_dir / "model_int8.onnx")
        report = assert_not_constant(
            path, calibration.holdout, request.allow_degenerate, fp32_report
        )
        print(f"  wrote {path} - prediction spread {report.describe()}")
        written.append(path)
        entries.append(
            manifest_entry(
                request.model_type, "int8_dynamic", path, provenance, ONNX_OPSET,
                extra={"recipe": "quantize_dynamic(weight_type=QInt8)",
                 "holdout_prediction_counts": list(report.counts),
                 "fp32_holdout_prediction_counts": list(fp32_report.counts)},
            )
        )

    if request.static and calibration is not None:
        path = quantize_int8_static(
            fp32_path,
            output_dir / "model_int8_static.onnx",
            calibration,
            per_channel=request.per_channel,
        )
        report = assert_not_constant(
            path, calibration.holdout, request.allow_degenerate, fp32_report
        )
        print(f"  wrote {path} - prediction spread {report.describe()}")
        written.append(path)
        entries.append(
            manifest_entry(
                request.model_type, "int8_static", path, provenance, ONNX_OPSET,
                calibration,
                extra={
                    "recipe": (
                        "quantize_static(quant_format=QDQ, weight_type=QInt8, "
                        "activation_type=QInt8, calibrate_method=MinMax, "
                        f"per_channel={request.per_channel}, reduce_range=False)"
                    ),
                    "per_channel": request.per_channel,
                    "reproduces_published_graph": not request.per_channel,
                    "holdout_prediction_counts": list(report.counts),
                    "fp32_holdout_prediction_counts": list(fp32_report.counts)
                    if fp32_report is not None else None,
                },
            )
        )

    if request.fp16:
        path = convert_fp16(fp32_path, output_dir / "model_fp16.onnx")
        print(f"  wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")
        written.append(path)
        entries.append(
            manifest_entry(
                request.model_type, "fp16", path, provenance, ONNX_OPSET,
                extra={"recipe": "onnxconverter_common.float16.convert_float_to_float16"},
            )
        )

    manifest_path = write_manifest(output_dir, entries)
    print(f"  manifest {manifest_path}")
    return tuple(written)

# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_EXPORT_DIR: Final[Path] = REPO_ROOT / "exports"
DEFAULT_DATASET: Final[Path] = (
    REPO_ROOT / "data" / "circor" / "train-circor-data-30s.npy"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mirasol.deployment.export",
        description=(
            "Export a trained MIRASOL checkpoint to ONNX (FP32, INT8 dynamic, "
            "INT8 static, FP16). Static INT8 uses per-channel weight scales by "
            "default; pass --per-tensor to reproduce the paper's graphs."
        ),
    )
    parser.add_argument("--model-type", required=True, choices=MODEL_TYPES)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_EXPORT_DIR,
        help="graphs go to <output-dir>/<model-type>/ (default: exports/)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="prepared .npy used for INT8 calibration and the constant-"
             "classifier guard",
    )
    parser.add_argument("--classes-num", type=int, default=3, choices=(2, 3))
    parser.add_argument(
        "--eval-fold", type=int, default=0,
        help="held-out fold; calibration is drawn from the OTHER folds",
    )
    parser.add_argument("--dynamic", action="store_true",
                        help="also export weight-only INT8 (model_int8.onnx)")
    parser.add_argument("--static", action="store_true",
                        help="also export static INT8 (model_int8_static.onnx)")
    parser.add_argument("--fp16", action="store_true",
                        help="also export FP16 (model_fp16.onnx)")
    parser.add_argument(
        "--per-tensor", dest="per_channel", action="store_false", default=True,
        help="per-tensor INT8 weight scales, reproducing the paper's published "
             "static-INT8 graphs; three of four collapse to a constant class, "
             "so this needs --allow-degenerate",
    )
    parser.add_argument("--num-calib", dest="n_calibration", type=int,
                        default=DEFAULT_CALIBRATION_SEGMENTS,
                        help=f"calibration segments (default "
                             f"{DEFAULT_CALIBRATION_SEGMENTS}, the published recipe)")
    parser.add_argument("--num-holdout", dest="n_holdout", type=int,
                        default=DEFAULT_HOLDOUT_SEGMENTS,
                        help=f"held-out segments for the constant-classifier "
                             f"guard (default {DEFAULT_HOLDOUT_SEGMENTS})")
    parser.add_argument(
        "--allow-degenerate", action="store_true",
        help="keep a quantized graph whose argmax is constant over the hold-out "
             "set. Reproduction only - such a graph must never be deployed.",
    )
    parser.add_argument("--exp-name", help="override the parsed experiment name")
    parser.add_argument("--fold", type=int, help="override the parsed held-out fold")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.dynamic or args.static or args.fp16):
        print("Only FP32 will be exported; add --dynamic / --static / --fp16 "
              "for quantized graphs.")
    request = ExportRequest(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        dataset=args.dataset,
        classes_num=args.classes_num,
        eval_fold=args.eval_fold,
        static=args.static,
        dynamic=args.dynamic,
        fp16=args.fp16,
        per_channel=args.per_channel,
        n_calibration=args.n_calibration,
        n_holdout=args.n_holdout,
        allow_degenerate=args.allow_degenerate,
        exp_name=args.exp_name,
        fold=args.fold,
    )
    written = export_model(request)
    print(f"\nExported {len(written)} graph(s) for {args.model_type}:")
    for path in written:
        print(f"  {path} ({path.stat().st_size / (1024 * 1024):.2f} MiB)")
    if not request.per_channel and request.static:
        print(
            "\nNOTE: --per-tensor was used. model_int8_static.onnx is a "
            "reproduction artifact for Sec. 3.5(3), NOT a deployable model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
