"""Patient-level evaluation of the exported ONNX graphs: the Sec. 3.5(3) numbers.

Paper artifact
--------------
Sec. 3.5(3) of the paper: "EfficientNet-B0 INT8-static falls to Challenge Score
0.784 (from 0.889 FP32, -11.8%), TinyViT to 0.784 (from 0.894, -12.3%), and
FastViT degrades most, to 0.311 (from 0.888, -65.3%)", together with the
observation that two of those INT8 scores are identical to four decimal places
because the graphs have collapsed to a near-constant prediction.

This is the only script that reproduces those numbers. It runs the exported graphs
over one held-out fold, aggregates to the patient level with the same rule as
Table 1, and prints the segment-level prediction distribution beside the score --
which is what makes a constant classifier visible instead of leaving 0.784 to look
like retained accuracy. 0.784 is simply the weighted score of a graph that answers
Absent for every segment.

Three things the earlier version of this script got wrong are fixed here:

* it fell back from ``model_int8_static.onnx`` to ``model_int8.onnx`` when the
  static build was missing, so a dynamic-INT8 result could be reported under a
  static heading. The format is now an explicit argument and is printed with every
  row; a missing file is an error.
* it mean-pooled *softmax probabilities* while the FP32 table pools *logits*, so
  part of the reported drop was an aggregation change rather than quantization.
  Both poolings are now reported side by side, and the headline column is the
  logit pooling that Table 1 uses.
* it defaulted a missing segment name to ``unknown_0``, which would silently merge
  unrelated segments into one pseudo-patient. The name is now required.

An FP32 pass over the same fold is included by default so the delta is internally
consistent rather than compared against a five-fold mean from a different table.

Requirements
------------
The exported graphs (a GitHub Release asset, see ``docs/edge_setup.md``) and the
prepared dataset array built by ``mirasol.data.prepare``. Neither is committed.

Usage
-----
    python -m analysis.patient_level_int8 --formats fp32 int8_static
    python -m analysis.patient_level_int8 --models efficientnet_b0 --fold 0
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
import pandas as pd

from analysis._common import (
    ARCH_KEYS,
    CLASS_NAMES,
    MAX_FOLD_PATIENTS,
    MIN_FOLD_PATIENTS,
    OFFICIAL_METRIC,
    add_out_argument,
    pool_by_patient,
    print_header,
    resolve_out_dir,
    softmax,
    three_class_metrics,
)
from mirasol.paths import CIRCOR_AUDIO, REPO_ROOT
from mirasol.signal.mel import waveform_to_mel_spectrogram

#: Exported graph filename per format key.
FORMAT_FILES: Final[Mapping[str, str]] = {
    "fp32": "model_fp32.onnx",
    "fp16": "model_fp16.onnx",
    "int8": "model_int8.onnx",
    "int8_static": "model_int8_static.onnx",
}

#: How each format is described in the printed table.
FORMAT_LABELS: Final[Mapping[str, str]] = {
    "fp32": "FP32",
    "fp16": "FP16",
    "int8": "INT8 dynamic",
    "int8_static": "INT8 static",
}

#: ONNX Runtime type string -> numpy dtype for the input cast.
INPUT_DTYPES: Final[Mapping[str, type]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
}

DEFAULT_DATASET_NPY: Final[str] = "train-circor-data-30s.npy"


@dataclass(frozen=True)
class Segment:
    """One evaluation segment: its patient-scoped name, target and model input."""

    name: str
    target: int
    mel: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    """Patient-level result of one (architecture, format) pair."""

    model: str
    format_key: str
    n_segments: int
    n_patients: int
    segment_distribution: tuple[int, ...]
    metrics_logit_pooled: Mapping[str, float]
    metrics_softmax_pooled: Mapping[str, float]

    @property
    def format_label(self) -> str:
        return FORMAT_LABELS.get(self.format_key, self.format_key)

    @property
    def is_constant(self) -> bool:
        """True when every segment received the same class."""
        return sum(1 for count in self.segment_distribution if count) == 1


def load_fold_segments(dataset_npy: Path, fold: int) -> list[Segment]:
    """Load one fold of the prepared dataset and precompute its model inputs.

    Each stored 30 s waveform is concatenated with itself before the log-mel front
    end, exactly as the training data loader does, which is what yields the 2001
    frames the models expect.
    """
    if not dataset_npy.is_file():
        raise SystemExit(
            f"error: prepared dataset '{dataset_npy.name}' not found. Build it with "
            "'python -m mirasol.data.prepare --num-classes 3'; see data/README.md."
        )
    folds = np.load(dataset_npy, allow_pickle=True)
    if not 0 <= fold < len(folds):
        raise SystemExit(
            f"error: fold {fold} is out of range; '{dataset_npy.name}' holds "
            f"{len(folds)} folds."
        )
    segments: list[Segment] = []
    for index, sample in enumerate(folds[fold]):
        if "name" not in sample:
            raise SystemExit(
                f"error: sample {index} of fold {fold} has no 'name' field, so its "
                "patient cannot be identified. Rebuild the dataset with "
                "mirasol.data.prepare."
            )
        waveform = np.asarray(sample["waveform"], dtype=np.float32)
        target = sample["target"]
        segments.append(
            Segment(
                name=str(sample["name"]),
                target=int(target) if np.ndim(target) == 0 else int(np.argmax(target)),
                mel=waveform_to_mel_spectrogram(np.concatenate([waveform, waveform])),
            )
        )
    return segments


def graph_path(onnx_dir: Path, model: str, format_key: str) -> Path:
    """Locate one exported graph, with no substitution of a different format."""
    directory = onnx_dir / model
    if not directory.is_dir():
        raise SystemExit(
            f"error: no exported graphs for '{model}' under '{onnx_dir.name}/'. "
            "Fetch the release assets; see docs/edge_setup.md."
        )
    path = directory / FORMAT_FILES[format_key]
    if not path.is_file():
        available = sorted(p.name for p in directory.glob("*.onnx"))
        raise SystemExit(
            f"error: '{model}' has no {FORMAT_LABELS[format_key]} build "
            f"({FORMAT_FILES[format_key]}). Present: {available}. This script never "
            "substitutes a different format."
        )
    return path


def run_graph(path: Path, segments: Sequence[Segment]) -> np.ndarray:
    """Run one graph over every segment and return its raw logits."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    dtype = INPUT_DTYPES.get(spec.type)
    if dtype is None:
        raise SystemExit(
            f"error: '{path.name}' declares an unsupported input type {spec.type!r}."
        )
    output_dim = session.get_outputs()[0].shape[-1]
    if isinstance(output_dim, int) and output_dim != len(CLASS_NAMES):
        raise SystemExit(
            f"error: '{path.name}' emits {output_dim} logits but the three-class "
            f"convention has {len(CLASS_NAMES)}."
        )
    logits = np.empty((len(segments), len(CLASS_NAMES)), dtype=np.float64)
    for index, segment in enumerate(segments):
        batch = segment.mel[np.newaxis, :].astype(dtype)
        logits[index] = session.run(None, {spec.name: batch})[0][0]
    return logits


def evaluate(
    model: str, format_key: str, segments: Sequence[Segment], logits: np.ndarray
) -> Evaluation:
    """Aggregate segment logits to patients under both pooling rules."""
    frame = pd.DataFrame(
        {
            "logit_0": logits[:, 0],
            "logit_1": logits[:, 1],
            "logit_2": logits[:, 2],
            "target": [s.target for s in segments],
            "filename": [s.name for s in segments],
        }
    )
    frame["patient"] = frame["filename"].str.split("_").str[0]
    logit_pooled = pool_by_patient(frame)

    probabilities = softmax(logits)
    probability_frame = frame.assign(
        logit_0=probabilities[:, 0],
        logit_1=probabilities[:, 1],
        logit_2=probabilities[:, 2],
    )
    softmax_pooled = pool_by_patient(probability_frame)

    if not MIN_FOLD_PATIENTS <= len(logit_pooled) <= MAX_FOLD_PATIENTS:
        raise SystemExit(
            f"error: {model}/{format_key} covers {len(logit_pooled)} patients; a "
            f"single fold holds {MIN_FOLD_PATIENTS}-{MAX_FOLD_PATIENTS}."
        )
    segment_predictions = np.argmax(logits, axis=1)
    distribution = tuple(
        int(np.sum(segment_predictions == index)) for index in range(len(CLASS_NAMES))
    )
    return Evaluation(
        model=model,
        format_key=format_key,
        n_segments=len(segments),
        n_patients=len(logit_pooled),
        segment_distribution=distribution,
        metrics_logit_pooled=three_class_metrics(logit_pooled),
        metrics_softmax_pooled=three_class_metrics(softmax_pooled),
    )


def print_results(results: Sequence[Evaluation], fold: int) -> None:
    """One row per (architecture, format), with the segment distribution."""
    print()
    print_header(
        f"Patient-level results on held-out fold {fold} "
        f"({results[0].n_patients} patients, {results[0].n_segments} segments)"
    )
    header = (
        f"  {'Model':<16}{'Format':<14}{'Chall (logit)':>15}"
        f"{'Chall (softmax)':>17}{'Accuracy':>10}{'Macro F1':>10}"
        f"{'segments P/U/A':>18}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result in results:
        distribution = "/".join(str(c) for c in result.segment_distribution)
        flag = "  <- CONSTANT" if result.is_constant else ""
        print(
            f"  {result.model:<16}{result.format_label:<14}"
            f"{result.metrics_logit_pooled['Challenge Score']:>15.4f}"
            f"{result.metrics_softmax_pooled['Challenge Score']:>17.4f}"
            f"{result.metrics_logit_pooled['Accuracy']:>10.4f}"
            f"{result.metrics_logit_pooled['Macro F1']:>10.4f}"
            f"{distribution:>18}{flag}"
        )
    print(
        "\n  'segments P/U/A' is the segment-level argmax distribution. A graph that\n"
        "  answers Absent for every segment scores 0.784 under the project metric,\n"
        "  which is the prior of the Absent class and not retained accuracy."
    )


def print_deltas(results: Sequence[Evaluation]) -> None:
    """FP32 -> quantized deltas measured on the same fold and the same pooling."""
    baseline = {
        result.model: result for result in results if result.format_key == "fp32"
    }
    if not baseline:
        return
    print()
    print_header("Quantization delta against the FP32 graph on the same fold")
    for result in results:
        if result.format_key == "fp32" or result.model not in baseline:
            continue
        reference = baseline[result.model].metrics_logit_pooled["Challenge Score"]
        value = result.metrics_logit_pooled["Challenge Score"]
        relative = 100.0 * (value - reference) / reference if reference else float("nan")
        print(
            f"  {result.model:<16}{result.format_label:<14}"
            f"{reference:.4f} -> {value:.4f}  ({relative:+.1f}%)"
        )


def print_official(results: Sequence[Evaluation]) -> None:
    """The same rows under the official PhysioNet weighted accuracy."""
    print()
    print_header("Same rows under the official PhysioNet murmur weighted accuracy")
    for result in results:
        print(
            f"  {result.model:<16}{result.format_label:<14}"
            f"{result.metrics_logit_pooled[OFFICIAL_METRIC]:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.patient_level_int8",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(ARCH_KEYS.values()),
        default=["efficientnet_b0", "tinyvit", "fastvit"],
        metavar="ARCH",
        help="architectures to evaluate",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=sorted(FORMAT_FILES),
        default=["fp32", "int8_static"],
        metavar="FORMAT",
        help="exported formats to evaluate; never substituted for one another",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory of exported graphs, one subdirectory per architecture "
             "(default: exports/)",
    )
    parser.add_argument(
        "--dataset-npy",
        type=Path,
        default=None,
        metavar="NPY",
        help=f"prepared dataset array (default: the CirCor root's {DEFAULT_DATASET_NPY})",
    )
    parser.add_argument("--fold", type=int, default=0, help="held-out fold to evaluate")
    add_out_argument(parser, "build/patient_level_int8")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print to stdout only; do not create the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    onnx_dir = args.onnx_dir or (REPO_ROOT / "exports")
    dataset_npy = args.dataset_npy or (CIRCOR_AUDIO.parent / DEFAULT_DATASET_NPY)

    # Resolve every graph before spending time on the mel front end, so a missing
    # file fails immediately rather than after several minutes of preprocessing.
    graphs = {
        (model, fmt): graph_path(onnx_dir, model, fmt)
        for model in args.models
        for fmt in args.formats
    }
    segments = load_fold_segments(dataset_npy, args.fold)
    print(f"prepared {len(segments)} log-mel inputs from fold {args.fold}")

    results: list[Evaluation] = []
    for (model, fmt), path in graphs.items():
        print(f"running {model} {FORMAT_LABELS[fmt]} ({path.name})")
        results.append(evaluate(model, fmt, segments, run_graph(path, segments)))

    print_results(results, args.fold)
    print_deltas(results)
    print_official(results)

    if not args.no_write:
        out = resolve_out_dir(args)
        rows = [
            {
                "model": r.model,
                "format": r.format_key,
                "fold": args.fold,
                "n_segments": r.n_segments,
                "n_patients": r.n_patients,
                "segments_present": r.segment_distribution[0],
                "segments_unknown": r.segment_distribution[1],
                "segments_absent": r.segment_distribution[2],
                "constant_classifier": r.is_constant,
                **{f"logit_{k}": v for k, v in r.metrics_logit_pooled.items()},
                **{f"softmax_{k}": v for k, v in r.metrics_softmax_pooled.items()},
            }
            for r in results
        ]
        target = out / f"patient_level_fold{args.fold}.csv"
        pd.DataFrame(rows).to_csv(target, index=False)
        print(f"\nwrote {target.name} to {out.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
