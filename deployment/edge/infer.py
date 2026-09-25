#!/usr/bin/env python3
"""Run a released MIRASOL heart-murmur model on .wav recordings, on the device.

This is the "run it on your Pi" entry point.  It is deliberately torch-free and
librosa-free: it needs only ``onnxruntime``, ``numpy``, ``scipy`` and
``soundfile`` (see ``requirements-edge.txt``), all of which install from wheels
on aarch64.  The log-mel front end is the vendored ``mel.py`` next to this
file, so the input representation is bit-identical to the one the models were
trained and evaluated on.

Decision rule
-------------
The rule implemented here is the rule the paper's numbers were computed with:

* a recording is split into **non-overlapping 30 s windows**; a short tail (or
  a recording shorter than 30 s) is **repeat-tiled**, never zero-padded;
* the three-class logits of every window are **averaged**, and the prediction
  is the argmax of that average -- pooling happens on unnormalised logits,
  before any softmax, because that is what the patient-level aggregation
  behind Table 1 and Table 2 does;
* with ``--patient``, the average additionally spans every auscultation
  location of the same patient (CirCor's ``{patient_id}_{location}.wav``
  convention), which is the patient-level rule the paper reports.

Reading only the first 30 s, zero-padding short clips, or classifying each
window independently all give a different rule from the evaluated one, and
therefore numbers that no table in the paper covers.

Safety
------
NOT A MEDICAL DEVICE.  Research prototype only.  Use ``model_fp32.onnx``: the
static-INT8 graphs published for reproducing Sec. 3.5(3) include four constant
classifiers (three always answer "Absent", one always "Unknown") and must not
be used as screeners.  See the repository README and ``docs/quantization.md``.

Usage
-----
    python3 infer.py --model models/efficientnet_b0/model_fp32.onnx \
        --audio 9979_TV.wav

    python3 infer.py --model models/resnet18/model_fp32.onnx \
        --audio-dir recordings/ --patient --json results.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# `mel` is the vendored front end sitting next to this file, not a PyPI package;
# the whole point of deployment/edge/ is that it is self-contained.
import mel
import numpy as np
import onnxruntime as ort
import soundfile as sf

# Class order verified against mirasol.data.prepare: Present=0, Unknown=1,
# Absent=2.  Every released graph emits three logits in this order.
CLASS_NAMES: Final[tuple[str, str, str]] = ("Present", "Unknown", "Absent")

# Cortex-A53 on the Orange Pi Zero 2W has four cores.
DEFAULT_THREADS: Final[int] = 4

_ORT_TO_NUMPY: Final[dict[str, np.dtype]] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
}

_AUDIO_SUFFIXES: Final[tuple[str, ...]] = (".wav", ".flac", ".ogg")


@dataclass(frozen=True)
class RecordingResult:
    """Pooled prediction for one recording (one auscultation location)."""

    path: str
    patient_id: str
    n_windows: int
    pooled_logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    predicted_class: int
    latency_ms_per_window: tuple[float, ...]

    @property
    def predicted_name(self) -> str:
        return CLASS_NAMES[self.predicted_class]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.predicted_class]

    @property
    def total_latency_ms(self) -> float:
        return float(sum(self.latency_ms_per_window))


@dataclass(frozen=True)
class PatientResult:
    """Prediction pooled across every window of every location of a patient."""

    patient_id: str
    recordings: tuple[str, ...]
    n_windows: int
    pooled_logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    predicted_class: int

    @property
    def predicted_name(self) -> str:
        return CLASS_NAMES[self.predicted_class]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.predicted_class]


# ═════════════════════════════════════════════════════════════════════════════
# Session handling
# ═════════════════════════════════════════════════════════════════════════════

def open_session(model_path: Path, threads: int = DEFAULT_THREADS) -> ort.InferenceSession:
    """Open an ONNX Runtime CPU session and validate its interface.

    Raises:
        FileNotFoundError: the model file does not exist.
        ValueError: the graph's declared output width is not
            ``len(CLASS_NAMES)``.  Reporting a label for the wrong number of
            logits is exactly the failure this script used to hide.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}. Run ./fetch_models.sh --group fp32 "
            "to download the released graphs into models/."
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    declared = declared_num_classes(session)
    if declared is not None and declared != len(CLASS_NAMES):
        raise ValueError(
            f"{model_path} emits {declared} logits but this script knows "
            f"{len(CLASS_NAMES)} class names {CLASS_NAMES}. Refusing to label "
            "predictions against a mismatched head."
        )
    return session


def declared_num_classes(session: ort.InferenceSession) -> int | None:
    """Static width of the logits output, or None if the graph leaves it symbolic.

    The released HTS-AT graph leaves it symbolic (``Flattenlogits_dim_1``),
    so the check has to be repeated against the first real output as well.
    """
    shape = session.get_outputs()[0].shape
    if not shape:
        return None
    last = shape[-1]
    return int(last) if isinstance(last, int) else None


def session_input_dtype(session: ort.InferenceSession) -> np.dtype:
    """numpy dtype the graph declares for its input.

    The FP16 graphs reject float32 input with INVALID_ARGUMENT; casting per the
    declared type is why the FP16 rows in the shipped benchmark JSON are
    harness failures rather than hardware limits.
    """
    ort_type = session.get_inputs()[0].type
    try:
        return _ORT_TO_NUMPY[ort_type]
    except KeyError:
        raise ValueError(
            f"unsupported ONNX input type {ort_type!r}; expected one of "
            f"{sorted(_ORT_TO_NUMPY)}"
        ) from None


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits.astype(np.float64) - np.max(logits)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def run_windows(
    session: ort.InferenceSession, windows: Sequence[np.ndarray]
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Run every window through the graph.

    Args:
        session: an open session.
        windows: model inputs of shape ``(1, 1, n_frames, n_mels)``.

    Returns:
        ``(logits, latencies_ms)`` where ``logits`` is a float64 array of shape
        ``(len(windows), len(CLASS_NAMES))``.

    Raises:
        ValueError: the graph returned a width other than ``len(CLASS_NAMES)``.
    """
    input_name = session.get_inputs()[0].name
    dtype = session_input_dtype(session)
    logits: list[np.ndarray] = []
    latencies: list[float] = []
    for window in windows:
        batch = np.ascontiguousarray(window, dtype=dtype)
        started = time.perf_counter()
        outputs = session.run(None, {input_name: batch})
        latencies.append((time.perf_counter() - started) * 1000.0)
        row = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if row.size != len(CLASS_NAMES):
            raise ValueError(
                f"graph returned {row.size} logits but {len(CLASS_NAMES)} class "
                f"names are defined {CLASS_NAMES}"
            )
        logits.append(row)
    return np.stack(logits), tuple(latencies)


def load_waveform(path: Path) -> np.ndarray:
    """Decode a recording to mono float32 at the training sample rate.

    ``soundfile`` at the file's native rate is bit-identical to the
    ``librosa.load(sr=4000)`` used to build the training data for CirCor's
    16-bit 4 kHz WAVs, which is why no resampler is vendored here.

    Raises:
        ValueError: the file is not at ``mel.SAMPLE_RATE``.  Resampling is
            deliberately not done silently: a different resampler would change
            the input representation, and CirCor is natively 4 kHz.
    """
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != mel.SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sample_rate} Hz; the released models require "
            f"{mel.SAMPLE_RATE} Hz. Resample it first, e.g. "
            f"'sox in.wav -r {mel.SAMPLE_RATE} out.wav'."
        )
    return np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)


def patient_id_of(path: Path, require_location: bool) -> str:
    """CirCor patient identifier implied by a filename.

    CirCor names recordings ``{patient_id}_{location}.wav``; the paper's
    patient-level aggregation splits on the first underscore, so this does too.

    Raises:
        ValueError: ``require_location`` is set and the name has no underscore,
            so per-patient pooling would silently treat each file as its own
            patient.
    """
    stem = path.stem
    if "_" not in stem:
        if require_location:
            raise ValueError(
                f"--patient needs CirCor's '{{patient_id}}_{{location}}.wav' naming, "
                f"but {path.name} has no underscore. Rename the file or drop --patient."
            )
        return stem
    return stem.split("_", 1)[0]


def classify_recording(
    session: ort.InferenceSession, path: Path, require_location: bool
) -> RecordingResult:
    """Preprocess, run and mean-pool one recording."""
    waveform = load_waveform(path)
    windows = mel.preprocess_recording(waveform)
    logits, latencies = run_windows(session, windows)
    pooled = logits.mean(axis=0)
    probabilities = softmax(pooled)
    return RecordingResult(
        path=str(path),
        patient_id=patient_id_of(path, require_location),
        n_windows=len(windows),
        pooled_logits=tuple(float(v) for v in pooled),
        probabilities=tuple(float(v) for v in probabilities),
        predicted_class=int(np.argmax(pooled)),
        latency_ms_per_window=latencies,
    )


def pool_by_patient(results: Sequence[RecordingResult]) -> tuple[PatientResult, ...]:
    """Pool recording-level logits into patient-level predictions.

    The mean is weighted by each recording's window count, so the result is the
    flat mean over all of the patient's segments -- identical to the
    ``groupby(patient).mean()`` over segment logits used for Table 1/Table 2.
    """
    order: list[str] = []
    grouped: dict[str, list[RecordingResult]] = {}
    for result in results:
        if result.patient_id not in grouped:
            grouped[result.patient_id] = []
            order.append(result.patient_id)
        grouped[result.patient_id].append(result)

    pooled_results: list[PatientResult] = []
    for patient_id in order:
        group = grouped[patient_id]
        total_windows = sum(r.n_windows for r in group)
        summed = np.sum(
            [np.asarray(r.pooled_logits) * r.n_windows for r in group], axis=0
        )
        pooled = summed / total_windows
        probabilities = softmax(pooled)
        pooled_results.append(
            PatientResult(
                patient_id=patient_id,
                recordings=tuple(Path(r.path).name for r in group),
                n_windows=total_windows,
                pooled_logits=tuple(float(v) for v in pooled),
                probabilities=tuple(float(v) for v in probabilities),
                predicted_class=int(np.argmax(pooled)),
            )
        )
    return tuple(pooled_results)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def collect_audio_files(
    audio: Sequence[str], audio_dir: str | None
) -> tuple[Path, ...]:
    """Resolve the ``--audio`` / ``--audio-dir`` arguments into a sorted file list."""
    files: list[Path] = [Path(p) for p in audio]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
    if audio_dir is not None:
        directory = Path(audio_dir)
        if not directory.is_dir():
            raise NotADirectoryError(f"--audio-dir is not a directory: {directory}")
        files.extend(
            sorted(
                p for p in directory.iterdir()
                if p.suffix.lower() in _AUDIO_SUFFIXES
            )
        )
    if not files:
        raise FileNotFoundError(
            "no audio files to process; pass --audio FILE or --audio-dir DIR "
            f"containing {'/'.join(_AUDIO_SUFFIXES)} files"
        )
    return tuple(files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Heart-murmur screening inference on ARM Linux (Orange Pi Zero 2W). "
            "NOT A MEDICAL DEVICE."
        )
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="path to a released .onnx graph, e.g. models/resnet18/model_fp32.onnx",
    )
    parser.add_argument(
        "--audio", action="append", default=[],
        help="a single recording; repeat the flag for several files",
    )
    parser.add_argument("--audio-dir", help="directory of recordings to process")
    parser.add_argument(
        "--patient", action="store_true",
        help="also pool logits across a patient's auscultation locations "
             "(needs CirCor's {patient_id}_{location}.wav naming)",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help=f"intra-op threads (default {DEFAULT_THREADS}, one per Cortex-A53 core)",
    )
    parser.add_argument("--json", type=Path, help="write the full results as JSON")
    parser.add_argument(
        "--verbose", action="store_true", help="print per-class probabilities"
    )
    return parser


def print_recording_table(
    results: Sequence[RecordingResult], verbose: bool
) -> None:
    print(f"\n{'File':<34} {'Windows':>7} {'Prediction':<11} {'Conf':>7} {'Latency':>11}")
    print("-" * 74)
    for result in results:
        print(
            f"{Path(result.path).name:<34} {result.n_windows:>7} "
            f"{result.predicted_name:<11} {result.confidence:>6.1%} "
            f"{result.total_latency_ms:>8.1f} ms"
        )
        if verbose:
            for name, probability in zip(CLASS_NAMES, result.probabilities, strict=True):
                print(f"    {name:<10} p={probability:.4f}")


def print_patient_table(results: Sequence[PatientResult]) -> None:
    print(f"\n{'Patient':<12} {'Locations':>9} {'Windows':>7} {'Prediction':<11} {'Conf':>7}")
    print("-" * 52)
    for result in results:
        print(
            f"{result.patient_id:<12} {len(result.recordings):>9} "
            f"{result.n_windows:>7} {result.predicted_name:<11} "
            f"{result.confidence:>6.1%}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    files = collect_audio_files(args.audio, args.audio_dir)
    session = open_session(args.model, args.threads)
    input_meta = session.get_inputs()[0]
    print(f"Model:  {args.model}")
    print(f"Input:  {input_meta.name} {input_meta.shape} {input_meta.type}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Files:  {len(files)}")

    results = tuple(
        classify_recording(session, path, require_location=args.patient)
        for path in files
    )
    print_recording_table(results, args.verbose)

    patients: tuple[PatientResult, ...] = ()
    if args.patient:
        patients = pool_by_patient(results)
        print_patient_table(patients)

    latencies = [ms for r in results for ms in r.latency_ms_per_window]
    print(
        f"\nWindows: {len(latencies)}  |  per-window latency "
        f"avg {np.mean(latencies):.1f} ms, min {np.min(latencies):.1f} ms, "
        f"max {np.max(latencies):.1f} ms"
    )
    flagged = sum(1 for r in results if r.predicted_name == "Present")
    print(f"Recordings predicted Present: {flagged}/{len(results)}")
    print("Research prototype - NOT A MEDICAL DEVICE. Do not use for diagnosis.")

    if args.json is not None:
        payload = {
            "model": str(args.model),
            "class_names": list(CLASS_NAMES),
            "threads": args.threads,
            "onnxruntime_version": ort.__version__,
            "recordings": [vars(r) | {"predicted_name": r.predicted_name} for r in results],
            "patients": [vars(p) | {"predicted_name": p.predicted_name} for p in patients],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
