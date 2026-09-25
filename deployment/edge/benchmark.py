#!/usr/bin/env python3
"""On-device ONNX latency / memory benchmark -- regenerates Table 4's raw data.

Paper artifact: Table 4 (Orange Pi Zero 2W, Cortex-A53 x4 @ 1.5 GHz, 4 GB RAM,
ONNX Runtime 1.24.2, 4 intra-op threads, 3 warm-up + 20 timed runs).  The
committed record of the published run is ``results/benchmarks/*.json``; this
script is what produced it and what a reader re-runs on their own board.

Dependencies: ``onnxruntime`` and ``numpy`` only.  The input mel-spectrogram is
pre-computed (``bench_input.npy``, synthetic, sha256 recorded in
``docs/edge_setup.md``) so the benchmark never needs scipy or an audio decoder,
and so every board times the *same* tensor.

Differences from the harness that produced the published JSON, all of which
change reported values and are therefore listed here rather than hidden:

* input is cast to the dtype the graph declares.  Without this, every FP16 row
  fails with INVALID_ARGUMENT; the published FP16 rows are those harness
  failures, not hardware limits.
* peak memory is read from ``VmHWM`` (process high-water mark) and, by default,
  each model is timed in its own subprocess.  A single long-lived process
  reports a peak inherited from whichever model was loaded first, which is why
  the published TinyViT RSS (195 MB) is really HTS-AT's.
* the ``.onnx.data`` sibling is counted only when the graph actually declares
  external initializers.  Counting a stale orphan is what made Table 4 report
  FastViT FP32 as 24.9 MB instead of 12.5 MB.
* an environment block (ORT/numpy/Python versions, machine, thread count, CPU
  frequency, SoC temperature, input sha256) is serialised with the results.

Usage
-----
    python3 benchmark.py --models-dir models --input bench_input.npy \\
        --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

CLASS_NAMES: Final[tuple[str, str, str]] = ("Present", "Unknown", "Absent")

# Table 4's caption: 3 warm-up runs, 20 timed runs, 4 intra-op threads.
WARMUP_RUNS: Final[int] = 3
BENCHMARK_RUNS: Final[int] = 20
DEFAULT_THREADS: Final[int] = 4

_ORT_TO_NUMPY: Final[dict[str, str]] = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
}

_BYTES_PER_MIB: Final[float] = 1024.0 * 1024.0

# One-model-per-subprocess protocol: the child prints exactly this prefix
# followed by a single JSON object on stdout.
_RESULT_PREFIX: Final[str] = "@@BENCH_RESULT@@ "


@dataclass(frozen=True)
class BenchmarkResult:
    """One (architecture, variant) measurement, or the error that stopped it."""

    model: str
    variant: str
    model_size_mib: float
    external_data_mib: float
    load_time_s: float | None = None
    vm_hwm_mib: float | None = None
    latency_avg_ms: float | None = None
    latency_std_ms: float | None = None
    latency_min_ms: float | None = None
    latency_max_ms: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    throughput_ips: float | None = None
    input_dtype: str | None = None
    prediction: str | None = None
    confidence: float | None = None
    error: str | None = None


# ═════════════════════════════════════════════════════════════════════════════
# Host facts
# ═════════════════════════════════════════════════════════════════════════════

def _read_first_line(path: str) -> str | None:
    """Read one line from a sysfs/procfs file, or None if it is unavailable."""
    try:
        with open(path, encoding="ascii") as handle:
            return handle.readline().strip()
    except OSError:
        return None


def peak_rss_mib() -> float | None:
    """Process peak resident set size (``VmHWM``) in MiB.

    ``VmHWM`` is a high-water mark, so unlike ``VmRSS`` it is meaningful even
    after the allocator has released pages -- but it never decreases, which is
    why models are benchmarked one per process.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def soc_temperature_c() -> float | None:
    """SoC temperature in degrees Celsius, or None off Linux."""
    raw = _read_first_line("/sys/class/thermal/thermal_zone0/temp")
    if raw is None or not raw.lstrip("-").isdigit():
        return None
    return int(raw) / 1000.0


def cpu_frequency_khz() -> int | None:
    """Current CPU0 frequency in kHz, or None when cpufreq is unavailable."""
    raw = _read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return int(raw) if raw is not None and raw.isdigit() else None


def cpu_governor() -> str | None:
    """Current CPU0 scaling governor, or None when cpufreq is unavailable."""
    return _read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


def sha256_of_array(array: np.ndarray) -> str:
    """sha256 of the array's C-contiguous bytes, for input provenance."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_of_file(path: Path) -> str:
    """sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_block(
    mel_input: np.ndarray, input_path: Path, threads: int
) -> dict[str, object]:
    """Everything about the host and the input that changes the numbers.

    Both hashes are recorded: ``input_file_sha256`` is what
    ``docs/edge_setup.md`` publishes for ``bench_input.npy`` and what a
    reproducer checks with ``sha256sum``; ``input_array_sha256`` covers the
    decoded tensor, so a re-generated but equivalent .npy still compares equal.
    """
    import onnxruntime as ort

    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
        "intra_op_threads": threads,
        "cpu_governor": cpu_governor(),
        "cpu_freq_khz": cpu_frequency_khz(),
        "soc_temp_c": soc_temperature_c(),
        "input_path": str(input_path),
        "input_shape": list(mel_input.shape),
        "input_dtype": str(mel_input.dtype),
        "input_file_sha256": sha256_of_file(input_path),
        "input_array_sha256": sha256_of_array(mel_input),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Measurement
# ═════════════════════════════════════════════════════════════════════════════

def _proto_references(model_path: Path, needle: bytes) -> bool:
    """True if the serialised ONNX proto contains ``needle``.

    Scanned in 4 MiB chunks with an overlap, so a 109 MiB graph is not read
    into memory on a 4 GB board.
    """
    chunk_size = 4 * 1024 * 1024
    overlap = len(needle) - 1
    with model_path.open("rb") as handle:
        tail = b""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return False
            if needle in tail + chunk:
                return True
            tail = chunk[-overlap:] if overlap > 0 else b""


def external_data_bytes(model_path: Path) -> int:
    """Size of the ``.onnx.data`` sibling, but only if the graph references it.

    An orphaned ``.onnx.data`` left behind by an earlier export must not be
    added to the reported model size; that double-count is the origin of Table
    4's incorrect FastViT FP32 figure (12.5 MB reported as 24.9 MB).  A graph
    that really stores its initializers externally records the data file's name
    in its proto, so that name is what is looked for.
    """
    sibling = model_path.with_name(model_path.name + ".data")
    if not sibling.is_file():
        return 0
    if not _proto_references(model_path, sibling.name.encode("utf-8")):
        return 0
    return sibling.stat().st_size


def benchmark_model(
    model_path: Path,
    mel_input: np.ndarray,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCHMARK_RUNS,
    threads: int = DEFAULT_THREADS,
) -> BenchmarkResult:
    """Time one ONNX graph on one pre-computed input.

    Returns a :class:`BenchmarkResult`; a load or inference failure is recorded
    in its ``error`` field with the timing fields left as ``None``, never as a
    plausible-looking zero.
    """
    import onnxruntime as ort

    architecture = model_path.parent.name
    variant = model_path.stem
    external_mib = external_data_bytes(model_path) / _BYTES_PER_MIB
    size_mib = model_path.stat().st_size / _BYTES_PER_MIB + external_mib

    def failure(message: str) -> BenchmarkResult:
        return BenchmarkResult(
            model=architecture, variant=variant,
            model_size_mib=round(size_mib, 3),
            external_data_mib=round(external_mib, 3), error=message,
        )

    print(f"\n{'=' * 62}")
    print(f"  {architecture} / {variant}  ({size_mib:.2f} MiB)")
    print(f"{'=' * 62}")

    gc.collect()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads

    started = time.perf_counter()
    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as error:  # ORT raises many concrete types here.
        print(f"  LOAD FAILED: {error}")
        return failure(f"load: {error}")
    load_time_s = time.perf_counter() - started

    ort_input_type = session.get_inputs()[0].type
    if ort_input_type not in _ORT_TO_NUMPY:
        return failure(f"unsupported input type {ort_input_type}")
    dtype = np.dtype(_ORT_TO_NUMPY[ort_input_type])
    batch = np.ascontiguousarray(mel_input, dtype=dtype)
    input_name = session.get_inputs()[0].name
    print(f"  Load: {load_time_s:.2f}s | input {input_name} as {dtype}")

    try:
        for _ in range(warmup):
            session.run(None, {input_name: batch})
        latencies = []
        for _ in range(runs):
            started = time.perf_counter()
            outputs = session.run(None, {input_name: batch})
            latencies.append((time.perf_counter() - started) * 1000.0)
    except Exception as error:  # ORT raises many concrete types here.
        print(f"  INFERENCE FAILED: {error}")
        return failure(f"inference: {error}")

    samples = np.asarray(latencies, dtype=np.float64)
    average = float(samples.mean())
    logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
    shifted = np.exp(logits - logits.max())
    probabilities = shifted / shifted.sum()
    predicted = int(np.argmax(probabilities))
    label = (
        CLASS_NAMES[predicted] if predicted < len(CLASS_NAMES)
        else f"class_{predicted}"
    )

    peak_mib = peak_rss_mib()
    result = BenchmarkResult(
        model=architecture,
        variant=variant,
        model_size_mib=round(size_mib, 3),
        external_data_mib=round(external_mib, 3),
        load_time_s=round(load_time_s, 3),
        vm_hwm_mib=None if peak_mib is None else round(peak_mib, 1),
        latency_avg_ms=round(average, 2),
        latency_std_ms=round(float(samples.std()), 2),
        latency_min_ms=round(float(samples.min()), 2),
        latency_max_ms=round(float(samples.max()), 2),
        latency_p50_ms=round(float(np.percentile(samples, 50)), 2),
        latency_p95_ms=round(float(np.percentile(samples, 95)), 2),
        latency_p99_ms=round(float(np.percentile(samples, 99)), 2),
        throughput_ips=round(1000.0 / average, 3),
        input_dtype=str(dtype),
        prediction=label,
        confidence=round(float(probabilities[predicted]), 4),
    )
    print(
        f"  Avg {result.latency_avg_ms} ms  Std {result.latency_std_ms} ms  "
        f"P95 {result.latency_p95_ms} ms  {result.throughput_ips} inf/s"
    )
    print(f"  Peak VmHWM: {result.vm_hwm_mib} MiB | Pred: {label} "
          f"({result.confidence})")
    del session
    gc.collect()
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Model discovery and subprocess isolation
# ═════════════════════════════════════════════════════════════════════════════

def discover_models(models_dir: Path, only: Sequence[str]) -> tuple[Path, ...]:
    """Find ``<models_dir>/<architecture>/*.onnx``, sorted, optionally filtered.

    Raises:
        NotADirectoryError: ``models_dir`` does not exist.
        FileNotFoundError: no matching graphs were found.
    """
    if not models_dir.is_dir():
        raise NotADirectoryError(
            f"{models_dir} does not exist. Run ./fetch_models.sh --group fp32 first."
        )
    found = [
        path
        for architecture in sorted(p for p in models_dir.iterdir() if p.is_dir())
        for path in sorted(architecture.glob("*.onnx"))
        if not only or architecture.name in only
    ]
    if not found:
        wanted = f" for {', '.join(only)}" if only else ""
        raise FileNotFoundError(f"no .onnx graphs under {models_dir}{wanted}")
    return tuple(found)


def run_in_subprocess(
    model_path: Path, args: argparse.Namespace
) -> BenchmarkResult:
    """Benchmark one model in a fresh interpreter so ``VmHWM`` is its own.

    Raises:
        RuntimeError: the child produced no result line, which would otherwise
            silently drop a row from the table.
    """
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--single", str(model_path),
        "--input", str(args.input),
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--threads", str(args.threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    sys.stdout.write(
        "\n".join(
            line for line in completed.stdout.splitlines()
            if not line.startswith(_RESULT_PREFIX)
        )
        + "\n"
    )
    for line in completed.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return BenchmarkResult(**json.loads(line[len(_RESULT_PREFIX):]))
    raise RuntimeError(
        f"benchmark subprocess for {model_path} exited with "
        f"{completed.returncode} and produced no result:\n{completed.stderr}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ONNX latency/memory benchmark for the Orange Pi Zero 2W"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="directory of <architecture>/<variant>.onnx (default: models, "
             "the layout fetch_models.sh creates)",
    )
    parser.add_argument(
        "--input", type=Path, default=Path("bench_input.npy"),
        help="pre-computed mel-spectrogram .npy (default: bench_input.npy)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results.json"),
        help="where to write the results JSON",
    )
    parser.add_argument(
        "--model", action="append", default=[], metavar="ARCH",
        help="benchmark only this architecture directory; repeatable",
    )
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS,
                        help=f"timed runs per model (default {BENCHMARK_RUNS})")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS,
                        help=f"warm-up runs per model (default {WARMUP_RUNS})")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help=f"intra-op threads (default {DEFAULT_THREADS})")
    parser.add_argument(
        "--cooldown", type=float, default=0.0, metavar="SECONDS",
        help="sleep between models so the SoC does not thermally throttle "
             "into the next measurement (default 0)",
    )
    parser.add_argument(
        "--in-process", action="store_true",
        help="run every model in this process instead of one subprocess each; "
             "faster, but peak-memory figures become cumulative and wrong",
    )
    parser.add_argument(
        "--single", type=Path, help=argparse.SUPPRESS,
    )
    return parser


def load_input(path: Path) -> np.ndarray:
    """Load and validate the pre-computed mel-spectrogram tensor."""
    if not path.is_file():
        raise FileNotFoundError(
            f"benchmark input not found: {path}. It ships next to this script as "
            "bench_input.npy, or regenerate it with "
            "'python -m deployment.generate_bench_input'."
        )
    array = np.load(path)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 1:
        raise ValueError(
            f"{path} has shape {array.shape}; expected (1, 1, time_steps, mel_bins)"
        )
    return array


def print_summary(results: Sequence[BenchmarkResult]) -> None:
    print(f"\n\n{'=' * 92}")
    print("SUMMARY")
    print(f"{'=' * 92}")
    print(f"{'Model':<16} {'Variant':<19} {'Size':>9} {'Load':>7} {'Avg':>9} "
          f"{'P95':>9} {'Inf/s':>7} {'VmHWM':>8}")
    print("-" * 92)
    for result in results:
        if result.error is not None:
            print(f"{result.model:<16} {result.variant:<19} ERROR: {result.error[:44]}")
            continue
        print(
            f"{result.model:<16} {result.variant:<19} "
            f"{result.model_size_mib:>7.2f}Mi {result.load_time_s:>6.2f}s "
            f"{result.latency_avg_ms:>7.1f}ms {result.latency_p95_ms:>7.1f}ms "
            f"{result.throughput_ips:>7.2f} {result.vm_hwm_mib!s:>6}Mi"
        )
    print(f"{'=' * 92}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mel_input = load_input(args.input)

    # Child mode: one model, one JSON line on stdout, then exit.
    if args.single is not None:
        result = benchmark_model(
            args.single, mel_input, args.warmup, args.runs, args.threads
        )
        print(_RESULT_PREFIX + json.dumps(asdict(result)))
        return 0

    print("=" * 62)
    print("MIRASOL on-device ONNX benchmark")
    print("=" * 62)
    environment = environment_block(mel_input, args.input, args.threads)
    for key, value in environment.items():
        print(f"  {key}: {value}")

    models = discover_models(args.models_dir, args.model)
    print(f"\n  {len(models)} graph(s) to benchmark:")
    for path in models:
        print(f"    {path}")

    results: list[BenchmarkResult] = []
    for index, path in enumerate(models):
        if index and args.cooldown > 0.0:
            print(f"\n  cooling down for {args.cooldown:.1f} s ...")
            time.sleep(args.cooldown)
        if args.in_process:
            results.append(
                benchmark_model(path, mel_input, args.warmup, args.runs, args.threads)
            )
        else:
            results.append(run_in_subprocess(path, args))

    print_summary(results)

    payload = {
        "device": platform.node() or "unknown",
        "target_device": "Orange Pi Zero 2W (Cortex-A53 x4 @ 1.5 GHz, 4 GB)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": args.runs,
        "warmup": args.warmup,
        "isolation": "in-process" if args.in_process else "subprocess-per-model",
        "environment": environment,
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")
    return 1 if any(r.error is not None for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
