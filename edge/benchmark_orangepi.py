#!/usr/bin/env python3
"""
Ultra-lightweight ONNX benchmark for Orange Pi Zero 2W.

Dependencies: onnxruntime, numpy ONLY (no librosa/scipy needed).
Input mel-spectrogram is pre-computed and loaded from a .npy file.

Usage:
    python3 benchmark_orangepi.py --models_dir ./models --input bench_input.npy
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

CLASS_NAMES = ["Present", "Unknown", "Absent"]
WARMUP_RUNS = 3
BENCHMARK_RUNS = 20


def get_mem_mb():
    """Get current process RSS in MB from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except Exception:
        pass
    return 0.0


def benchmark_model(model_path, mel_input, num_warmup=WARMUP_RUNS, num_runs=BENCHMARK_RUNS):
    """Benchmark a single ONNX model."""
    import onnxruntime as ort

    model_name = os.path.basename(os.path.dirname(model_path))
    variant_name = os.path.splitext(os.path.basename(model_path))[0]

    # Model size (include .data file if present)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    data_file = model_path + ".data"
    if os.path.exists(data_file):
        model_size_mb += os.path.getsize(data_file) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  {model_name} / {variant_name}  ({model_size_mb:.1f} MB)")
    print(f"{'='*60}")

    gc.collect()
    rss_before = get_mem_mb()

    # Load model
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(model_path, sess_options=opts,
                                     providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  FAILED: {e}")
        return {"model": model_name, "variant": variant_name, "error": str(e)}

    load_time = time.perf_counter() - t0
    rss_loaded = get_mem_mb()
    inp_name = sess.get_inputs()[0].name
    # FP16 exports take a float16 input
    if sess.get_inputs()[0].type == "tensor(float16)":
        mel_input = mel_input.astype(np.float16)

    print(f"  Load: {load_time:.2f}s | RSS: {rss_before:.0f} -> {rss_loaded:.0f} MB")

    # Warmup
    for _ in range(num_warmup):
        try:
            sess.run(None, {inp_name: mel_input})
        except Exception as e:
            print(f"  Inference FAILED: {e}")
            return {"model": model_name, "variant": variant_name,
                    "model_size_mb": round(model_size_mb, 2),
                    "load_time_s": round(load_time, 3), "error": str(e)}

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = sess.run(None, {inp_name: mel_input})
        latencies.append((time.perf_counter() - t0) * 1000)

    rss_peak = get_mem_mb()
    lat = np.array(latencies)

    avg = float(np.mean(lat))
    std = float(np.std(lat))
    mn, mx = float(np.min(lat)), float(np.max(lat))
    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))
    p99 = float(np.percentile(lat, 99))
    tput = 1000.0 / avg

    # Prediction
    logits = out[0][0]
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    pred = int(np.argmax(probs))

    print(f"  Avg: {avg:.1f}ms  Std: {std:.1f}ms  Min: {mn:.1f}ms  Max: {mx:.1f}ms")
    print(f"  P50: {p50:.1f}ms  P95: {p95:.1f}ms  P99: {p99:.1f}ms")
    print(f"  Throughput: {tput:.2f} inf/s  |  Peak RSS: {rss_peak:.0f} MB")
    print(f"  Pred: {CLASS_NAMES[pred]} ({probs[pred]:.4f})")

    result = {
        "model": model_name, "variant": variant_name,
        "model_size_mb": round(model_size_mb, 2),
        "load_time_s": round(load_time, 3),
        "rss_before_mb": round(rss_before, 1),
        "rss_loaded_mb": round(rss_loaded, 1),
        "rss_peak_mb": round(rss_peak, 1),
        "latency_avg_ms": round(avg, 2),
        "latency_std_ms": round(std, 2),
        "latency_min_ms": round(mn, 2),
        "latency_max_ms": round(mx, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "throughput_ips": round(tput, 3),
        "prediction": CLASS_NAMES[pred],
        "confidence": round(float(probs[pred]), 4),
    }

    del sess
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="ONNX Benchmark - Orange Pi")
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--input", default="bench_input.npy",
                        help="Pre-computed mel .npy file")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    args = parser.parse_args()

    import platform
    import onnxruntime as ort

    print("=" * 60)
    print("ONNX Benchmark — Orange Pi Zero 2W")
    print("=" * 60)
    print(f"  Arch:   {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  ORT:    {ort.__version__}")
    print(f"  NumPy:  {np.__version__}")

    # Load pre-computed input
    mel = np.load(args.input)
    print(f"  Input:  {mel.shape} {mel.dtype}")

    # Discover models
    models = []
    for d in sorted(os.listdir(args.models_dir)):
        sd = os.path.join(args.models_dir, d)
        if not os.path.isdir(sd):
            continue
        for f in sorted(os.listdir(sd)):
            if f.endswith(".onnx"):
                models.append(os.path.join(sd, f))

    print(f"\n  Found {len(models)} models:")
    for m in models:
        print(f"    {m}")

    # Benchmark
    results = []
    for m in models:
        r = benchmark_model(m, mel, args.warmup, args.runs)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<16} {'Variant':<18} {'Size':>7} {'Load':>7} "
          f"{'Avg':>8} {'P95':>8} {'Inf/s':>7} {'RSS':>6}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<16} {r['variant']:<18} ERROR")
            continue
        print(f"{r['model']:<16} {r['variant']:<18} "
              f"{r['model_size_mb']:>6.1f}M {r['load_time_s']:>6.2f}s "
              f"{r['latency_avg_ms']:>7.1f}ms {r['latency_p95_ms']:>7.1f}ms "
              f"{r['throughput_ips']:>6.2f} {r['rss_peak_mb']:>5.0f}M")
    print(f"{'='*90}")

    # Save
    data = {
        "device": "Orange Pi Zero 2W",
        "cpu": "Cortex-A53 x4 @ 1.5GHz",
        "ram_gb": 4,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": args.runs, "warmup": args.warmup,
        "input_shape": list(mel.shape),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
