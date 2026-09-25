#!/usr/bin/env python3
"""
Validate INT8-quantized ONNX model accuracy against the FP32 PyTorch baseline.

Runs both models on the test dataset and compares:
  - Accuracy, Precision, Recall, F1
  - Per-sample prediction agreement
  - Latency comparison

Usage:
    python scripts/validate_quantized.py --model_type fastvit \
        --checkpoint data/results/fastvit_3class_fold0/.../<ckpt>.ckpt \
        --onnx_model exports/fastvit/model_int8_static.onnx
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import esc_config as config
from model.factory import SUPPORTED_MODELS, create_model
from src.metrics import challenge_score
from src.sed_model import SEDWrapper

# Reuse export wrappers and mel generation from export_onnx
from export_onnx import (
    get_export_wrapper,
    load_model,
    waveform_to_mel_spectrogram,
)


def load_test_data():
    """Load the test fold from the dataset."""
    dataset_path = os.path.join(
        config.dataset_path,
        getattr(config, "dataset_npy_name", "train-circor-data-30s.npy"),
    )
    full_dataset = np.load(dataset_path, allow_pickle=True)

    # Test fold = config.esc_fold (fold 0 by default)
    test_samples = list(full_dataset[config.esc_fold])
    print(f"Loaded {len(test_samples)} test samples from fold {config.esc_fold}")
    return test_samples


def get_label(sample) -> int:
    """Extract integer class label from a sample."""
    target = sample["target"]
    if isinstance(target, (int, float, np.integer)):
        return int(target)
    else:
        return int(np.argmax(target))


def precompute_mel_specs(test_samples: list) -> list:
    """Pre-compute all mel spectrograms once (shared by both models)."""
    print("\n  Pre-computing mel spectrograms...")
    mel_specs = []
    for i, sample in enumerate(test_samples):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(test_samples)}")
        waveform = sample["waveform"].astype(np.float32)
        waveform = np.concatenate([waveform, waveform])  # ESC_Dataset doubling
        mel = waveform_to_mel_spectrogram(waveform)
        mel_specs.append(mel)
    print(f"  Done — {len(mel_specs)} mel specs, shape: {mel_specs[0].shape}")
    return mel_specs


def run_pytorch_inference(model_type: str, checkpoint_path: str, mel_specs: list):
    """Run inference through the PyTorch model on pre-computed mel specs."""
    print("\n── PyTorch FP32 Inference ──")
    sed_model = load_model(model_type, checkpoint_path)
    export_model = get_export_wrapper(model_type, sed_model)
    export_model.eval()

    predictions = []
    logits_list = []
    latencies = []

    with torch.no_grad():
        for i, mel in enumerate(mel_specs):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(mel_specs)}")
            mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, 1, T, mel_bins)

            t0 = time.perf_counter()
            logits = export_model(mel_tensor)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            logits_np = logits.cpu().numpy()[0]
            logits_list.append(logits_np)
            predictions.append(int(np.argmax(logits_np)))

    avg_latency = np.mean(latencies)
    print(f"  Avg latency: {avg_latency:.1f} ms")
    return predictions, logits_list, avg_latency


def run_onnx_inference(onnx_path: str, mel_specs: list):
    """Run inference through the ONNX model on pre-computed mel specs."""
    import onnxruntime as ort

    print("\n── ONNX Inference ──")
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"❌ Failed to initialize ONNX runtime session: {e}")
        return [], [], 0.0

    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type

    predictions = []
    logits_list = []
    latencies = []

    for i, mel in enumerate(mel_specs):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(mel_specs)}")
        mel_batch = mel[np.newaxis, :]  # (1, 1, T, mel_bins)
        if input_type == 'tensor(float16)':
            mel_batch = mel_batch.astype(np.float16)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: mel_batch})
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        logits_np = outputs[0][0]
        logits_list.append(logits_np)
        predictions.append(int(np.argmax(logits_np)))

    avg_latency = np.mean(latencies)
    print(f"  Avg latency: {avg_latency:.1f} ms")
    return predictions, logits_list, avg_latency


def compute_metrics(predictions: list, labels: list):
    """Compute classification metrics for 3-class target."""
    from sklearn.metrics import accuracy_score, f1_score
    
    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')

    return {
        "accuracy": accuracy,
        "f1": f1,
        "challenge_score": challenge_score(labels, predictions),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate INT8 ONNX model vs FP32 PyTorch baseline"
    )
    parser.add_argument(
        "--model_type",
        required=True,
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the PyTorch .ckpt file",
    )
    parser.add_argument(
        "--onnx_model",
        required=True,
        help="Path to the INT8 ONNX .onnx file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of test samples (for quick validation)",
    )
    args = parser.parse_args()

    # ── Load test data ───────────────────────────────────────────────────
    test_samples = load_test_data()

    if args.max_samples and args.max_samples < len(test_samples):
        print(f"Limiting to {args.max_samples} samples (out of {len(test_samples)})")
        test_samples = test_samples[:args.max_samples]

    labels = [get_label(s) for s in test_samples]

    class_names = ["Present", "Unknown", "Absent"]
    print(f"Label distribution: {dict(zip(class_names, [labels.count(i) for i in range(len(class_names))]))}")

    # ── Pre-compute mel spectrograms (shared) ────────────────────────────
    mel_specs = precompute_mel_specs(test_samples)

    # ── Run PyTorch inference ────────────────────────────────────────────
    pt_preds, pt_logits, pt_latency = run_pytorch_inference(
        args.model_type, args.checkpoint, mel_specs
    )

    # ── Run ONNX inference ───────────────────────────────────────────────
    onnx_preds, onnx_logits, onnx_latency = run_onnx_inference(
        args.onnx_model, mel_specs
    )

    if not onnx_preds:
        print("⚠️ Validation aborted due to ONNX inference error.")
        return

    # ── Compute metrics ──────────────────────────────────────────────────
    pt_metrics = compute_metrics(pt_preds, labels)
    onnx_metrics = compute_metrics(onnx_preds, labels)

    # ── Agreement analysis ───────────────────────────────────────────────
    agreement = np.sum(np.array(pt_preds) == np.array(onnx_preds)) / len(pt_preds)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"VALIDATION REPORT — {args.model_type}")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'PyTorch FP32':>15} {'ONNX':>15} {'Delta':>10}")
    print("-" * 62)
    for metric_name in ["accuracy", "f1", "challenge_score"]:
        pt_val = pt_metrics[metric_name]
        onnx_val = onnx_metrics[metric_name]
        delta = onnx_val - pt_val
        sign = "+" if delta >= 0 else ""
        print(f"{metric_name.capitalize():<20} {pt_val:>14.4f} {onnx_val:>14.4f} {sign}{delta:>9.4f}")

    print(f"\n{'Metric':<20} {'PyTorch FP32':>15} {'ONNX':>15}")
    print("-" * 52)
    print(f"{'Avg Latency (ms)':<20} {pt_latency:>14.1f} {onnx_latency:>14.1f}")

    print(f"\nPrediction Agreement: {agreement:.1%} ({int(agreement * len(pt_preds))}/{len(pt_preds)})")

    # ── Flag disagreements ───────────────────────────────────────────────
    disagreements = []
    for i in range(len(pt_preds)):
        if pt_preds[i] != onnx_preds[i]:
            disagreements.append({
                "index": i,
                "name": test_samples[i].get("name", f"sample_{i}"),
                "label": class_names[labels[i]],
                "pytorch": class_names[pt_preds[i]],
                "onnx": class_names[onnx_preds[i]],
            })

    if disagreements:
        print(f"\n⚠️  {len(disagreements)} samples changed prediction after quantization:")
        print(f"  {'Index':<8} {'Name':<25} {'Label':<10} {'PyTorch':<10} {'ONNX':<10}")
        print(f"  {'-'*63}")
        for d in disagreements[:20]:  # Show at most 20
            print(f"  {d['index']:<8} {d['name']:<25} {d['label']:<10} {d['pytorch']:<10} {d['onnx']:<10}")
        if len(disagreements) > 20:
            print(f"  ... and {len(disagreements) - 20} more")
    else:
        print("\n✅ All predictions match between PyTorch and ONNX!")

    # ── Final verdict ────────────────────────────────────────────────────
    f1_delta = abs(onnx_metrics["f1"] - pt_metrics["f1"])
    if f1_delta <= 0.02:
        print(f"\n✅ PASS — F1 delta ({f1_delta:.4f}) is within 2% tolerance. Safe to deploy.")
    elif f1_delta <= 0.05:
        print(f"\n⚠️  WARNING — F1 delta ({f1_delta:.4f}) is between 2-5%. Review before deploying.")
    else:
        print(f"\n❌ FAIL — F1 delta ({f1_delta:.4f}) exceeds 5%. Consider re-calibrating or using FP16.")


if __name__ == "__main__":
    main()
