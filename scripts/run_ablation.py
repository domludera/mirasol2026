#!/usr/bin/env python3
"""Automated ablation study runner.

Runs the baseline and 5 ablation conditions for TinyViT (fold 0), each with:
  1) Dataset generation (via prepare_data_ablation.py; the baseline uses the
     standard dataset from src/prepare_data.py)
  2) Config patching (esc_config.py)
  3) Training (main_moody.py train)
  4) Testing (main_moody.py test)
  5) Evaluation (patient-level accuracy, macro F1, challenge score)

Results are collected into analysis/ablation_results.json and printed as tables
(analysis/paper_tables.py recomputes Table 5 from the prediction CSVs).

Usage:
    python scripts/run_ablation.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import (
    PREDS_DIR, PYTHON, REPO_ROOT, TASKS, Expr,
    find_best_checkpoint, patch_config, run_cmd,
)
from src.metrics import challenge_score

RESULTS_JSON = os.path.join("analysis", "ablation_results.json")

# ── Ablation conditions ──────────────────────────────────────────────────────

ABLATIONS = [
    # Baseline: 30 s windows, 25-400 Hz bandpass, multi-location (Table 5, first row)
    {
        "name": "baseline",
        "exp_name": "tinyvit_3class",
        "group": "Baseline",
        "description": "30 s / 25-400 Hz / multi-location",
        "dataset_args": None,
        "dataset_npy": TASKS["3class"]["dataset_npy_name"],
        "clip_seconds": 30,
    },
    # A1: Location restructuring — single-location
    {
        "name": "single_location",
        "group": "Location Restructuring",
        "description": "Single location per patient",
        "dataset_args": [
            "--clip_seconds", "30",
            "--single_location",
            "--output_name", "ablation-single-loc-30s.npy",
        ],
        "dataset_npy": "ablation-single-loc-30s.npy",
        "clip_seconds": 30,
    },
    # A2a: Window length — 10s
    {
        "name": "window_10s",
        "group": "Window Length",
        "description": "10-second clips",
        "dataset_args": [
            "--clip_seconds", "10",
            "--output_name", "ablation-10s.npy",
        ],
        "dataset_npy": "ablation-10s.npy",
        "clip_seconds": 10,
    },
    # A2b: Window length — 15s
    {
        "name": "window_15s",
        "group": "Window Length",
        "description": "15-second clips",
        "dataset_args": [
            "--clip_seconds", "15",
            "--output_name", "ablation-15s.npy",
        ],
        "dataset_npy": "ablation-15s.npy",
        "clip_seconds": 15,
    },
    # A3a: Bandpass — no filter
    {
        "name": "no_bandpass",
        "group": "Bandpass Filter",
        "description": "No bandpass filter",
        "dataset_args": [
            "--clip_seconds", "30",
            "--no_bandpass",
            "--output_name", "ablation-nofilter-30s.npy",
        ],
        "dataset_npy": "ablation-nofilter-30s.npy",
        "clip_seconds": 30,
    },
    # A3b: Bandpass — wide (25–1000 Hz)
    {
        "name": "wide_bandpass",
        "group": "Bandpass Filter",
        "description": "25–1000 Hz bandpass",
        "dataset_args": [
            "--clip_seconds", "30",
            "--bandpass_high", "1000",
            "--output_name", "ablation-wide-30s.npy",
        ],
        "dataset_npy": "ablation-wide-30s.npy",
        "clip_seconds": 30,
    },
]


def patch_ablation_config(exp_name, dataset_npy, clip_seconds):
    """Patch esc_config.py for a given ablation condition (TinyViT, 3-class)."""
    patch_config(
        model_type="tinyvit",
        exp_name=exp_name,
        esc_fold=0,
        **{**TASKS["3class"], "dataset_npy_name": dataset_npy},
        clip_seconds=clip_seconds,
        clip_samples=Expr("sample_rate * clip_seconds"),
        resume_checkpoint=None,
    )


def restore_config():
    """Restore config to baseline TinyViT 30s settings."""
    patch_ablation_config("tinyvit_3class", TASKS["3class"]["dataset_npy_name"], 30)


def evaluate_condition(exp_name):
    """Compute patient-level metrics from the test predictions of a condition.

    Returns dict with {accuracy, macro_f1, challenge_score} or None.
    """
    pred_path = os.path.join(REPO_ROOT, PREDS_DIR, f"{exp_name}.data.csv")
    if not os.path.exists(pred_path):
        print(f"  No predictions found at {pred_path}")
        return None

    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score

        df = pd.read_csv(pred_path, header=None,
                         names=["prob_0", "prob_1", "prob_2", "target", "filename"])

        # Patient-level aggregation
        df["patient"] = df["filename"].apply(lambda x: x.split("_")[0])
        patient_preds = (df.groupby("patient")[["prob_0", "prob_1", "prob_2", "target"]]
                         .mean().reset_index())

        preds = patient_preds[["prob_0", "prob_1", "prob_2"]].values
        y_pred = np.argmax(preds, axis=1)
        y_true = patient_preds["target"].astype(int).values

        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "challenge_score": round(challenge_score(y_true, y_pred), 4),
        }
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None


def main():
    print("=" * 70)
    print("ABLATION STUDY RUNNER — TinyViT")
    print("=" * 70)
    print(f"Conditions: {len(ABLATIONS)}")
    for a in ABLATIONS:
        print(f"  - {a['name']}: {a['description']}")
    print()

    results = []

    for i, abl in enumerate(ABLATIONS, 1):
        name = abl["name"]
        exp_name = abl.get("exp_name", f"ablation_{name}")

        print(f"\n{'='*70}")
        print(f"[{i}/{len(ABLATIONS)}] {abl['group']}: {abl['description']}")
        print(f"  Experiment: {exp_name}")
        print(f"{'='*70}")

        t_start = time.time()

        # Step 1: Generate dataset
        print("\n  Step 1: Generating dataset...")
        if abl["dataset_args"] is None:
            gen_cmd = [PYTHON, "-m", "src.prepare_data"]
        else:
            gen_cmd = [PYTHON, "-m", "src.prepare_data_ablation"] + abl["dataset_args"]
        rc = run_cmd(gen_cmd, "Data generation")
        if rc != 0:
            print(f"  ❌ Dataset generation failed, skipping {name}")
            results.append({"name": name, "group": abl["group"],
                            "description": abl["description"], "error": "data_gen_failed"})
            continue

        # Step 2: Patch config
        print("\n  Step 2: Patching config...")
        patch_ablation_config(exp_name, abl["dataset_npy"], abl["clip_seconds"])

        # Step 3: Train
        print("\n  Step 3: Training TinyViT...")
        rc = run_cmd([PYTHON, "main_moody.py", "train"], "Training")
        if rc != 0:
            print(f"  ❌ Training failed for {name}")
            results.append({"name": name, "group": abl["group"],
                            "description": abl["description"], "error": "train_failed"})
            restore_config()
            continue

        # Step 4: Test
        print("\n  Step 4: Testing...")
        # Paper runs used the most recent checkpoint (not the best-acc one)
        best_ckpt = find_best_checkpoint(exp_name, by="mtime")
        if best_ckpt:
            print(f"  Best checkpoint: {best_ckpt}")
            patch_config(resume_checkpoint=best_ckpt)
        else:
            print(f"  ⚠ No checkpoint found for {exp_name}")
            results.append({"name": name, "group": abl["group"],
                            "description": abl["description"], "error": "no_checkpoint"})
            restore_config()
            continue

        rc = run_cmd([PYTHON, "main_moody.py", "test"], "Testing")

        # Step 5: Evaluate
        print("\n  Step 5: Evaluating...")
        metrics = evaluate_condition(exp_name)

        elapsed = time.time() - t_start
        result = {
            "name": name,
            "group": abl["group"],
            "description": abl["description"],
            "dataset_npy": abl["dataset_npy"],
            "clip_seconds": abl["clip_seconds"],
            "elapsed_min": round(elapsed / 60, 1),
        }
        if metrics:
            result.update(metrics)
            print(f"\n  ✅ {name}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['macro_f1']:.4f}, "
                  f"Score={metrics['challenge_score']:.4f} "
                  f"({elapsed/60:.1f} min)")
        else:
            result["error"] = "eval_failed"
            print(f"\n  ⚠ {name}: Evaluation failed ({elapsed/60:.1f} min)")

        results.append(result)

    # Restore config
    restore_config()

    # Print summary
    print(f"\n\n{'='*80}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"{'Condition':<22} {'Description':<25} {'Acc':>7} {'F1':>7} {'Score':>7}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<22} {r['description']:<25} {'ERROR':>7}")
        else:
            print(f"{r['name']:<22} {r['description']:<25} "
                  f"{r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} "
                  f"{r['challenge_score']:>7.4f}")
    print(f"{'='*80}")

    # Save results
    out_path = RESULTS_JSON
    with open(os.path.join(REPO_ROOT, out_path), "w") as f:
        json.dump({
            "conditions": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
