"""
Quantization-Aware Training (QAT) for TinyViT and EfficientNet-B0.

Loads a pretrained FP32 checkpoint, inserts fake-quantization observers,
fine-tunes for a few epochs, then evaluates the quantized model at patient level.

Paper Sec. 3.4: QAT (10 epochs, lr 1e-5) does not recover the patient-level
accuracy lost to post-training INT8 quantization.

Usage:
    python scripts/run_qat.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.quantization as tq
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import esc_config as config
from model.factory import create_model
from scripts.experiment_utils import TASKS, find_best_checkpoint, patch_config
from src.metrics import challenge_score
from src.sed_model import SEDWrapper
from src.data_generator import ESC_Dataset


QAT_EPOCHS = 10
QAT_LR = 1e-5
FOLD = 0
BATCH_SIZE = 16


def patch_config_for_3class(fold):
    """Ensure config is set for 3-class, correct fold."""
    patch_config(esc_fold=fold, **TASKS["3class"])
    # Reload config
    import importlib
    from config import esc_config
    importlib.reload(esc_config)
    return esc_config


def find_checkpoint(model_name, fold):
    """Find best checkpoint for a given model and fold."""
    return find_best_checkpoint(f"{model_name}_3class_fold{fold}")


def load_datasets(cfg):
    """Load train and eval datasets for the configured fold."""
    npy_path = os.path.join(cfg.dataset_path,
                            getattr(cfg, "dataset_npy_name", "train-circor-data-30s.npy"))
    full_dataset = np.load(npy_path, allow_pickle=True)

    train_dataset = ESC_Dataset(dataset=full_dataset, config=cfg, eval_mode=False)
    eval_dataset = ESC_Dataset(dataset=full_dataset, config=cfg, eval_mode=True)
    return train_dataset, eval_dataset


def evaluate_patient_level(model, eval_loader, device):
    """Run inference and compute patient-level metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_names = []

    with torch.no_grad():
        for batch in eval_loader:
            waveform = batch["waveform"].to(device)
            output_dict = model.sed_model(waveform, None, infer_mode=True)
            pred = output_dict["clipwise_output"]
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())
            all_names.extend(batch["audio_name"])

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Patient-level aggregation
    df = pd.DataFrame({
        "prob_0": preds[:, 0],
        "prob_1": preds[:, 1],
        "prob_2": preds[:, 2],
        "target": targets,
        "filename": all_names,
    })
    df["patient"] = df["filename"].apply(lambda x: x.split("_")[0])

    grouped = df.groupby("patient").agg({
        "prob_0": "mean", "prob_1": "mean", "prob_2": "mean", "target": "first",
    }).reset_index()

    probs = grouped[["prob_0", "prob_1", "prob_2"]].values
    y_pred = np.argmax(probs, axis=1)
    y_true = grouped["target"].astype(int).values

    cs = challenge_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {"Challenge Score": cs, "Accuracy": acc, "Macro F1": f1, "n_patients": len(grouped)}


def run_qat_for_model(model_name):
    """Run QAT for a single model."""
    print(f"\n{'='*60}")
    print(f"QAT for {model_name}")
    print(f"{'='*60}")

    # Find checkpoint
    ckpt_path = find_checkpoint(model_name, FOLD)
    if not ckpt_path:
        print(f"  No checkpoint found for {model_name} fold {FOLD}")
        return None
    print(f"  Checkpoint: {ckpt_path}")

    # Patch config
    cfg = patch_config_for_3class(FOLD)

    # Override model_type
    cfg.model_type = model_name
    cfg.classes_num = 3
    cfg.save_preds = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    sed_model = create_model(cfg)
    model = SEDWrapper(sed_model=sed_model, config=cfg, dataset=None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)

    # Load data
    train_dataset, eval_dataset = load_datasets(cfg)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8)

    # --- Step 1: Evaluate FP32 baseline ---
    print("\n  [FP32 Baseline]")
    fp32_metrics = evaluate_patient_level(model, eval_loader, device)
    print(f"    Challenge Score: {fp32_metrics['Challenge Score']:.4f}")
    print(f"    Accuracy: {fp32_metrics['Accuracy']:.4f}")
    print(f"    Macro F1: {fp32_metrics['Macro F1']:.4f}")

    # --- Step 2: Prepare QAT ---
    # We use PyTorch's fake-quantization approach:
    # Insert FakeQuantize modules that simulate INT8 during training
    model.train()

    # Set qconfig for QAT (per-channel weight, per-tensor activation)
    model.sed_model.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Fuse common patterns if possible (conv-bn-relu)
    # For timm models, we skip fusion as the architecture varies
    # Just prepare QAT directly
    tq.prepare_qat(model.sed_model, inplace=True)

    # --- Step 3: QAT Fine-tuning ---
    print(f"\n  [QAT Fine-tuning] {QAT_EPOCHS} epochs, lr={QAT_LR}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=QAT_LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(cfg.loss_weight).to(device)
    )

    for epoch in range(QAT_EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch in train_loader:
            waveform = batch["waveform"].to(device)
            target = batch["target"].to(device).long()

            output_dict = model.sed_model(waveform)
            pred = output_dict["clipwise_output"]
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate every few epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            qat_metrics = evaluate_patient_level(model, eval_loader, device)
            print(f"    Epoch {epoch+1}/{QAT_EPOCHS}: loss={avg_loss:.4f} "
                  f"Score={qat_metrics['Challenge Score']:.4f} "
                  f"Acc={qat_metrics['Accuracy']:.4f} "
                  f"F1={qat_metrics['Macro F1']:.4f}")

    # --- Step 4: Convert to quantized model and evaluate ---
    print("\n  [Post-QAT Evaluation]")
    model.eval()
    model.sed_model = model.sed_model.cpu()

    # Convert fake-quant to actual quantized ops
    try:
        quantized_model_inner = tq.convert(model.sed_model, inplace=False)
        # Put wrapper back together for evaluation
        model.sed_model = quantized_model_inner
        model = model.cpu()

        # Evaluate on CPU (quantized models run on CPU)
        eval_loader_cpu = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=4)
        qat_final_metrics = evaluate_patient_level(model, eval_loader_cpu, torch.device("cpu"))
        qat_eval = "converted INT8"
    except Exception as e:
        print(f"    Convert failed ({e}), using fake-quant metrics instead")
        # Fall back to the last fake-quant evaluation (simulated INT8, not a converted model)
        qat_final_metrics = qat_metrics
        qat_eval = "fake-quant"

    print(f"    Evaluated as: {qat_eval}")
    print(f"    Challenge Score: {qat_final_metrics['Challenge Score']:.4f}")
    print(f"    Accuracy: {qat_final_metrics['Accuracy']:.4f}")
    print(f"    Macro F1: {qat_final_metrics['Macro F1']:.4f}")

    # --- Summary ---
    ptq_scores = {
        "tinyvit": 0.784,
        "efficientnet_b0": 0.784,
    }

    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {model_name}")
    print(f"  {'='*50}")
    print(f"  FP32 Challenge Score:     {fp32_metrics['Challenge Score']:.4f}")
    if model_name in ptq_scores:
        print(f"  PTQ INT8 Challenge Score: {ptq_scores[model_name]:.4f} "
              f"(delta: {ptq_scores[model_name] - fp32_metrics['Challenge Score']:+.4f})")
    print(f"  QAT INT8 Challenge Score: {qat_final_metrics['Challenge Score']:.4f} [{qat_eval}] "
          f"(delta: {qat_final_metrics['Challenge Score'] - fp32_metrics['Challenge Score']:+.4f})")
    if model_name in ptq_scores:
        recovery = qat_final_metrics['Challenge Score'] - ptq_scores[model_name]
        print(f"  QAT Recovery over PTQ:    {recovery:+.4f}")

    return {
        "model": model_name,
        "fp32": fp32_metrics,
        "qat": qat_final_metrics,
        "qat_eval": qat_eval,
    }


def main():
    results = []
    for model_name in ["tinyvit", "efficientnet_b0"]:
        try:
            result = run_qat_for_model(model_name)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {model_name} QAT failed: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print(f"\n\n{'='*60}")
        print("FINAL QAT RESULTS")
        print(f"{'='*60}")
        for r in results:
            m = r["model"]
            print(f"  {m}: FP32={r['fp32']['Challenge Score']:.4f} -> "
                  f"QAT={r['qat']['Challenge Score']:.4f} "
                  f"(delta: {r['qat']['Challenge Score'] - r['fp32']['Challenge Score']:+.4f}) "
                  f"[{r['qat_eval']}]")


if __name__ == "__main__":
    main()
