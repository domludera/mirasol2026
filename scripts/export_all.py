"""
Export trained checkpoints to ONNX and validate the quantized models.

Models are evaluated (and INT8-calibrated) on fold 0. For each model, picks the
newest checkpoint of experiment
"{model}{exp_suffix}", runs export_onnx.py, then validate_quantized.py
on every quantized artifact produced.

Usage:
    python scripts/export_all.py all                     # FP32 + dynamic INT8
    python scripts/export_all.py tinyvit --static --fp16 # FP32 + static INT8 + FP16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.factory import SUPPORTED_MODELS
from scripts.experiment_utils import PYTHON, REPO_ROOT, find_best_checkpoint, patch_config, run_cmd


def export_model(model_name: str, exp_name: str, static: bool, fp16: bool) -> None:
    print(f"\n{'=' * 50}\nONNX export & validation: {model_name}\n{'=' * 50}")
    patch_config(model_type=model_name, exp_name=exp_name, esc_fold=0)

    ckpt = find_best_checkpoint(exp_name, by="mtime")
    if not ckpt:
        print(f"  WARNING: no checkpoints found for {exp_name}")
        return
    print(f"  Checkpoint: {ckpt}")

    export_cmd = [PYTHON, "export_onnx.py", "--model_type", model_name, "--checkpoint", ckpt]
    if static:
        export_cmd.append("--static")
    if fp16:
        export_cmd.append("--fp16")
    if run_cmd(export_cmd, "Export") != 0:
        return

    artifacts = ["model_int8_static.onnx" if static else "model_int8.onnx"]
    if fp16:
        artifacts.append("model_fp16.onnx")

    for artifact in artifacts:
        onnx_path = os.path.join("exports", model_name, artifact)
        if not os.path.exists(os.path.join(REPO_ROOT, onnx_path)):
            print(f"  ERROR: {onnx_path} not found")
            continue
        run_cmd([PYTHON, "scripts/validate_quantized.py", "--model_type", model_name,
                 "--checkpoint", ckpt, "--onnx_model", onnx_path], f"Validate {artifact}")


def main():
    parser = argparse.ArgumentParser(description="Batch ONNX export + validation.")
    parser.add_argument("models", nargs="+", choices=SUPPORTED_MODELS + ("all",))
    parser.add_argument("--exp_suffix", default="_3class",
                        help="Experiment name suffix, e.g. _3class or _3class_fold0")
    parser.add_argument("--static", action="store_true", help="Static instead of dynamic INT8")
    parser.add_argument("--fp16", action="store_true", help="Also export and validate FP16")
    args = parser.parse_args()

    models = list(SUPPORTED_MODELS) if "all" in args.models else args.models
    for model_name in models:
        export_model(model_name, f"{model_name}{args.exp_suffix}", args.static, args.fp16)
    print("\nAll models processed.")


if __name__ == "__main__":
    main()
