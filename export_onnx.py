#!/usr/bin/env python3
"""
Export trained PyTorch heart-sound models to ONNX and INT8-quantized ONNX.

Supports every model in model.factory.SUPPORTED_MODELS. The ONNX input is the
log-mel spectrogram (see waveform_to_mel_spectrogram), not the raw waveform.

Usage:
    # FP32 + dynamic INT8 (model_int8.onnx)
    python export_onnx.py --model_type tinyvit --checkpoint data/results/<exp>/.../<ckpt>.ckpt

    # FP32 + static INT8 (model_int8_static.onnx, calibrated on the training folds) + FP16
    python export_onnx.py --model_type tinyvit --checkpoint <ckpt> --static --fp16

    # Batch export of all fold-0 models, as in the paper: scripts/export_all.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import esc_config as config
from model.factory import SUPPORTED_MODELS, create_model
from src.sed_model import SEDWrapper

# ── Registry of model → checkpoint for --all mode ───────────────────────────
# e.g. "tinyvit": {"checkpoint": "data/results/tinyvit_3class_fold0/.../<ckpt>.ckpt"}
MODEL_REGISTRY = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Export Wrappers — isolate mel-spectrogram → logits for clean ONNX export
# ═══════════════════════════════════════════════════════════════════════════════

class TimmExportWrapper(nn.Module):
    """Export wrapper for timm-based models (FastViT, TinyViT, EfficientNet-B0).

    Input:  mel-spectrogram of shape (batch, 1, time_steps, mel_bins)
    Output: logits of shape (batch, num_classes)
    """

    def __init__(self, model):
        super().__init__()
        self.bn0 = model.bn0
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, mel_bins)
        x = x.transpose(1, 3)        # (B, mel_bins, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)        # (B, 1, T, mel_bins)
        x = x.transpose(2, 3)        # (B, 1, mel_bins, T) — vision layout
        logits = self.backbone(x)     # (B, num_classes)
        return logits


class EfficientNetExportWrapper(nn.Module):
    """Export wrapper for the torchvision EfficientNet-B0 model.

    Input:  mel-spectrogram of shape (batch, 1, time_steps, mel_bins)
    Output: logits of shape (batch, num_classes)
    """

    def __init__(self, model):
        super().__init__()
        self.bn0 = model.bn0
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.transpose(2, 3)        # (B, 1, mel_bins, T)
        logits = self.backbone(x)
        return logits


class HTSATExportWrapper(nn.Module):
    """Export wrapper for HTS-AT (Swin Transformer).

    Flattens the branching forward() into the simple path:
      mel-spec → reshape_wav2img → forward_features → clipwise_output

    Input:  mel-spectrogram of shape (batch, 1, time_steps, mel_bins)
    Output: logits of shape (batch, num_classes)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, mel_bins)
        x = x.transpose(1, 3)        # (B, mel_bins, T, 1)
        x = self.model.bn0(x)
        x = x.transpose(1, 3)        # (B, 1, T, mel_bins)
        # HTS-AT reshape_wav2img asserts T <= target_T (1024). We use 2001 timesteps.
        # Bypass the assert, interpolate, and then apply the exact spatial formatting
        # from htsat.py lines 745-750.
        target_T = int(self.model.spec_size * self.model.freq_ratio)  # 256 * 4 = 1024
        target_F = self.model.spec_size // self.model.freq_ratio      # 256 / 4 = 64
        
        # Interpolate (B, 1, T, mel_bins) -> (B, 1, 1024, 64)
        x = torch.nn.functional.interpolate(
            x, (target_T, target_F), mode="bicubic", align_corners=True
        )
        
        # Exact reshaping logic from `reshape_wav2img`
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, 1, F, T) -> (B, 1, 64, 1024)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.model.freq_ratio, x.shape[3] // self.model.freq_ratio) # (B, 1, 64, 4, 256)
        x = x.permute(0, 1, 3, 2, 4).contiguous() # (B, 1, 4, 64, 256)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4]) # (B, 1, 256, 256)

        output_dict = self.model.forward_features(x)
        return output_dict["clipwise_output"]


def get_export_wrapper(model_type: str, sed_model: nn.Module) -> nn.Module:
    """Return the appropriate export wrapper for the given model type."""
    if model_type == "htsat":
        return HTSATExportWrapper(sed_model)
    elif model_type == "efficientnet_b0":
        return EfficientNetExportWrapper(sed_model)
    elif model_type in ("fastvit", "tinyvit", "mobilenetv3", "resnet18"):
        return TimmExportWrapper(sed_model)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mel-spectrogram generation (numpy/librosa — same as will run on edge)
# ═══════════════════════════════════════════════════════════════════════════════

def waveform_to_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Convert a raw waveform to a log-mel spectrogram using the project config.

    This replicates the torchlibrosa pipeline in numpy so the same
    preprocessing can run on the Orange Pi without PyTorch.

    Args:
        waveform: 1-D float32 array of shape (clip_samples,)

    Returns:
        mel: float32 array of shape (1, time_steps, mel_bins)
    """
    import librosa

    # STFT
    stft = librosa.stft(
        waveform,
        n_fft=config.window_size,
        hop_length=config.hop_size,
        win_length=config.window_size,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    power = np.abs(stft) ** 2  # power spectrogram

    # Mel filter bank
    mel_basis = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.window_size,
        n_mels=config.mel_bins,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    mel = mel_basis @ power  # (mel_bins, time_steps)

    # Log scale (matching torchlibrosa's LogmelFilterBank defaults)
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))

    # Transpose to (1, time_steps, mel_bins) — channel-first with 1 channel
    log_mel = log_mel.T[np.newaxis, :, :]  # (1, T, mel_bins)

    return log_mel.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration data reader for INT8 quantization
# ═══════════════════════════════════════════════════════════════════════════════

from onnxruntime.quantization import CalibrationDataReader as OrtCalibrationDataReader

class CalibrationDataReader(OrtCalibrationDataReader):
    """Provides representative mel-spectrograms for INT8 calibration."""

    def __init__(self, num_samples: int = 50):
        dataset_path = os.path.join(
            config.dataset_path,
            getattr(config, "dataset_npy_name", "train-circor-data-30s.npy"),
        )
        print(f"  Loading calibration data from {dataset_path}")
        full_dataset = np.load(dataset_path, allow_pickle=True)

        # ESC_Dataset uses fold 0 for eval, rest for train — use train folds
        train_samples = []
        for i in range(len(full_dataset)):
            if i != config.esc_fold:
                train_samples.extend(full_dataset[i])

        # Subsample
        rng = np.random.RandomState(42)
        indices = rng.choice(len(train_samples), size=min(num_samples, len(train_samples)), replace=False)

        self.mel_specs = []
        for idx in indices:
            sample = train_samples[idx]
            waveform = sample["waveform"].astype(np.float32)
            # ESC_Dataset doubles the waveform
            waveform = np.concatenate([waveform, waveform])
            mel = waveform_to_mel_spectrogram(waveform)
            # Add batch dimension: (1, T, mel_bins) → (1, 1, T, mel_bins)
            self.mel_specs.append(mel[np.newaxis, :])

        print(f"  Prepared {len(self.mel_specs)} calibration samples, mel shape: {self.mel_specs[0].shape}")
        self.index = 0
        self.input_name = None  # Set after ONNX model is loaded

    def get_next(self):
        if self.index >= len(self.mel_specs):
            return None
        data = {self.input_name: self.mel_specs[self.index]}
        self.index += 1
        return data

    def rewind(self):
        self.index = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Core export logic
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_type: str, checkpoint_path: str) -> nn.Module:
    """Load a trained model from a checkpoint and return the inner sed_model."""
    # Temporarily override config.model_type
    original_model_type = config.model_type
    config.model_type = model_type

    sed_model = create_model(config)

    # Create a dummy SEDWrapper to load the checkpoint properly
    # We need the full state_dict structure (prefixed with 'sed_model.')
    wrapper = SEDWrapper(sed_model=sed_model, config=config, dataset=None)

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        wrapper.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as e:
        print(f"  Warning: Strict load failed ({e}), falling back to partial load")
        wrapper.load_state_dict(ckpt["state_dict"], strict=False)

    # Restore config
    config.model_type = original_model_type

    sed_model = wrapper.sed_model
    sed_model.eval()
    return sed_model


def export_single_model(model_type: str, checkpoint_path: str, output_dir: str, is_static: bool = False, is_fp16: bool = False, num_calib: int = 50):
    """Export one model to FP32 ONNX and optionally INT8 quantized ONNX or FP16 ONNX."""
    print(f"\n{'='*70}")
    print(f"Exporting: {model_type}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load model ───────────────────────────────────────────────
    sed_model = load_model(model_type, checkpoint_path)

    # ── Step 2: Create export wrapper ────────────────────────────────────
    export_model = get_export_wrapper(model_type, sed_model)
    export_model.eval()

    # ── Step 3: Determine dummy input shape ──────────────────────────────
    # Generate a mel spectrogram from a dummy waveform to get exact shape
    dummy_waveform = np.zeros(config.clip_samples * 2, dtype=np.float32)  # doubled like ESC_Dataset
    dummy_mel = waveform_to_mel_spectrogram(dummy_waveform)
    mel_shape = dummy_mel.shape  # (1, T, mel_bins)
    print(f"  Mel-spectrogram shape: {mel_shape}")

    dummy_input = torch.from_numpy(dummy_mel).unsqueeze(0)  # (1, 1, T, mel_bins)
    print(f"  Model input tensor shape: {dummy_input.shape}")

    # ── Step 4: Verify forward pass ──────────────────────────────────────
    with torch.no_grad():
        test_output = export_model(dummy_input)
    print(f"  Forward pass OK — output shape: {test_output.shape}")

    # ── Step 5: Export to ONNX ───────────────────────────────────────────
    fp32_path = os.path.join(output_dir, "model_fp32.onnx")
    print(f"  Exporting FP32 ONNX to {fp32_path}")

    torch.onnx.export(
        export_model,
        dummy_input,
        fp32_path,
        opset_version=17,
        dynamo=False,  # Use legacy TorchScript exporter for quantization compatibility
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    fp32_size = os.path.getsize(fp32_path)
    print(f"  FP32 ONNX size: {fp32_size / 1024:.1f} KB")

    # ── Step 6: INT8 Quantization (Dynamic or Static) ───────────────────
    if is_static:
        print("  Running INT8 static quantization...")
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_static

        int8_path = os.path.join(output_dir, "model_int8_static.onnx")
        
        # Instantiate CalibrationDataReader
        calib_reader = CalibrationDataReader(num_samples=num_calib)
        calib_reader.input_name = "mel_spectrogram"

        quantize_static(
            model_input=fp32_path,
            model_output=int8_path,
            calibration_data_reader=calib_reader,
            quant_format=ort.quantization.QuantFormat.QDQ, # Use QDQ format
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )

        int8_size = os.path.getsize(int8_path)
        compression = (1 - int8_size / fp32_size) * 100
        print(f"  INT8 Static ONNX size: {int8_size / 1024:.1f} KB ({compression:.1f}% smaller)")
    else:
        print("  Running INT8 dynamic quantization...")
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = os.path.join(output_dir, "model_int8.onnx")

        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
        )

        int8_size = os.path.getsize(int8_path)
        compression = (1 - int8_size / fp32_size) * 100
        print(f"  INT8 Dynamic ONNX size: {int8_size / 1024:.1f} KB ({compression:.1f}% smaller)")

    # ── Step 6.5: FP16 Export ───────────────────────────────────────────
    if is_fp16:
        print("  Running FP16 conversion...")
        from onnxconverter_common import float16
        import onnx
        
        fp16_path = os.path.join(output_dir, "model_fp16.onnx")
        model_onnx = onnx.load(fp32_path)
        model_fp16 = float16.convert_float_to_float16(model_onnx)
        onnx.save(model_fp16, fp16_path)
        
        fp16_size = os.path.getsize(fp16_path)
        compression = (1 - fp16_size / fp32_size) * 100
        print(f"  FP16 ONNX size: {fp16_size / 1024:.1f} KB ({compression:.1f}% smaller)")

    # ── Step 7: Smoke test the INT8 model ────────────────────────────────
    print("  Smoke-testing INT8 model...")
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input_name = sess_int8.get_inputs()[0].name

    # Use a real calibration sample if available
    test_mel = dummy_mel[np.newaxis, :]  # (1, 1, T, mel_bins)
    t0 = time.perf_counter()
    result = sess_int8.run(None, {input_name: test_mel})
    latency = (time.perf_counter() - t0) * 1000
    print(f"  INT8 inference OK — output: {result[0]}, latency: {latency:.1f} ms")

    print(f"\n  ✅ {model_type} export complete!")
    print(f"     FP32: {fp32_path} ({fp32_size / 1024:.1f} KB)")
    print(f"     INT8: {int8_path} ({int8_size / 1024:.1f} KB)")

    return fp32_path, int8_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export heart-sound models to INT8-quantized ONNX for edge deployment."
    )
    parser.add_argument(
        "--model_type",
        choices=SUPPORTED_MODELS,
        help="Model architecture to export",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the .ckpt file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exports",
        help="Base output directory (default: exports/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all models defined in MODEL_REGISTRY",
    )
    parser.add_argument(
        "--num_calib",
        type=int,
        default=50,
        help="Number of calibration samples for INT8 quantization (default: 50)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use INT8 Static Quantization instead of Dynamic Quantization",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Also export to FP16 ONNX format",
    )
    args = parser.parse_args()

    if args.all:
        if not MODEL_REGISTRY:
            print("ERROR: MODEL_REGISTRY is empty. Edit export_onnx.py to add checkpoint paths.")
            sys.exit(1)
        print(f"Exporting {len(MODEL_REGISTRY)} model(s): {', '.join(MODEL_REGISTRY.keys())}")
        for mtype, minfo in MODEL_REGISTRY.items():
            out_dir = os.path.join(args.output_dir, mtype)
            export_single_model(mtype, minfo["checkpoint"], out_dir, args.static, args.fp16, args.num_calib)
    elif args.model_type and args.checkpoint:
        out_dir = os.path.join(args.output_dir, args.model_type)
        export_single_model(args.model_type, args.checkpoint, out_dir, args.static, args.fp16, args.num_calib)
    else:
        parser.error("Provide --model_type and --checkpoint, or use --all")


if __name__ == "__main__":
    main()
