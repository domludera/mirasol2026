import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import esc_config as config
from export_onnx import waveform_to_mel_spectrogram
from src.metrics import challenge_score

MODELS = {
    # Path to INT8 ONNX models
    "HTS-AT": "exports/htsat/model_int8_static.onnx",
    "EfficientNet-B0": "exports/efficientnet_b0/model_int8_static.onnx",
    "FastViT": "exports/fastvit/model_int8_static.onnx",
    "TinyViT": "exports/tinyvit/model_int8_static.onnx",
}

TEST_FOLD = 0  # INT8 models are calibrated on folds 1-4 and evaluated on fold 0


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def load_test_data():
    dataset_path = os.path.join(
        config.dataset_path,
        getattr(config, "dataset_npy_name", "train-circor-data-30s.npy"),
    )
    full_dataset = np.load(dataset_path, allow_pickle=True)
    test_samples = list(full_dataset[TEST_FOLD])
    return test_samples

def get_label(sample) -> int:
    target = sample["target"]
    if isinstance(target, (int, float, np.integer)):
        return int(target)
    else:
        return int(np.argmax(target))

def main():
    test_samples = load_test_data()
    print(f"Loaded {len(test_samples)} test segments.")
    
    # Precompute mel specs
    mel_specs = []
    labels = []
    filenames = []
    for sample in test_samples:
        waveform = sample["waveform"].astype(np.float32)
        waveform = np.concatenate([waveform, waveform])
        mel = waveform_to_mel_spectrogram(waveform)
        mel_specs.append(mel)
        labels.append(get_label(sample))
        filenames.append(sample.get("name", "unknown_0"))
        
    print(f"Precomputed {len(mel_specs)} mel spectrograms.")
    
    results = {}
    
    for model_name, onnx_path in MODELS.items():
        if not os.path.exists(onnx_path):
            print(f"Missing {onnx_path}, trying non-static...")
            onnx_path = onnx_path.replace("_static", "")
            if not os.path.exists(onnx_path):
                print(f"Skipping {model_name}: {onnx_path} not found.")
                continue
                
        print(f"\nRunning INT8 inference for {model_name}...")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        
        all_probs = []
        for i, mel in enumerate(mel_specs):
            mel_batch = mel[np.newaxis, :]
            if input_type == 'tensor(float16)':
                mel_batch = mel_batch.astype(np.float16)
            
            outputs = session.run(None, {input_name: mel_batch})
            logits = outputs[0][0]
            prob = softmax(logits)
            all_probs.append(prob)
            
        all_probs = np.array(all_probs)
        
        # Build DataFrame
        df = pd.DataFrame({
            "prob_0": all_probs[:, 0],
            "prob_1": all_probs[:, 1],
            "prob_2": all_probs[:, 2],
            "target": labels,
            "filename": filenames
        })
        
        df["patient"] = df["filename"].apply(lambda x: x.split("_")[0])
        
        # Mean pool
        grouped = df.groupby("patient").agg({
            "prob_0": "mean",
            "prob_1": "mean",
            "prob_2": "mean",
            "target": "first",
        }).reset_index()
        
        pat_probs = grouped[["prob_0", "prob_1", "prob_2"]].values
        pat_preds = np.argmax(pat_probs, axis=1)
        pat_targets = grouped["target"].astype(int).values
        
        acc = accuracy_score(pat_targets, pat_preds)
        f1 = f1_score(pat_targets, pat_preds, average="macro", zero_division=0)
        c_score = challenge_score(pat_targets, pat_preds)
        
        results[model_name] = {
            "Accuracy": acc,
            "Macro F1": f1,
            "Challenge Score": c_score
        }
        
    print("\n\n" + "="*50)
    print("PATIENT-LEVEL INT8 RESULTS (Mean Pooling)")
    print("="*50)
    print(f"{'Model':<20} | {'Chall Score':<12} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | "
              f"{metrics['Challenge Score']:<12.4f} | "
              f"{metrics['Accuracy']:<10.4f} | "
              f"{metrics['Macro F1']:<10.4f}")

if __name__ == "__main__":
    main()
