# A Low-Cost AI Stethoscope for Resource-Constrained Heart Murmur Screening

Code, trained ONNX models, deployment scripts and hardware documentation for the
MIRASOL 2026 (MICCAI workshop) paper.

> **Dominik Ludera, Marta Kersten-Oertel** (Concordia University). *A Low-Cost AI Stethoscope
> for Resource-Constrained Heart Murmur Screening.* MIRASOL Workshop, MICCAI 2026, LNCS.

We benchmark six CNN and transformer architectures (1.5M–27.6M parameters) for
3-class murmur detection (Present / Unknown / Absent) on the
[CirCor DigiScope](https://physionet.org/content/circor-heart-sound/) phonocardiogram
dataset, under patient-level 5-fold cross-validation, and deploy them with ONNX Runtime
on an offline prototype with a ~$153 bill of materials, built around an Orange Pi Zero 2 W (4x Cortex-A53).

## Repository layout

```
main_moody.py        Train / test one configuration (PyTorch Lightning), settings in config/esc_config.py
export_onnx.py       Checkpoint -> ONNX (FP32, dynamic / static INT8, FP16); input is the mel-spectrogram
config/              esc_config.py — all hyperparameters (paper values are the defaults)
model/               MobileNetV3-S, FastViT, EfficientNet-B0, TinyViT, ResNet-18, HTS-AT (+ factory registry)
src/                 Data preparation, dataset, Lightning wrapper, PhysioNet challenge score
scripts/             Experiment runners: 5-fold CV, ablation, class-weight sweep, QAT, batch export
analysis/            Scripts that produce the paper's tables and figures
edge/                PyTorch-free inference + benchmark for the device
hardware/            Bill of materials, prototype photo, device setup
data/splits/         Patient-level fold assignment used for every experiment
predictions/         Segment-level test predictions behind Tables 2 and 5
exports/             On-device benchmark results + checksums of the released ONNX models
tests/               pytest suite
```

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Python 3.12
pip install -r requirements.txt
```

**Trained models.** Download `onnx-models.tar.gz` from the
[v1.0 release](https://github.com/domludera/mirasol2026/releases/tag/v1.0) and extract it in
the repository root (`tar xzf onnx-models.tar.gz`). It contains `exports/<model>/model_*.onnx`
for all six models and `exports/bench_input.npy`; checksums are in
[`exports/MANIFEST.txt`](exports/MANIFEST.txt).

**Data.** Download CirCor DigiScope from PhysioNet into `data/circor/`, so that
`data/circor/training_data/*.wav` and `data/circor/training_data.csv` exist.

## Reproducing the paper

### Tables from the released predictions (no data or GPU needed)

```bash
python analysis/paper_tables.py
```

Recomputes Table 2 (5-fold CV, all six models), Table 5 (preprocessing ablation) and the
class-weight sensitivity results (Sec. 3.5) from [`predictions/`](predictions/). Segment
logits are mean-pooled per patient and argmax'd; the Challenge Score is the PhysioNet
2022 weighted accuracy ([`src/metrics.py`](src/metrics.py)). Standard deviations are over
the five folds (population s.d.).

### Full pipeline

```bash
# 1. Dataset: 25-400 Hz Butterworth bandpass, z-score, 30 s windows, location-aware
#    restructuring, patient-level round-robin folds (seed 42)
#    -> data/circor/train-circor-data-30s.npy   (folds match data/splits/folds_3class.csv)
python -m src.prepare_data

# 2. Patient-level 5-fold CV for all six models (Table 2); skips finished folds
python scripts/run_5fold.py all
python analysis/paper_tables.py --preds_dir outputs/preds_targets

# 3. Baseline + ablations (Table 5) and class-weight sensitivity (Sec. 3.5), TinyViT, fold 0
python scripts/run_ablation.py
python scripts/run_class_weight_sensitivity.py

# 4. ONNX export (FP32 / static INT8 / FP16) of the fold-0 models + accuracy check against PyTorch
python scripts/export_all.py all --exp_suffix _3class_fold0 --static --fp16

# 5. Quantization (Sec. 3.4): patient-level INT8 accuracy, and QAT
python analysis/patient_level_int8.py
python scripts/run_qat.py

# Fig. 1 (preprocessing)
python analysis/fig1_preprocessing.py
```

Checkpoints go to `data/results/<experiment>/` and test predictions to
`outputs/preds_targets/<experiment>.data.csv`. Pass `--preds_dir outputs/preds_targets` to
`paper_tables.py` to score your own runs; missing experiments are skipped.

### Implementation notes

- **Training** (all architectures, [`config/esc_config.py`](config/esc_config.py)): AdamW
  (lr 5e-4 with warm-up, weight decay 0.02), batch 16, 150 epochs, class-weighted
  cross-entropy [4, 10, 1] (Present, Unknown, Absent), class-balanced resampling of the
  training set, random time shift and SpecAugment. The Mixup code path inherited from HTS-AT
  is only active for AudioSet and is not used. The timm backbones and EfficientNet-B0 start
  from ImageNet weights; HTS-AT is trained from scratch.
- **Data**: 941 patients / 3,120 recordings / 3,721 30-s instances after removing repeated
  recordings of the same location and one patient without usable audio;
  [`data/splits/folds_3class.csv`](data/splits/folds_3class.csv) lists the fold of every
  recording.
- **Checkpoints**: the 5-fold runs are tested on the checkpoint with the highest validation
  accuracy; the ablation runs and the released ONNX models use the most recent checkpoint
  of each run ([`exports/MANIFEST.txt`](exports/MANIFEST.txt) lists the exact files).
- **Quantization** (Sec. 3.4): ONNX models are exported from fold-0 runs, statically
  calibrated on folds 1-4 and evaluated on fold 0.
- **On-device inference** uses the same 30 s windowing as training (short recordings are
  repeated to fill the window) and mean-pools logits across windows.

| Paper | Code |
| --- | --- |
| Sec. 2.2, Fig. 1 — preprocessing | `src/prepare_data.py`, `analysis/fig1_preprocessing.py` |
| Sec. 2.3 — location-aware restructuring | `src/prepare_data.py`, `src/prepare_data_ablation.py --single_location` |
| Sec. 2.5 — patient-level folds | `src/prepare_data.py`, `data/splits/folds_3class.csv` |
| Table 1, Fig. 2 — hardware | [`hardware/`](hardware/) |
| Table 2 — 5-fold CV | `scripts/run_5fold.py`, `analysis/paper_tables.py` |
| Table 4 — on-device benchmark | `edge/benchmark_orangepi.py`, `exports/orangepi_benchmark.json` |
| Sec. 3.4 — quantization | `export_onnx.py`, `scripts/validate_quantized.py`, `analysis/patient_level_int8.py`, `scripts/run_qat.py` |
| Table 5, Sec. 3.5 — ablation, class weights | `scripts/run_ablation.py`, `scripts/run_class_weight_sensitivity.py`, `analysis/paper_tables.py` |

## On-device inference

The `edge/` scripts need only `edge/requirements.txt` (onnxruntime, numpy, librosa, scipy):

```bash
# Classify a recording (4 kHz PCG; resampled if needed)
python edge/inference_edge.py --model exports/efficientnet_b0/model_int8_static.onnx --audio recording.wav

# Latency / memory benchmark (paper Table 4: ONNX Runtime 1.24.2, 4 threads, 20 runs after 3 warm-ups)
python edge/benchmark_orangepi.py --models_dir exports --input exports/bench_input.npy
```

See [`hardware/README.md`](hardware/README.md) for the device and bill of materials.

## Tests

```bash
pytest                 # includes a forward pass for every model
pytest -m "not slow"
```

## Intended use

Research code. The models are trained on recordings from a single, predominantly pediatric
public dataset acquired with an electronic stethoscope and have not been validated on
recordings from the prototype's low-cost microphone. They are not a medical device and must
not be used for clinical decisions.

## License and acknowledgements

Code is released under the [MIT License](LICENSE). The CirCor DigiScope dataset is
distributed by PhysioNet under the ODC Attribution License; cite Oliveira et al. (2022) and
Reyna et al. (2023) when using it. The HTS-AT implementation is adapted from
[RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)
(Chen et al., ICASSP 2022, MIT License); other backbones come from
[timm](https://github.com/huggingface/pytorch-image-models).

## Citation

```bibtex
@inproceedings{ludera2026lowcost,
  title     = {A Low-Cost AI Stethoscope for Resource-Constrained Heart Murmur Screening},
  author    = {Ludera, Dominik and Kersten-Oertel, Marta},
  booktitle = {MIRASOL Workshop, Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```
