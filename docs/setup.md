# Setup

Three installation profiles, in increasing order of cost. Pick the smallest one that
covers what you want to do; `docs/reproduce.md` says which profile each paper artifact
needs.

| Profile | Command | Downloads | What it gets you |
|---|---|---|---|
| Analysis (default) | `pip install -e .` | ~90 MB of wheels, no torch | Every table in the paper, from the committed predictions |
| Training | `pip install -e '.[train]'` | ~2.5 GB (CUDA wheels) | Dataset preparation and training |
| Export | `pip install -e '.[export]'` | ~250 MB | ONNX export, quantization, desktop validation |
| Device | `pip install -r deployment/edge/requirements-edge.txt` | ~50 MB, aarch64 | On-device benchmark and inference; see `docs/edge_setup.md` |

The extras compose: `pip install -e '.[train,export,dev]'` is a full development install.

## Python version

**Python 3.12 is required** (`requires-python = ">=3.12"`). The published results were
produced on CPython 3.12.3 on Linux x86-64.

```bash
python3 --version          # expect 3.12.x
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

`docs/splits.md` explains one place where the *interpreter* matters and not just the
version: the fold assignment depends on CPython's Mersenne Twister implementation of
`random.shuffle`. `data/splits/folds.csv` is shipped so that nothing downstream depends on
reproducing that.

## Analysis install (no torch, no GPU, no dataset)

```bash
git clone https://github.com/domludera/mirasol2026.git
cd mirasol2026
pip install -e .
python -m analysis.table1_fivefold          # reproduces Table 1
```

This is deliberately torch-free. The base dependency set is numpy, pandas, scikit-learn,
scipy and matplotlib, and it is enough to regenerate Table 1, Table 4, Tables S7, S8 and
S10, and the parameter reconciliation, because all of those are computed post hoc from
`results/predictions/` and `results/benchmarks/`. `pip install -e '.[analysis]'` is
accepted as an alias and installs exactly the same thing.

## Training install

```bash
pip install -e '.[train]'
```

Adds torch, torchvision, timm, PyTorch Lightning, torchlibrosa, librosa, soundfile, tqdm
and tensorboard. On a machine without CUDA, `pip` will resolve the CPU build of torch;
training then runs but is impractically slow for the full protocol. A CPU smoke test is
supported on purpose:

```bash
mirasol-train train --accelerator cpu --devices 1 --set max_epoch=1
# equivalently: python -m mirasol.cli.train train --accelerator cpu --devices 1 --set max_epoch=1
```

The original code hard-coded `accelerator='gpu'` and divided by
`torch.cuda.device_count()`, so it raised `ZeroDivisionError` on any machine without a GPU.
That is fixed: the device count floors at 1 and the accelerator defaults to `gpu` when
CUDA is available and `cpu` otherwise, with `--accelerator` and `--devices` overriding both.

## Export install

```bash
pip install -e '.[export]'
```

Adds onnx, onnxruntime and onnxconverter-common. Note the deliberate version split: the
desktop side pins `onnxruntime==1.24.1`, which is what the export and desktop validation
were run with, while `deployment/edge/requirements-edge.txt` pins `onnxruntime==1.24.2`,
which is the version in the Table 4 caption and the one the device measurements used.
`quantize_static` defaults have changed between ONNX Runtime releases, so both pins are
exact rather than lower bounds.

## The verified environment

Every pin in `pyproject.toml` is a version that was actually present when the published
results were produced, read from the installed distributions rather than from a
requirements file:

| Package | Version | Needed for |
|---|---|---|
| Python | 3.12.3 | everything |
| numpy | 1.26.4 | everything |
| scipy | 1.17.0 | bandpass filter, Wilson intervals |
| pandas | 3.0.1 | analysis |
| scikit-learn | 1.8.0 | analysis metrics |
| matplotlib | 3.10.8 | Fig. 1, Fig. S1 |
| torch | 2.10.0 (+cu128) | training, export |
| torchvision | 0.25.0 | EfficientNet-B0 backbone and its ImageNet weights |
| timm | 1.0.24 | FastViT, TinyViT, MobileNetV3-S, ResNet-18 backbones |
| pytorch-lightning | 2.6.1 | training loop |
| torchlibrosa | 0.1.0 | STFT / log-mel front end inside the models |
| librosa | 0.11.0 | WAV loading and resampling during dataset preparation |
| soundfile | 0.13.1 | WAV I/O |
| tqdm | 4.67.3 | progress output during dataset preparation |
| tensorboard | 2.19.0 | Lightning's default logger writes tfevents |
| onnx | 1.20.1 | export |
| onnxruntime | 1.24.1 desktop / 1.24.2 device | quantization, benchmarking, inference |
| onnxconverter-common | 1.16.0 | FP16 conversion |

Two notes on this list, because it differs from the `requirements.txt` that preceded it.

- The old file omitted torch, torchvision, timm, onnx, onnxruntime and
  onnxconverter-common entirely - that is, the dependencies that define four of the six
  architectures, EfficientNet-B0's weights and the whole export and benchmark subsystem -
  and pinned `pytorch_lightning==1.5.9` against code that uses the Lightning 2.x API. It
  could not install or run anything. It is replaced by `pyproject.toml`.
- Dropped as dead: `museval` (a music-source-separation package, imported once and never
  used, but a hard install requirement for training), `torchcontrib` (imported for two
  commented-out lines), and `grad-cam`, `gdown`, `notebook`, `ipywidgets` and `wget`, which
  belonged to scripts that are not part of this release. `torchaudio` 2.10.0 was present in
  the environment but the code's single `import torchaudio` was unused, so it is not a
  dependency. `h5py` was needed only by the AudioSet HDF5 code path, which is also not part
  of this release.

## Hardware used for the published training runs

All 30 fold-runs behind Table 1, plus the ablations and the class-weight sweep, were
trained on a single workstation GPU: **NVIDIA GeForce RTX 3090 Ti (24 GB)**, one visible
device (`CUDA_VISIBLE_DEVICES: [0]` in the training logs), CUDA 12.8 build of torch, batch
size 16, 150 epochs per run. Nothing in the protocol needs more than one GPU, and nothing
needs more than 24 GB; HTS-AT at batch size 16 is the largest configuration.

On-device numbers came from an Orange Pi Zero 2 W - see `docs/edge_setup.md` and
`hardware/README.md`.

## Verifying the install

```bash
pip install -e '.[dev]'
pytest                       # unit tests: metrics, splits, mel parity, golden tables
ruff check .
```

`tests/` runs without the dataset and without a GPU. `tests/test_golden_tables.py` freezes
the regenerated Table 1, Table 2 and Tables S1-S8 against the published values, so if an
install of a different dependency version changes a number, the test suite says so.

## Configuration and paths

No path in this repository is absolute. `mirasol.paths` resolves everything from the
location of the package itself:

| Name | Environment variable | Default |
|---|---|---|
| repository root | - | derived from `mirasol/paths.py`'s own location |
| predictions | `MIRASOL_PREDS_DIR` | `results/predictions/` |
| results | `MIRASOL_RESULTS_DIR` | `results/` |
| CirCor metadata | `CIRCOR_ROOT` / `CIRCOR_META` | `data/circor/training_data.csv` |
| CirCor audio | `CIRCOR_ROOT` / `CIRCOR_AUDIO` | `data/circor/training_data/` |

Every script also takes the corresponding `--preds-dir`, `--circor-meta`, `--audio-dir` and
`--out` flags, which win over the environment variables. If a CirCor path is needed and
missing, the error names the PhysioNet URL rather than printing a traceback.

Training parameters come from `mirasol.config.defaults` (a read-only defaults module),
overridden by an optional YAML file and then by repeatable `--set key=value` arguments. An
override key that does not already exist is an error at startup, so a typo fails
immediately instead of silently doing nothing. Every run writes
`runs/<exp_name>/resolved_config.json`, which is the provenance record for that run. No
script writes to a version-controlled file.
