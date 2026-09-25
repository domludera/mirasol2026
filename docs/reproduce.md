# Reproducing every table and figure

One command per paper artifact, in paper order, each tagged with what it costs.

Read `docs/errata.md` alongside this page. Several scripts print a number that differs from
the printed paper, always for a reason that is documented there, and each script says so in
its own output.

## Tiers

| Tier | Requirement | Covers |
|---|---|---|
| **1** | a clean clone, `pip install -e .`, no downloads, no GPU (~3.2 MB of committed CSV/JSON) | Table 1, Table 4, Tables S7, S8, S10, the class-weight sweep, the quantization logs |
| **1b** | Tier 1 plus `pip install -e '.[train]'` - torch and timm, still no dataset, no GPU | parameter counts (Table 1 headers) |
| **2** | Tier 1 plus **one 114 KB file** from PhysioNet: `training_data.csv` | Table 2, Tables S1-S6, Fig. S1 |
| **3** | the full CirCor download (~450 MB zipped, ~560 MB unpacked), plus a CUDA GPU for anything that trains | Fig. 1, dataset rebuild, retraining, ONNX export, INT8 patient-level evaluation, QAT |
| **4** | an Orange Pi Zero 2 W | re-measuring Table 4 latency and memory |

Why Tier 2 exists at all: the cardiologist's `Outcome` and the murmur-grading columns are
CirCor's clinical labels. They are not redistributed here, so the clinical-endpoint tables
need that one CSV - but **not** the audio. See `data/README.md`.

**Everything post hoc.** Challenge Score, Macro F1, per-class metrics, Table 2 and Tables
S1-S8 are computed *after* training, from the out-of-fold prediction dumps in
`results/predictions/`. None of them is produced during training; training writes logits and
targets, nothing else. That is why Tier 1 exists, and it also means an analysis bug can be
fixed without retraining anything.

## Tier 1 - nothing to download

```bash
git clone https://github.com/domludera/mirasol2026.git
cd mirasol2026
pip install -e .
```

| Artifact | Command | Notes |
|---|---|---|
| **Table 1** patient-level five-fold CV, all 6 architectures, all 7 metric rows | `python -m analysis.table1_fivefold` | prints the paper's Challenge Score **and** the official PhysioNet score side by side (`docs/errata.md` A1). Asserts five folds and 941 patients, and exits non-zero on a missing fold file rather than printing a partial table |
| **Table 4** on-device benchmark | `python -m analysis.table4_ondevice` | regenerates the four-architecture table from `results/benchmarks/benchmark_2026-02_4arch.json` and additionally prints the full six-architecture table, the corrected FastViT FP32 size and the per-model dynamic-INT8 ratios (`docs/errata.md` C1-C3) |
| **Table S7** two-class configuration | `python -m analysis.supp_tables --tables s7` | Unknown merged into Present |
| **Table S8** checkpoint-selection sensitivity | `python -m analysis.checkpoint_selection` | reads `results/checkpoint_accuracies.csv` (30 rows) instead of globbing 26 GB of checkpoints (`docs/errata.md` F4) |
| **Table S10** preprocessing ablation | `python -m analysis.table_s10_ablation` | computes the baseline row rather than hard-coding it, and prints per-condition patient and segment counts (`docs/errata.md` F2) |
| **Class-weight sweep** (Sec. 3.6 running text) | `python -m analysis.table_s10_ablation --sweep` | `[2,5,1]` / `[4,10,1]` / `[6,15,1]` |
| **Sec. 3.5 evidence logs** | `less results/logs/int8_quantization.log` `results/logs/static_fp16.log` `results/logs/qat_results.log` | committed verbatim; read `docs/errata.md` E1 before quoting the QAT numbers |

Expected Tier-1 output for Table 1, both metrics, mean +/- s.d. over five folds:

| Model | Challenge Score (paper) | Challenge Score (official PhysioNet) |
|---|---|---|
| MobileNetV3-S | .886 +/- .008 | .690 +/- .030 |
| FastViT | .888 +/- .009 | .693 +/- .026 |
| EfficientNet-B0 | .889 +/- .014 | .684 +/- .042 |
| TinyViT | .894 +/- .015 | .705 +/- .041 |
| ResNet-18 | **.896** +/- .011 | .727 +/- .043 |
| HTS-AT | .875 +/- .020 | **.741** +/- .043 |

## Tier 1b - parameter counts

```bash
pip install -e '.[train]'
python -m mirasol.cli.count_params
```

Regenerates the parameter counts in Table 1's column headers (1.52M / 3.26M / 4.01M /
5.07M / 11.17M / 27.57M). This needs torch and timm, because the counts come from
instantiating the models, but it needs no dataset, no checkpoint and no GPU. The script
states its counting convention: trainable parameters for the single-channel, three-class
configuration, excluding the ~20.9k frozen torchlibrosa and `bn0` parameters.

## Tier 2 - one 114 KB CSV

Download only the metadata file from
<https://physionet.org/content/circor-heart-sound/> and put it where the resolver looks:

```bash
mkdir -p data/circor
cp /path/to/training_data.csv data/circor/training_data.csv     # 116,962 B
# or point at it instead:
export CIRCOR_META=/path/to/training_data.csv
```

| Artifact | Command |
|---|---|
| **Table 2** clinical endpoints, deployed rule | `python -m analysis.table2_clinical` |
| **Tables S1-S6** full interval estimates, per-fold detail, severity strata, sensitivity targets | `python -m analysis.supp_tables` |
| **Fig. S1** sensitivity vs referral burden, and sensitivity by severity | `python -m analysis.figures --figures clinical --out figures/` |
| Sec. 3.2 running text (referral %, PPV, NPV, NNR) | `python -m analysis.clinical_endpoint --report` |

All four join `results/predictions/` against `Outcome`, `Systolic murmur grading` and
`Diastolic murmur grading`. Endpoint prevalences, for a sanity check: E1 455/941 (48.4%),
E2 150/941 (15.9%), E3 50/941 (5.3%). Intervals are Wilson score 95% intervals throughout.

Every one of these scripts takes `--circor-meta` and `--preds-dir`, so nothing has to be
copied into the tree if you would rather not.

## Tier 3 - the full dataset, and a GPU for training

```bash
pip install -e '.[train,export]'
```

### Fig. 1, the preprocessing pipeline

Needs exactly **one** WAV file, `9979_TV.wav` (patient 9979, grade III/VI holosystolic
murmur):

```bash
python -m analysis.figures --figures preprocessing --audio-dir /path/to/training_data --out figures/
```

The two figure targets are separate subcommands on purpose: the previous version always
built the preprocessing figure first and died on the missing WAV, which put the
clinical-endpoint figure out of reach for anyone without the audio.

Note one substantive change: the figure now imports the bandpass from
`mirasol.signal.mel`, which applies the **causal** `sosfilt` the training pipeline uses. The
original figure script used zero-phase `sosfiltfilt`, so panels (b)-(d) did not show the
filter the models were actually trained with.

### Rebuilding the prepared datasets

See `data/README.md` for the exact commands and the `.npy` filenames the drivers expect
(the script defaults do not match them, which is a trap worth reading about before you
start). Note that each prepared file (1.66 GiB) is larger than the dataset it is built
from, because every recording is stored as an uncompressed float32 waveform at 4 kHz.

### Retraining

```bash
# one architecture, all five folds, 3-class protocol
python experiments/run_experiment.py cv --models efficientnet_b0 --task 3class

# the full Table 1 set: 6 architectures x 5 folds
python experiments/run_experiment.py cv \
    --models mobilenetv3 fastvit efficientnet_b0 tinyvit resnet18 htsat --task 3class

# Table S7
python experiments/run_experiment.py cv --models efficientnet_b0 fastvit tinyvit htsat --task 2class

# Table S10
python experiments/run_experiment.py ablation --all

# class-weight sweep
python experiments/run_experiment.py sweep --weights 2,5,1 4,10,1 6,15,1
```

Cost: 30 fold-runs x 150 epochs for Table 1. Each run writes
`runs/<exp_name>/predictions.csv` and `runs/<exp_name>/resolved_config.json`. `--overwrite`
is required to replace an existing predictions file; the driver refuses otherwise, and the
prediction writer opens in `w` mode and asserts its row and patient counts, so the old
append-mode bug that could silently double every row cannot recur.

Then re-run the Tier-1 analysis commands with `--preds-dir runs/` to score your own runs.

Two disclosures to keep in mind when comparing your numbers to Table 1: the best checkpoint
is selected by accuracy on the fold that is also the test fold (Table S8 bounds the
optimism at 0.38-1.44 pp), and class-balanced resampling was active alongside the
`[4,10,1]` loss weights (`docs/errata.md` F3). Runs are not bitwise reproducible; compare
five-fold means, not single folds (`docs/splits.md`).

### ONNX export and quantization

```bash
python -m deployment.export_onnx --model_type efficientnet_b0 --checkpoint <ckpt> --static
python -m deployment.validate_quantized --model_type efficientnet_b0 --patient-level
python -m analysis.patient_level_int8 --model efficientnet_b0 --format int8_static
```

`--per-tensor` reproduces the published graphs; the default is now per-channel. See
`docs/quantization.md`, and note that the exporter fails if the INT8 argmax is constant
across the calibration set.

### Quantization-aware training

```bash
python experiments/run_qat.py --model tinyvit --checkpoint <ckpt> --qbackend qnnpack
```

Read `docs/errata.md` E1 first. The published QAT figure was produced by a configuration
this script no longer permits.

## Tier 4 - the device

```bash
scp -r deployment/edge/ user@orangepi:~/mirasol-edge/
# on the device
pip install -r requirements-edge.txt
bash fetch_models.sh --group fp32
python3 benchmark.py --models-dir models --input bench_input.npy --threads 4
```

See `docs/edge_setup.md`. Without the board, `results/benchmarks/*.json` is the record;
`benchmark.py` runs on any Linux machine but produces numbers that are not comparable with
Table 4.

## Not regenerated by any script

- **Table 3** (comparison with prior work) is a literature table. Its Group A, B and C
  scores come from the cited publications and from Table 1 respectively; nothing computes
  it. `docs/errata.md` A1 explains why the cross-group comparison needs care - the groups
  differ in metric definition as well as in split.
- **Fig. S2** is a photograph of the prototype: `hardware/prototype.jpg`.
- **Table S9** is the bill of materials: `hardware/README.md`.

## Verifying the bibliography

```bash
python -m analysis.verify_references path/to/MIRASOL_023.tex
```

The LaTeX source is not part of this repository, so the path is a required argument.
Checks every bibliography entry against Crossref and arXiv: 35 entries, 0 unresolvable at
the time of the camera-ready. Anything not machine-checkable is listed explicitly rather
than silently passed. Needs network access, retries three times with backoff, and
distinguishes a network failure from a genuine 404 in its exit code.

## If a number does not match

In order of likelihood:

1. You are comparing the paper's Challenge Score against the official PhysioNet formula, or
   the reverse. Both are printed; `docs/errata.md` A1.
2. You retrained and are comparing a single fold. Runs are not bitwise reproducible; compare
   five-fold means (`docs/splits.md`).
3. You reconstructed the folds from the supplement's prose, which omits a shuffle. Use
   `data/splits/folds.csv` (`docs/errata.md` D1).
4. You are looking at a Table 4 row affected by a harness defect - FastViT FP32 size,
   TinyViT peak RSS, or any FP16 row (`docs/errata.md` C1, C4, C5).
5. Your dependency versions differ from `docs/setup.md`. `tests/test_golden_tables.py`
   catches this: run `pytest`.

If none of those explains it, please open an issue with the command you ran and the output.
