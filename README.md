# A Low-Cost AI Stethoscope for Resource-Constrained Heart Murmur Screening

Code, models, hardware documentation and analysis scripts for:

> Dominik Ludera and Marta Kersten-Oertel. **A Low-Cost AI Stethoscope for
> Resource-Constrained Heart Murmur Screening.** MIRASOL Workshop, MICCAI 2026
> (Strasbourg). In *Medical Image Computing and Computer Assisted Intervention -
> MICCAI 2026 Satellite Events*, Lecture Notes in Computer Science, Springer.

DOI: _to be added once Springer assigns it._ The camera-ready is final and cites this
repository's URL.

> [!NOTE]
> **Release status.** The trained ONNX graphs are attached to the tagged GitHub release, not
> committed to the repository - see [Downloading the trained models](#downloading-the-trained-models).
> If an asset you need is not there yet, or something is missing, please open an issue.

> [!CAUTION]
> **This is a research prototype and not a medical device.** It has not been reviewed or
> approved by any regulatory authority for any clinical purpose. The models were evaluated
> on recordings made with a research-grade electronic stethoscope, whereas the prototype
> uses a low-cost USB microphone; a paired-recording study is a prerequisite to any clinical
> claim about the assembled device. Evaluation rests on a single, predominantly paediatric
> screening cohort, so the reported predictive values are conditional on its disease
> prevalence. **A negative screen does not exclude structural heart disease.**

**Contents:** [What this is](#what-this-is) - [Safety notice](#safety-notice-on-the-published-int8-models) -
[Quickstart](#quickstart-60-seconds-no-downloads) - [Reproducibility](#reproducibility-what-costs-what) -
[Layout](#repository-layout) - [Install](#installation) - [Data](#data) -
[Tables and figures](#reproducing-each-table-and-figure) - [Training](#training-from-scratch) -
[Models](#downloading-the-trained-models) - [Edge](#edge-deployment) -
[Known differences](#known-differences-between-the-paper-and-these-artifacts) -
[Licence](#licence-and-attribution) - [Citing](#citing-this-work) -
[Limitations](#limitations-and-clinical-status)

## What this is

Heart murmur screening on a ~$153 fully offline edge device. The study asks a deployment
question rather than an accuracy question: **when compute is fixed, which model should you
actually deploy?**

Six CNN and transformer architectures spanning an 18x parameter range are trained under one
protocol on the CirCor DigiScope phonocardiogram dataset, with patient-level five-fold
cross-validation, and then benchmarked on the target hardware - an Orange Pi Zero 2 W
(Allwinner H618, Cortex-A53 x4 @ 1.5 GHz, ~$36).

| Architecture | Parameters | Challenge Score (paper metric) | On-device FP32 latency |
|---|---|---|---|
| MobileNetV3-Small | 1.52 M | .886 +/- .008 | **67 ms** |
| FastViT | 3.26 M | .888 +/- .009 | 511 ms |
| EfficientNet-B0 | 4.01 M | .889 +/- .014 | 457 ms |
| TinyViT | 5.07 M | .894 +/- .015 | 1,436 ms |
| ResNet-18 | 11.17 M | **.896** +/- .011 | 697 ms |
| HTS-AT | 27.57 M | .875 +/- .020 | 1,393 ms |

Three findings:

1. **Diagnostic performance is largely independent of architecture.** Challenge Scores span
   0.875-0.896 across 1.5 M to 27.6 M parameters, with overlapping standard deviations. This
   holds under the official PhysioNet metric too, although the numbers differ - see
   [`docs/errata.md`](docs/errata.md) A1.
2. **Deployment efficiency is not.** The audio transformer is ~7x slower and ~25x larger for
   no diagnostic gain. Post-training INT8 costs 12-65% of patient-level accuracy for the
   architectures in the paper's Table 4, and quantization-aware training did not recover it -
   so on hardware without native INT8 acceleration, choosing a compact FP32 model beats
   compressing a larger one.
3. **Detecting *any* murmur is not the clinically actionable question.** Evaluated against
   the examining cardiologist's diagnosis - recorded in CirCor independently of the murmur
   annotation, and blind to the recordings - screening behaves as a high-precision triage
   test rather than a rule-out test: 12-17% of patients referred at a positive predictive
   value of 0.83-0.96, and 92-98% of non-innocent murmurs detected, but a sensitivity
   ceiling that binds expert auscultation equally.

## Safety notice on the published INT8 models

> [!WARNING]
> **Four of the five published static-INT8 graphs are degenerate - two of them exactly
> constant. Do not deploy them as screeners.**
>
> | Graph | Measured behaviour |
> |---|---|
> | `mobilenetv3/model_int8_static.onnx` | predicts *Absent* for all 739 fold-0 segments |
> | `efficientnet_b0/model_int8_static.onnx` | predicts *Absent* for 737 of 739 fold-0 segments |
> | `tinyvit/model_int8_static.onnx` | predicts *Absent* for all 739 fold-0 segments |
> | `fastvit/model_int8_static.onnx` | predicts *Unknown* for 710 of 739 fold-0 segments |
> | `resnet18/model_int8_static.onnx` | **not degenerate** - healthy prediction spread |
>
> Measured over all 739 segments and all 189 patients of fold 0. An always-*Absent* murmur
> screener never refers anyone, which in a screening context is a safety problem rather than
> a low score. The four degenerate graphs are published **only** to reproduce Sec. 3.5(3),
> in a separately named download, and this includes the EfficientNet-B0 INT8 build the
> paper's abstract describes.
>
> **The recommended deployable build is FP32**, which is Sec. 3.5's own conclusion. On the
> measured device that means MobileNetV3-Small FP32: 67 ms, 5.81 MB, non-degenerate.
>
> Details and the recipe-level explanation: [`docs/quantization.md`](docs/quantization.md)
> and [`docs/errata.md`](docs/errata.md) B1-B3.

## Quickstart, 60 seconds, no downloads

```bash
git clone https://github.com/domludera/mirasol2026.git
cd mirasol2026
pip install -e '.[analysis]'
python -m analysis.table1_fivefold
```

That reproduces Table 1 - all six architectures, all seven metric rows - from the
out-of-fold predictions committed in `results/predictions/`. No dataset, no GPU, no model
download, ~3 MB of CSV. It prints both the paper's Challenge Score and the official
PhysioNet formula side by side.

## Reproducibility: what costs what

Honest tiers. Most of the paper is in tier 1.

| Tier | You need | You can reproduce |
|---|---|---|
| **1** | a clean clone. No downloads, no GPU. ~3.2 MB of committed CSV and JSON | **Table 1** (7 rows x 6 architectures), **Table 4** plus the full six-architecture on-device table, **Table S7**, **Table S8**, **Table S10** including the class-weight sweep, and the quantization / FP16 / QAT logs |
| **1b** | tier 1 plus torch and timm (`.[train]`), still no data | parameter counts (Table 1 column headers) |
| **2** | **one 114 KB file** from PhysioNet - `training_data.csv`, *not* the audio | **Table 2**, **Tables S1-S6**, **Fig. S1**. These need CirCor's cardiologist `Outcome` and murmur-grading columns, which are not ours to redistribute |
| **3** | the full CirCor download (~450 MB zipped, ~560 MB unpacked) and a CUDA GPU for anything that trains | **Fig. 1** (needs one WAV, `9979_TV.wav`), dataset rebuild (1.66 GiB per configuration), retraining (30 fold-runs x 150 epochs), ONNX re-export, INT8 patient-level evaluation, QAT |
| **4** | an Orange Pi Zero 2 W | re-measuring Table 4 latency and memory. Without one, `results/benchmarks/*.json` is the record; `deployment/edge/benchmark.py` runs on any Linux box but produces numbers that are not comparable |

Table 3 is a literature comparison and is not regenerated by anything; Fig. S2 and Table S9
are the prototype photograph and bill of materials in `hardware/`.

Full command list: [`docs/reproduce.md`](docs/reproduce.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/mirasol/` | the installable package: config, paths, metrics, signal front end, dataset, six model definitions, training loop, ONNX export, CLI |
| `experiments/` | thin drivers that build argv and never edit tracked files: `run_experiment.py cv\|ablation\|sweep\|export`, plus `run_qat.py` |
| `deployment/` | ONNX export and quantization entry points, benchmark-input generator |
| `deployment/edge/` | **self-contained and torch-free** - `scp` this one directory to the device: benchmark, inference, vendored log-mel, `bench_input.npy`, model fetcher, pinned requirements |
| `analysis/` | one script per paper artifact; every table and figure in the paper and supplement |
| `results/predictions/` | 61 out-of-fold prediction dumps (~3.1 MB). The reason tier 1 exists. **ODC-By**, see below |
| `results/benchmarks/` | the two on-device benchmark JSONs behind Table 4 |
| `results/logs/` | the INT8, FP16 and QAT logs - the only evidence for the Sec. 3.5 claims |
| `results/checkpoint_accuracies.csv` | 30 `(model, fold, checkpoint, accuracy)` rows, replacing 26 GB of checkpoints |
| `data/README.md`, `data/splits/folds.csv` | how to get CirCor, and the 941-row patient-to-fold assignment. **ODC-By** |
| `hardware/` | bill of materials, platform configuration, prototype photograph |
| `docs/` | setup, data, splits, reproduce, edge setup, quantization, **errata** |
| `tests/` | metrics, split integrity, mel parity, edge-vendored-mel parity, golden tables |

No path anywhere in this repository is absolute; `mirasol.paths` resolves everything from the
package's own location, overridable by environment variable and by CLI flag.

## Installation

Python **3.12** is required. Pick the smallest profile that covers what you need:

```bash
pip install -e .                     # analysis: numpy, pandas, scikit-learn, scipy, matplotlib
pip install -e '.[train]'            # + torch, torchvision, timm, Lightning, torchlibrosa, librosa
pip install -e '.[export]'           # + onnx, onnxruntime, onnxconverter-common
pip install -e '.[train,export,dev]' # everything, plus pytest and ruff
```

The device installs separately and never needs torch:

```bash
pip install -r deployment/edge/requirements-edge.txt   # onnxruntime==1.24.2, numpy, soundfile, scipy
```

Every version is pinned to what the published results were produced with. The full verified
list, and the GPU used for training, are in [`docs/setup.md`](docs/setup.md).

## Data

The [CirCor DigiScope dataset](https://physionet.org/content/circor-heart-sound/) is **not
redistributed here**. No CirCor audio file and no CirCor metadata file is committed;
`.gitignore` blocks `data/**` and CI fails if one appears. Download it from PhysioNet under
the Open Data Commons Attribution License (ODC-By 1.0).

For the clinical-endpoint analysis you need **only `training_data.csv` (116,962 B)**, not the
whole dataset. Everything else in tier 1 needs nothing at all.

```bash
mkdir -p data/circor
curl -O https://physionet.org/files/circor-heart-sound/1.0.3/training_data.csv
mv training_data.csv data/circor/
```

Building the prepared datasets is a single command per configuration, but **note that the
preparation script's defaults are not the paper's configuration** - it defaults to 10 s clips
and a different output filename than the drivers expect. The exact commands are in
[`data/README.md`](data/README.md); the columns used, the label definitions and the
fold recipe are in [`docs/data.md`](docs/data.md) and [`docs/splits.md`](docs/splits.md).

Please cite CirCor if you use it - see [Citing this work](#citing-this-work).

## Reproducing each table and figure

An important property of this pipeline: **Challenge Score, Macro F1, Table 2 and Tables
S1-S8 are all computed post hoc** from `results/predictions/`, never during training.
Training writes logits and targets; everything else is derived. That is what makes tier 1
possible.

| Artifact | Command | Tier |
|---|---|---|
| Fig. 1 preprocessing pipeline | `python -m analysis.figures --figures preprocessing --audio-dir <dir>` | 3 |
| **Table 1** five-fold CV, six architectures | `python -m analysis.table1_fivefold` | **1** |
| Table 1 parameter headers | `python -m mirasol.cli.count_params` | 1b |
| **Table 2** clinical endpoints | `python -m analysis.table2_clinical` | 2 |
| Table 3 prior work | (literature table, not generated) | - |
| **Table 4** on-device benchmark | `python -m analysis.table4_ondevice` | **1** |
| Sec. 3.5 INT8 patient-level | `python -m analysis.patient_level_int8 --model <m> --format int8_static` | 3 |
| Tables S1-S6, Fig. S1 | `python -m analysis.supp_tables`, `python -m analysis.figures --figures clinical` | 2 |
| **Table S7** two-class task | `python -m analysis.supp_tables --tables s7` | **1** |
| **Table S8** checkpoint selection | `python -m analysis.checkpoint_selection` | **1** |
| Fig. S2, Table S9 prototype and BOM | [`hardware/README.md`](hardware/README.md) | - |
| **Table S10** ablation + class-weight sweep | `python -m analysis.table_s10_ablation [--sweep]` | **1** |
| Bibliography verification | `python -m analysis.verify_references <paper.tex>` | 1 + network |

Details, expected values and a troubleshooting list: [`docs/reproduce.md`](docs/reproduce.md).

## Training from scratch

One driver, four subcommands. The seven scripts that used to do this by rewriting a tracked
config file in place are retired; the migration table is in
[`experiments/README.md`](experiments/README.md).

```bash
python experiments/run_experiment.py cv --models mobilenetv3 fastvit efficientnet_b0 \
                                                 tinyvit resnet18 htsat --task 3class
python experiments/run_experiment.py cv --models efficientnet_b0 fastvit tinyvit htsat --task 2class
python experiments/run_experiment.py ablation --all
python experiments/run_experiment.py sweep --weights 2,5,1 4,10,1 6,15,1
python experiments/run_experiment.py export --models efficientnet_b0 --static
```

`--task` is a **complete** parameter set: `3class` implies `classes_num=3`,
`loss_weight=[4,10,1]` and `train-circor-data-30s.npy`; `2class` implies `classes_num=2`,
`loss_weight=[4,1]` and `train-circor-data-30s-binary.npy`. Nothing is inherited from a
previous invocation, and no driver writes to a version-controlled file. Each run emits
`runs/<exp>/predictions.csv` and `runs/<exp>/resolved_config.json`.

Two seeds: **42** for the patient split, **123777** for training. Runs are not bitwise
reproducible (`deterministic=False`), so compare five-fold means, not single folds.

Two disclosures the paper does not fully make, both in [`docs/errata.md`](docs/errata.md):
the best checkpoint is selected by accuracy on the fold that is also the test fold (Table S8
bounds the optimism at 0.38-1.44 pp), and **class-balanced resampling was active alongside
the `[4,10,1]` loss weights** in every reported run.

## Downloading the trained models

Weights ship as **GitHub Release assets**, not in the repository: 18 ONNX files,
316,643,361 B (301.9 MiB) in total, one of them 109.4 MiB, which is above GitHub's 100 MiB
hard blob limit.

```bash
cd deployment/edge
bash fetch_models.sh --group fp32           # the recommended deployable set
bash fetch_models.sh --model mobilenetv3    # one architecture
bash fetch_models.sh --group all
```

`fetch_models.sh` is POSIX `sh` using only `curl`, `sha256sum` and `tar`, verifies every file
against `SHA256SUMS.txt` from the same release, and refuses to continue on a mismatch.

| Asset | Size | Contents | Status |
|---|---|---|---|
| `mirasol-onnx-fp32-cnn.tar.gz` | 63.7 MiB | mobilenetv3 (5.81), efficientnet_b0 (15.30), resnet18 (42.61) | **recommended deployable set** |
| `mirasol-onnx-fp32-transformer.tar.gz` | 32.6 MiB | fastvit (12.51), tinyvit (20.10) | deployable, but slower for no gain |
| `htsat-model_fp32.onnx` | 109.4 MiB | HTS-AT FP32, loose file | **latency-benchmark graph only, not accuracy-equivalent** (0.618 vs 0.871 on fold 0) |
| `resnet18-model_int8_static.onnx` | 10.7 MiB | ResNet-18 static INT8 | the one static-INT8 build that survives quantization (0.8975 vs 0.9128 FP32) |
| `mirasol-onnx-int8-dynamic.tar.gz` | 44.8 MiB | efficientnet_b0 (4.19), fastvit (3.46), tinyvit (5.99), htsat (31.12) | evidence for Sec. 3.5(1); on-device ratios 3.32x / 3.00x / 1.42x / **0.78x** |
| `mirasol-onnx-fp16.tar.gz` | 24.7 MiB | efficientnet_b0 (7.72), fastvit (6.33), tinyvit (10.61) | evidence for Sec. 3.5(2); the TinyViT file does not load in any ONNX Runtime and ships as documented negative evidence |
| `mirasol-onnx-int8-static-DEGENERATE-do-not-deploy.tar.gz` | 16.1 MiB | efficientnet_b0 (4.33), fastvit (3.67), tinyvit (6.42), mobilenetv3 (1.65) | **constant classifiers. Reproduction artifacts for Sec. 3.5(3) only** |
| `export_manifest.json` | - | per-graph architecture, format, source checkpoint and accuracy, experiment, fold, tool versions, sha256, FP32 reference metric | provenance |
| `SHA256SUMS.txt` | - | checksums for every asset above | consumed by `fetch_models.sh` |

Sizes are MiB of the individual `.onnx` files. `bench_input.npy` is **not** a Release asset:
it is committed to the repository, because it is synthetic, 512,384 B, and the only input to
the Table 4 benchmark.

## Edge deployment

```bash
scp -r deployment/edge/ user@orangepi:~/mirasol-edge/
# on the device
cd ~/mirasol-edge
pip install -r requirements-edge.txt
bash fetch_models.sh --group fp32
python3 benchmark.py --models-dir models --input bench_input.npy --threads 4
python3 infer.py --model models/mobilenetv3/model_fp32.onnx --audio-dir patient_dir/ --patient
```

`infer.py` implements the decision rule the paper's numbers were produced with: repeat-tiled
30 s windows, mean-pooled logits across windows, and `--patient` pooling across auscultation
locations. Setup, governor, thermal notes and the `bench_input.npy` checksum are in
[`docs/edge_setup.md`](docs/edge_setup.md).

> [!IMPORTANT]
> **The device application layer is not part of this release.** There is no code here for
> USB-microphone capture, the OLED display, the push buttons or the PiJuice power HAT, and
> no CAD or STL files for the enclosure. What is released is the bill of materials, the
> platform configuration, and the model and benchmark pipeline. See
> [`hardware/README.md`](hardware/README.md).
>
> Also note that the models were trained on Littmann-quality recordings while the prototype
> uses a low-cost USB microphone, and no paired-recording study has been done.

## Known differences between the paper and these artifacts

Stated here rather than left to be discovered. Each links to the measured detail in
[`docs/errata.md`](docs/errata.md).

- The reported **Challenge Score is a weighted-credit variant**, not the official PhysioNet
  2022 formula. Under the official formula the spread across architectures widens from 2.1 pp
  to 5.6 pp and the leader changes from ResNet-18 to HTS-AT; the architecture-independence
  claim holds under both. Every script prints both. (A1)
- **FastViT FP32 is 12.5 MB, not 24.9 MB** - the benchmark harness double-counted an
  orphaned `.onnx.data` file. (C1)
- **MobileNetV3-S is in fact the fastest and smallest configuration measured** (67 ms FP32,
  59 ms INT8), and neither it nor ResNet-18 appears in Table 4. (C2)
- **ResNet-18 static INT8 does not collapse**, so the abstract's 12-65% range describes four
  architectures, not post-training INT8 in general. (B2)
- The **static-INT8 collapse is largely a per-tensor artifact**: with per-channel weights
  EfficientNet-B0 loses ~2% rather than 11.8%. The paper's *mechanism* claim about attention
  layers comes out better supported; its *magnitude* claim is recipe-dependent. (B3)
- The **HTS-AT ONNX is a latency-benchmark graph**, not accuracy-equivalent to the trained
  model. (C6)
- The **published QAT figure is a fake-quant measurement**: `tq.convert` failed, a bare
  `except` substituted the last fake-quant evaluation, and that run had diverged below the
  three-class chance floor. (E1)
- The **supplement's fold description omits a shuffle**, so the splits are not
  reconstructible from the prose. Use `data/splits/folds.csv`. (D1)
- The **Table 4 dynamic-INT8 footnote is a desktop measurement**; on device HTS-AT dynamic
  INT8 is *faster* than FP32. (C3)
- The **FP16 rows in the benchmark JSON are harness dtype failures**, not hardware evidence.
  (C4)
- **Peak RSS for TinyViT and HTS-AT-INT8 is a shared-process high-water mark.** (C5)
- Two different runs are called "**baseline**" in the same supplement section, and the
  **single-location ablation row was evaluated on a different set of held-out patients**.
  (F1, F2)
- **Class-balanced resampling was active** alongside the class-weighted loss and is not
  mentioned in the paper. (F3)

## Licence and attribution

Code in this repository: **MIT**, (c) 2026 Dominik Ludera and Marta Kersten-Oertel. See
[`LICENSE`](LICENSE). Four carve-outs, all detailed in
[`THIRD_PARTY.md`](THIRD_PARTY.md):

| Material | Licence |
|---|---|
| This repository's own code | MIT |
| Training core derived from [HTS-AT](https://github.com/RetroCirce/HTS-Audio-Transformer) | MIT, (c) 2022 Knut(Ke) Chen - original headers preserved verbatim in seven files |
| Windowed attention and layers from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | MIT, (c) Microsoft Corporation - notice preserved in two files |
| timm primitives vendored in `models/layers.py` | Apache-2.0, Copyright 2019 Ross Wightman - with the required notice and a change statement |
| `do_mixup` / `interpolate` from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) | MIT, (c) 2018-2020 Qiuqiang Kong |
| Every released **FastViT** ONNX graph | derives from `fastvit_t8.apple_in1k`, (C) 2023 **Apple Inc.**, under [Apple's own licence](https://github.com/apple/ml-fastvit) - **not** MIT |
| `results/predictions/` and `data/splits/` | **ODC-By 1.0** derived data, not MIT, with CirCor attribution |
| The CirCor DigiScope dataset | ODC-By 1.0, not redistributed here |

The HTS-AT headers, including `knutchen@ucsd.edu`, are **copyright notices** and are
retained deliberately. A PII or docstring cleanup pass that strips them breaks the licence;
CI asserts their exact bytes.

## Citing this work

If you use this code, the released models, or the committed predictions, cite the paper -
see [`CITATION.cff`](CITATION.cff), which GitHub renders as a "Cite this repository" button.

If you use the dataset - which you do, if you retrain anything - **you must also cite
CirCor**, as ODC-By requires:

> Oliveira, J., Renna, F., Costa, P.D., et al. The CirCor DigiScope dataset: from murmur
> detection to murmur classification. *IEEE Journal of Biomedical and Health Informatics*
> 26(6), 2524-2535 (2022). <https://doi.org/10.1109/JBHI.2021.3137048>

> Reyna, M.A., Kiarashi, Y., Elola, A., et al. Heart murmur detection from phonocardiogram
> recordings: the George B. Moody PhysioNet Challenge 2022. *PLOS Digital Health* 2(9),
> e0000324 (2023). <https://doi.org/10.1371/journal.pdig.0000324>

> Goldberger, A.L., Amaral, L.A.N., Glass, L., et al. PhysioBank, PhysioToolkit, and
> PhysioNet: components of a new research resource for complex physiologic signals.
> *Circulation* 101(23), e215-e220 (2000). <https://doi.org/10.1161/01.CIR.101.23.e215>

## Limitations and clinical status

This is a research prototype and **not a medical device**.

- **Acquisition mismatch.** Every diagnostic number was measured on CirCor recordings made
  with a research-grade Littmann electronic stethoscope. The prototype uses an $11 USB
  microphone module coupled to a stethoscope head. No paired-recording study has been
  performed, so the reported accuracy characterises the model under Littmann-quality input,
  not the assembled device.
- **One cohort, one prevalence.** A single public, predominantly paediatric screening cohort
  (664 children, 125 infants, 72 adolescents, 6 neonates, age unrecorded for 74) with 48.4%
  prevalence of abnormal cardiac status, reflecting a referral-enriched campaign. Positive
  and negative predictive values are prevalence-conditional even where sensitivity and
  specificity transfer.
- **A negative screen does not exclude structural heart disease.** Sensitivity for abnormal
  cardiac status is bounded near 0.33 *even for a human expert*, because 262 of 455 abnormal
  patients in this cohort have no audible murmur. This is a property of auscultation, not of
  the models. Programmes aiming at case detection need echocardiographic screening.
- **What a positive screen is good for.** High precision at low referral volume suits
  triaging access to a scarce specialist or echocardiography slot. It is not a rule-out test.
- **The *Unknown* class remains hard** for every architecture (best F1 = 0.185), reflecting
  68 patients in the whole cohort. The label marks recordings the annotator could not judge
  rather than a distinct clinical state, so on device it should trigger re-acquisition.
- **Ablations used one architecture and one fold**, and the quantization study used one fold
  throughout.
- **External validation across populations, prevalences and acquisition devices remains
  future work.**

## Issues and contributions

Please open an issue at <https://github.com/domludera/mirasol2026/issues>. Reports that an
attribution or licence notice is missing or wrong are handled ahead of anything else. If you
build the hardware and write the capture/display layer, a wiring description for
`hardware/README.md` would be very welcome.
