# Differences between the paper and the released artifacts

This page records every place where what is in this repository differs from, or is more
specific than, what the paper and supplement say. It exists so that a reader who
reproduces our numbers is never surprised, and so that anyone building on the artifacts
knows exactly what they are holding.

**How to read this page.** Each entry states what the paper says, what the artifacts do,
and what was measured. Every number below was recomputed from the files in this
repository, and each entry names the command that regenerates it. Where something could
not be measured, that is said explicitly rather than glossed. The paper is the record of
the study as submitted and reviewed; this page is the record of the artifacts. Nothing
here is a retraction of a result, and the central claims are unaffected - but two entries
(A1 and B1) change how a specific number should be interpreted, so they are first.

Contents:

- [A. Metric definition](#a-metric-definition)
- [B. Quantization](#b-quantization)
- [C. The on-device table](#c-the-on-device-table)
- [D. Fold construction](#d-fold-construction)
- [E. Quantization-aware training - partly unverified](#e-quantization-aware-training---partly-unverified)
- [F. Provenance notes on individual numbers](#f-provenance-notes-on-individual-numbers)

---

## A. Metric definition

### A1. The reported Challenge Score is a weighted-credit variant, not the official PhysioNet 2022 formula

**Paper.** The caption of Table 1 reads "Challenge Score follows the PhysioNet 2022
class-weighted formulation."

**Artifacts.** The implementation computes a partial-credit score with a diagonal-weight
normaliser. With the weight matrix

```
        predicted
        Pres Unk  Abs
true Pres  5    3    0
     Unk   1    3    1
     Abs   0    1    5
```

it returns

```
score = sum(confusion * W) / sum_c ( N_c * W[c, c] )
```

where `N_c` is the number of patients whose true class is `c`. Off-diagonal cells
therefore earn partial credit: predicting *Unknown* for a *Present* patient scores 3 out
of the 5 available, and the *Unknown* row contributes credit for every prediction it
makes. The official PhysioNet 2022 murmur weighted accuracy gives credit only on the
diagonal:

```
score = sum_i w_i * TP_i / sum_i w_i * N_i     with w = [5, 3, 1]
```

for Present / Unknown / Absent.

**Measured.** Recomputed from `results/predictions/`, patient level, mean +/- s.d. over the
five folds (population s.d., matching the paper):

| Model | As implemented (the paper's Table 1) | Official PhysioNet 2022 |
|---|---|---|
| MobileNetV3-S | 0.886 +/- 0.008 | 0.690 +/- 0.030 |
| FastViT | 0.888 +/- 0.009 | 0.693 +/- 0.026 |
| EfficientNet-B0 | 0.889 +/- 0.014 | 0.684 +/- 0.042 |
| TinyViT | 0.894 +/- 0.015 | 0.705 +/- 0.041 |
| ResNet-18 | **0.896** +/- 0.011 | 0.727 +/- 0.043 |
| HTS-AT | 0.875 +/- 0.020 | **0.741** +/- 0.043 |

Reproduce with `python -m analysis.table1_fivefold`, which prints both columns side by
side. `mirasol.metrics` implements both: `challenge_score()` is the variant used in the
paper, `official_challenge_score()` is the PhysioNet formula, and the docstrings say so.

**What this changes.**

- The spread across the six architectures widens from 2.1 pp to 5.6 pp, and the leading
  architecture changes from ResNet-18 to HTS-AT.
- The paper's central claim survives. Under the official metric the best and worst
  architectures, HTS-AT (0.741 +/- 0.043) and EfficientNet-B0 (0.684 +/- 0.042), still
  overlap within one standard deviation, so "diagnostic performance is largely
  independent of architecture" holds under either definition.
- The direction of the comparison in Table 3 does change. Under the variant metric our
  Group C scores (0.875-0.896) sit above the Group A hidden-test entries quoted there
  (0.771-0.780); under the official metric they sit below them (0.684-0.741). Table 3's
  own caption already states that scores are not comparable across protocol groups, and
  that caveat is doing real work here: the two groups differ in split *and*, as it turns
  out, in metric definition. Group A's figures are official-formula scores on a hidden
  test set; ours are cross-validation scores on the public partition.
- Every table script in `analysis/` reports both metrics, so no downstream number in this
  repository depends on which one a reader prefers.

---

## B. Quantization

### B1. Four of the five static-INT8 graphs are constant classifiers, and one of them is the build the abstract recommends

**Paper.** Sec. 3.5(3) reports that EfficientNet-B0 INT8-static falls to Challenge Score
0.784 and TinyViT to 0.784, and notes that the two agreeing "to four decimal places"
indicates "both collapse to a degenerate near-constant prediction". The abstract
nevertheless describes "EfficientNet-B0 with static INT8 [running] in 210 ms using 4.3 MB"
as the deployment configuration.

**Artifacts and measurement.** The collapse is not near-constant; it is constant. Measured
on the released graphs over 60 real fold-0 segments (true distribution 9 Present /
4 Unknown / 47 Absent), each segment passed through the same log-mel front end used for
export:

| Model | FP32 (P / U / A) | INT8-static (P / U / A) |
|---|---|---|
| MobileNetV3-S | 6 / 2 / 52 | **0 / 0 / 60 - always Absent** |
| EfficientNet-B0 | 8 / 2 / 50 | **0 / 0 / 60 - always Absent** |
| TinyViT | 9 / 5 / 46 | **0 / 0 / 60 - always Absent** |
| FastViT | 6 / 2 / 52 | 0 / 57 / 3 - collapsed to Unknown |
| ResNet-18 | 12 / 1 / 47 | 7 / 2 / 51 - healthy |
| HTS-AT | 12 / 15 / 33 | no static build; INT8-dynamic is identical to FP32 |

A wider sweep over all 739 fold-0 segments and all 189 fold-0 patients sharpens the
picture. Measured per-segment predictions (Present / Unknown / Absent):

| Model | INT8-static over all 739 fold-0 segments |
|---|---|
| MobileNetV3-S | 0 / 0 / **739** - literally constant |
| TinyViT | 0 / 0 / **739** - literally constant |
| EfficientNet-B0 | 2 / 0 / 737 - effectively constant, not exactly |
| FastViT | 4 / 710 / 25 - collapsed onto *Unknown* |
| ResNet-18 | 77 / 18 / 644 - healthy |

For scale, fold 0 contains roughly 124 genuinely *Present* segments, so two Present
predictions is not a working detector. The resulting patient-level scores are
0.784 / 0.784 / 0.784 / 0.311. The value 0.784 is simply what the project metric returns
for an all-*Absent* classifier on fold 0, which is why three architectures agree to four
decimal places.

**Consequence, and what this repository does about it.** An all-*Absent* murmur screener
never refers anyone. Publishing such a graph as the recommended deployable build would be
a safety problem in a clinical-screening context, so:

- The recommended deployable build in this repository is **FP32**, which is also Sec. 3.5's
  own conclusion ("selecting an inherently compact FP32 model is more effective than
  compressing a larger one").
- The four degenerate graphs are published as a separate, clearly named Release asset,
  `mirasol-onnx-int8-static-DEGENERATE-do-not-deploy.tar.gz`, with the measured constant
  prediction listed per file. They exist to reproduce Sec. 3.5(3), nothing else.
- `resnet18/model_int8_static.onnx` is published as its own asset so that it does not
  inherit the warning: it is the one static-INT8 build that survives quantization (see
  B2).
- `mirasol.deployment.export` now fails the export if the INT8 argmax is constant across
  the calibration set. That check would have caught all four of these at build time.

Reproduce with `python -m analysis.patient_level_int8 --model efficientnet_b0 --format
int8_static` (Tier 3: needs the dataset and the downloaded graphs).

### B2. ResNet-18 static INT8 does not collapse, so the abstract's "12-65%" range covers four architectures, not all of them

**Paper.** The abstract states "Post-training INT8 quantization reduces patient-level
accuracy by 12-65%". Table 4 and Sec. 3.5 cover EfficientNet-B0, FastViT, TinyViT and
HTS-AT.

**Measured.** ResNet-18 static INT8 reaches patient-level Challenge Score 0.8975 against
0.9128 for its FP32 build on fold 0, a 1.7% relative drop, with a healthy prediction
spread (644 Absent / 77 Present / 18 Unknown over 739 segments; independently confirmed as
103 / 15 / 2 on a 120-segment subsample). MobileNetV3-S was not part of the quantization
study either.

The "12-65%" range is therefore a statement about the four architectures in Table 4, not
about post-training INT8 in general, and one architecture in the six-model set behaves
qualitatively differently.

### B3. The collapse is largely an artifact of per-tensor weight quantization

**Artifacts.** The shipped recipe is `quantize_static(quant_format=QDQ,
weight_type=QInt8, activation_type=QInt8)` with `per_channel` left unset - hence `False` -
`calibrate_method` left unset - hence MinMax - `reduce_range=False`, no
`quant_pre_process`, and 50 calibration segments. `docs/quantization.md` documents the
recipe in full.

**Measured.** Recipe sweep over 200 fold-0 segments / 126 patients, patient-level majority
vote scored with the project metric. Only `per_channel=True` mattered; QUInt8 activations,
percentile calibration and a 5x larger calibration set each changed nothing on their own:

| Architecture | FP32 | shipped INT8 (per-tensor) | + per-channel weights | delta vs FP32 |
|---|---|---|---|---|
| EfficientNet-B0 | 0.883 | 0.786 (constant) | **0.862** | -2.4% (the paper reports -11.8%) |
| MobileNetV3-S | 0.887 | 0.786 (constant) | 0.812 | -8.5% |
| TinyViT | 0.872 | 0.786 (constant) | 0.746 | -14.5%, still poor |
| FastViT | 0.879 | 0.314 | 0.264 | still broken |

The control reproduces the published 0.784 (measured 0.786 on this subsample), which is
what makes the recovered figures credible.

**How this bears on the paper's claims.**

- The **mechanism** claim comes out better supported, not worse. "Attention layers appear
  more sensitive to reduced precision" becomes a clean split: both CNNs recover with
  per-channel weights, neither transformer does.
- The **magnitude** claim is recipe-dependent. The abstract's 12-65% describes a
  per-tensor export. With per-channel weights EfficientNet-B0 loses about 2%.
- **Not measured: per-channel INT8 latency on the Cortex-A53.** Per-channel quantization
  can be slower than per-tensor, and the 210 ms / "fastest and smallest" figures in the
  paper are for the per-tensor build. The Orange Pi was not available when this sweep was
  run, so no per-channel latency exists to report. The deployment recommendation in this
  repository is FP32 for the accuracy reason in B1, and nothing here implies a different
  latency conclusion.

**What changed in the code.** `per_channel=True` is now the default in
`mirasol.deployment.export`. The per-tensor recipe stays reachable with `--per-tensor`,
which is what reproduces the published graphs. The published per-tensor graphs are
labelled reproduction artifacts, not deployable models.

---

## C. The on-device table

All entries in this section are reproducible with `python -m analysis.table4_ondevice`,
which reads `results/benchmarks/` (Tier 1: no downloads).

### C1. FastViT FP32 is 12.5 MB, not 24.9 MB

Table 4 lists FastViT FP32 at 24.9 MB. The benchmark harness added the size of any
sibling `model_fp32.onnx.data` file it found next to the graph. For FastViT that sibling
was `model_fp32.onnx.data` (13,041,664 B), a stale orphan from an earlier export:
`fastvit/model_fp32.onnx` (13,119,096 B = 12.51 MiB) declares **zero** external
initializers and loads and runs correctly from an isolated directory. 12.51 + 12.44 =
24.95, which is the 24.9 in the table.

The orphan is not part of this release, the released FastViT graph is self-contained, and
`deployment/edge/benchmark.py` now adds external-data size only when the graph actually
declares external initializers.

### C2. MobileNetV3-S is the fastest and smallest configuration measured, and it is absent from Table 4

Table 4 covers four architectures and describes EfficientNet-B0 INT8-static as "fastest
and smallest" at 210 ms / 4.3 MB. MobileNetV3-S and ResNet-18 were benchmarked on the same
device with the same harness, and those rows exist in
`results/benchmarks/benchmark_2026-07_6arch.json`:

| Model | Format | Latency (ms) | Size (MB) | Peak RSS (MB) | Throughput (inf/s) |
|---|---|---|---|---|---|
| MobileNetV3-S | INT8 static | **58.7** | **1.65** | 72.0 | 17.04 |
| MobileNetV3-S | FP32 | **67.4** | 5.81 | 70.6 | 14.83 |
| EfficientNet-B0 | INT8 static | 210.1 | 4.33 | 101.9 | 4.76 |
| ResNet-18 | INT8 static | 375.3 | 10.71 | 90.7 | 2.67 |
| EfficientNet-B0 | FP32 | 456.5 | 15.30 | 132.9 | 2.19 |
| FastViT | FP32 | 510.5 | 12.51 (see C1) | 109.7 | 1.96 |
| FastViT | INT8 static | 666.4 | 3.67 | 113.5 | 1.50 |
| ResNet-18 | FP32 | 696.9 | 42.61 | 130.0 | 1.44 |
| TinyViT | INT8 static | 1294.2 | 6.42 | 195.3 | 0.77 |
| HTS-AT | FP32 | 1392.7 | 109.44 | 248.7 | 0.72 |
| TinyViT | FP32 | 1436.0 | 20.10 | 195.3 | 0.70 |

MobileNetV3-S FP32 is faster than every INT8 build of every other architecture, at 67 ms.
Read together with B1 - the MobileNetV3-S INT8-static graph is one of the constant
classifiers - the deployable optimum among the measured configurations is **MobileNetV3-S
FP32**: 67 ms, 5.81 MB, and a non-degenerate prediction distribution. That reinforces
rather than undermines the paper's conclusion that a compact FP32 model beats compressing
a larger one.

### C3. The "2-7x slower" dynamic-INT8 footnote is a desktop measurement; on device the range is 0.78x to 3.3x

Table 4's footnote reads "Dynamic-INT8 omitted (2-7x slower than FP32)". The on-device
JSON gives dynamic-INT8 / FP32 latency ratios of:

| Model | FP32 (ms) | INT8-dynamic (ms) | Ratio |
|---|---|---|---|
| EfficientNet-B0 | 456.5 | 1516.5 | 3.32x slower |
| FastViT | 510.5 | 1533.3 | 3.00x slower |
| TinyViT | 1436.0 | 2039.8 | 1.42x slower |
| HTS-AT | 1392.7 | 1089.3 | **0.78x - i.e. faster than FP32** |

HTS-AT dynamic INT8 is the one configuration where dynamic quantization helps on this
hardware. The paper's conclusion that dynamic INT8 is not useful here holds for the three
smaller models; for HTS-AT it does not.

### C4. The FP16 rows are harness failures, not hardware evidence

Sec. 3.5(2) says FP16 "needs hardware float16 support the Cortex-A53 lacks". What the JSON
records is narrower than that. Three FP16 rows carry errors, and two of them are the
benchmark script feeding a float32 array to a float16 graph:

```
efficientnet_b0 / model_fp16:
  INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)),
  expected: (tensor(float16))
fastvit / model_fp16:
  INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)),
  expected: (tensor(float16))
```

Those two are a missing input cast in the harness, which
`deployment/edge/benchmark.py` now performs by reading `sess.get_inputs()[0].type`. They
say nothing about the CPU's FP16 capability. The third is a genuine defect in the exported
graph:

```
tinyvit / model_fp16:
  FAIL : Load model ... failed: Type Error: Type (tensor(float16)) of output arg
  (/backbone/stages/stages.1/blocks/blocks.0/attn/Cast_output_0) of node
  (/backbone/stages/stages.1/blocks/blocks.0/attn/Cast) does not match expected
  type (tensor(float)).
```

`tinyvit/model_fp16.onnx` does not load in any ONNX Runtime; it is published as documented
negative evidence. The FP32-agreement figures in Sec. 3.5(2) (100.0% and 99.9%) were
measured on the desktop and are unaffected - they are in
`results/logs/static_fp16.log`.

### C5. Peak RSS for TinyViT and HTS-AT-INT8 is a shared-process high-water mark

The benchmark ran all model variants inside one long-lived process and read `VmRSS`. Peak
RSS is therefore monotonic within a run: TinyViT's three variants all report 195.3 MB,
which is the high-water mark inherited from the preceding HTS-AT sessions rather than
TinyViT's own footprint. `deployment/edge/benchmark.py` now reads `VmHWM` and runs one
model per subprocess. Re-measuring requires the device (Tier 4); the numbers in
`results/benchmarks/` are the record until someone does.

### C6. The HTS-AT FP32 ONNX is a latency-benchmark graph and is not accuracy-equivalent to the trained model

The exported HTS-AT graph bicubically interpolates the 2001-frame log-mel input down to
1024 frames and runs a single forward pass. The trained evaluation path instead averages
three native-resolution 1024-frame crops. Measured patient-level Challenge Score on fold 0
is 0.618 for the ONNX graph against 0.871 for the PyTorch model. The graph is legitimate
for the Table 4 latency row - the compute is the same shape - and is not legitimate as a
classifier. It ships as a single loose Release asset, labelled "latency-benchmark graph
only, not accuracy-equivalent", so the caveat attaches to exactly one download.

### C7. The six-architecture benchmark JSON carries the four-architecture run's timestamp

`results/benchmarks/benchmark_2026-07_6arch.json` has the same `timestamp` field
(`2026-02-26 13:20:53`) as the four-architecture file, although the MobileNetV3-S and
ResNet-18 rows were produced in a later session and their ONNX graphs are dated
2026-07-31. The shared rows are identical between the two files. Treat the `timestamp`
field in the six-architecture file as the timestamp of the original run, not of the
MobileNetV3-S and ResNet-18 measurements. The benchmark script in this release serialises
a full environment block per run so this ambiguity cannot recur.

---

## D. Fold construction

### D1. The supplement's split description omits a shuffle

**Supplement.** "Patients are assigned to five folds by deterministic round-robin over
sorted patient identifiers with seed 42, so the splits are reconstructible without
shipping index files."

**Artifacts.** The implemented algorithm has three steps, not two:

```python
patients = sorted(patient_map.keys())   # 1. sort
rng = random.Random(seed)               # seed = 42
rng.shuffle(patients)                   # 2. shuffle
folds[i % n_folds].extend(...)          # 3. round-robin
```

A literal reading of the prose - sort, then round-robin, no shuffle - places **742 of the
941** patients in the wrong fold. The splits are therefore **not** reconstructible from the
paper as written, even though the abstract promises the data splits. The three-step recipe
above, by contrast, reproduces the published partition with **0 mismatches** against the
patient sets in `results/predictions/`.

**Resolution.** `data/splits/folds.csv` (941 rows, `patient_id,fold`) ships as ground
truth. `docs/splits.md` spells out the three-step recipe, including its dependence on
CPython's Mersenne Twister implementation of `random.shuffle`, and documents why the
count is 941 rather than 942. `tests/test_splits.py` asserts that the CSV matches both the
recipe and the patient partition implied by `results/predictions/`.

Per-fold patient counts, recomputed from `results/predictions/`: 189 / 188 / 188 / 188 /
188 = 941.

---

## E. Quantization-aware training - partly unverified

### E1. What is established about the published QAT figure

Sec. 3.5(3) reports that QAT "did not recover usable accuracy - on a representative fold
the Challenge Score fell from 0.923 (FP32) to 0.318". Four things about how that number
was produced are established directly from the shipped log and the driver source, and are
stated here because they change how the number should be read. They are **not** a claim
that a correctly configured QAT run would give a different answer; no such run has been
performed.

**Established (1): the conversion to a quantized model failed.** `results/logs/qat_results.log`
line 40, for TinyViT, and line 90, for EfficientNet-B0:

```
  [Post-QAT Evaluation]
    Convert failed (cannot pickle 'module' object), using fake-quant metrics instead
    Challenge Score: 0.3184
```

**Established (2): the reported score is the last fake-quant evaluation.** The log says so
itself, and the numbers are byte-identical. From `results/logs/qat_results.log`:

```
    37:    Epoch 10/10: loss=0.8073 Score=0.3184 Acc=0.0899 F1=0.0612
    40:    Convert failed (cannot pickle 'module' object), using fake-quant metrics instead
    41:    Challenge Score: 0.3184
    50:  QAT INT8 Challenge Score: 0.3184 (delta: -0.6041)
```

The score reported as "QAT INT8" is the epoch-10 fake-quant score, to four decimal places.
The same pattern appears for EfficientNet-B0 at lines 87, 90, 91 and 100 (0.3555). So the
published QAT figure measures the float model with FakeQuantize observers attached, not an
INT8 model. This is established from the shipped log alone.

**Established (3): the QAT run diverged.** The per-epoch trace for TinyViT in the same log:

```
    Epoch 1/10: loss=1.4558 Score=0.6194 Acc=0.5767 F1=0.3818
    Epoch 2/10: loss=0.9943 Score=0.4111 Acc=0.2487 F1=0.2083
    Epoch 8/10: loss=0.8185 Score=0.3152 Acc=0.0899 F1=0.0855
    Epoch 10/10: loss=0.8073 Score=0.3184 Acc=0.0899 F1=0.0612
```

Accuracy falls from 0.8836 at the start to 0.0899, which is below the chance floor of a
three-class problem. Whatever 0.318 measures, it is a diverged run.

**Reported but not independently confirmed: the quantization configuration was mis-scoped.**
An audit of the pre-release driver reported that it attached
`torch.ao.quantization.get_default_qat_qconfig("fbgemm")` - an x86 backend, for a
Cortex-A53 target - to the whole `sed_model`, placing FakeQuantize observers on the
torchlibrosa STFT and log-mel front end and on `bn0`, and that it skipped conv-bn fusion
explicitly.

That driver is **not shipped in this repository and was not re-run**, so a reader cannot
check those two claims here; only the convert failure and the divergence above are provable
from the shipped log. They are recorded because they are why `mirasol.training.qat` now
scopes the qconfig to the backbone, fuses conv-bn-relu before `prepare_qat`, and defaults to
the `qnnpack` backend - not as findings of this release.

**Not established.** What a correctly scoped, fused, qnnpack-configured QAT run would
score. That was not measured, and this repository does not claim it. The conclusion "QAT
does not recover this loss" currently rests on the configuration described above.

**What changed in the code.** `mirasol.training.qat` fails hard on a `convert` error
instead of substituting a fake-quant score, scopes the qconfig to the backbone (leaving
`spectrogram_extractor`, `logmel_extractor` and `bn0` in FP32), fuses conv-bn-relu before
`prepare_qat`, and takes `--qbackend`, defaulting to `qnnpack`. `results/logs/qat_results.log`
ships unmodified, and the figures in it should be read as: fake-quantized, eager mode,
unfused, front end quantized, training diverged, `tq.convert` failed.

---

## F. Provenance notes on individual numbers

### F1. Two different runs are called "baseline" in the same supplement section

Sec. "Preprocessing Ablation and Class-Weight Sensitivity" uses two different baselines:

| Row in the supplement | Values | Source file in `results/predictions/` |
|---|---|---|
| Table S10 "Baseline (30s / BP / Multi)" | acc 0.873, Macro F1 0.678, Chall. 0.907, Unk F1 0.316 | `tinyvit_3class.data.csv` |
| Class-weight baseline `[4,10,1]` | Chall. 0.918, Macro F1 0.683, Unk F1 0.300 | `tinyvit_3class_fold0.data.csv` |

Both are TinyViT on the held-out fold 0 with the same preprocessing, and both are
reproduced exactly by `python -m analysis.table_s10_ablation` (Tier 1). They are two
separate training runs, and `deterministic=False` (see `docs/splits.md`) is enough to
account for the difference. `tinyvit_3class.data.csv` is the older of the two, produced by
a driver that is not part of this release; its output ships precisely because it is the
only artifact that reproduces the published S10 baseline row.

### F2. The single-location ablation row is evaluated on a different set of held-out patients

Table S10 describes "a fixed held-out fold". That holds for four of the five conditions but
not for the fifth. Recomputed from `results/predictions/`:

| Condition | Patients evaluated | Segments | Same patients as the baseline fold? |
|---|---|---|---|
| Baseline (30 s / bandpass / multi-location) | 189 | 739 | - |
| Window: 10 s | 189 | 1,750 | yes, identical set |
| Window: 15 s | 189 | 1,279 | yes, identical set |
| Filter: none | 189 | 739 | yes, identical set |
| Filter: 25-1000 Hz | 189 | 739 | yes, identical set |
| **Location: single-location only** | **187** | **224** | **no - 56 patients in common** |

The cause is mechanical. In single-location mode the preparation script keeps exactly one
recording per patient - the most audible location for *Present* patients, the first listed
recording location otherwise - so the dataset shrinks from 3.9 to about 1.2 segments per
patient and the surviving patient list changes. The fold split is a shuffle over that list
(see D1), so a different list produces a different partition. The single-location run
therefore trained on a different four-fifths of the cohort and was evaluated on a mostly
different fifth.

The published values are reproduced exactly from the committed CSV (accuracy 0.904, Macro
F1 0.661, Challenge 0.945, Unknown F1 0.125), so nothing was transcribed incorrectly. But
the +3.1 pp accuracy and the Unknown F1 drop from 0.316 to 0.125 that Sec. 3.6 draws from
this row are a comparison across two different held-out patient sets, not a controlled
one. The qualitative conclusion the paper draws - that restructuring trades aggregate
accuracy for minority-class recall - is consistent with the numbers, and the mechanism is
plausible given that single-location supervision removes almost all *Unknown* training
instances, but a reader should treat the effect size for this one row as uncontrolled.
`analysis/table_s10_ablation.py` prints the per-condition patient and segment counts
alongside the metrics so this is visible in the regenerated table.

### F3. Class-balanced resampling was active alongside the class-weighted loss

The paper reports "class-weighted cross-entropy using weights [4.0, 10.0, 1.0]". It does
not mention that the training sampler was also class-balanced. In the shipped
configuration `balanced_data = True`, and the dataset regenerates its sampling queue at
the end of every epoch by drawing, with replacement, an equal number of segments per class
up to the dataset size. Both mechanisms were therefore active in every reported run: the
sampler equalises class frequency, and the loss then re-weights the equalised batches.

This is disclosed because it affects anyone trying to match our numbers with the loss
weights alone, and because minority-class segments are seen many times per epoch. The
released `mirasol/config/defaults.py` keeps the published behaviour, with a comment saying
what it does; the upstream comment claiming the flag "only works for audioset" was wrong
for this code path and has been corrected.

### F4. Checkpoint selection monitors accuracy on the fold that is also the test fold

The paper states this and bounds it in Table S8 (peak retained checkpoint exceeds the mean
of the four retained checkpoints by 0.38-1.44 pp, mean 0.79 pp). Two mechanical details
matter for reproduction. First, the original Table S8 generator globbed the Lightning
checkpoint directories, so it pooled repeated runs of the same fold - MobileNetV3-S fold 0
has ten `version_*` directories - which makes the published figure depend on how many
times a fold happened to be re-run. This release computes S8 from
`results/checkpoint_accuracies.csv`, a frozen 30-row table of `(model, fold, checkpoint,
accuracy)`, and `analysis/checkpoint_selection.py` states explicitly how duplicate `-v1`
checkpoints are counted. Second, `save_top_k = 2` on two monitored quantities produces
four retained checkpoints per run, which is the "four retained checkpoints" of Table S8.

### F5. Two seeds, and runs are not bitwise reproducible

Seed 42 is the **split** seed (`data/splits/folds.csv`). Seed 123777 is the **training**
seed. The paper mentions only the split seed. `pl.Trainer` was not run with
`deterministic=True`, so per-fold metrics reproduce to about a percentage point rather
than bitwise; see `docs/splits.md`.

### F6. Prediction CSVs hold logits, and patient pooling is over logits

`results/predictions/*.data.csv` columns 1-3 are raw logits, not probabilities, despite
the column names an older analysis script gave them. Patient-level aggregation mean-pools
the **logits** and then takes the argmax. One earlier INT8 analysis script mean-pooled
softmax probabilities instead, which is a different rule; part of the reported
0.889 -> 0.784 drop for EfficientNet-B0 INT8 is attributable to that change of
aggregation rather than to quantization. `mirasol.metrics` now provides the single pooling
implementation that every script uses, and `analysis/patient_level_int8.py` runs FP32
through the same path so its delta is internally consistent.

### F7. One patient of the 942 is excluded, and 46 recordings are unmatched

The public partition has 942 patients and 3,163 recordings; every analysis here is over
941 patients. Patient 50321 is dropped because all four of its recordings use the
duplicate-location naming (`50321_AV_1`, `50321_AV_2`, `50321_MV_1`, `50321_MV_2`), which
the `{patient}_{location}.wav` loader does not match; 43 of 3,163 files across 17 patients
are skipped for the same reason, but only patient 50321 loses all of its recordings. The
patient-level class counts recomputed from `results/predictions/` are 179 Present /
68 Unknown / 694 Absent, against the paper's 179 / 68 / 695 over 942, so the excluded
patient is an *Absent* one. `docs/splits.md` documents this, and
`mirasol.data.prepare` now prints the skipped files at the end of every run instead of
discarding them silently.
