# Quantization

> [!WARNING]
> **Four of the five published static-INT8 graphs are degenerate, two exactly constant.**
> MobileNetV3-S and TinyViT predict *Absent* for all 739 fold-0 segments and
> EfficientNet-B0 for 737 of them; FastViT
> predicts *Unknown* for almost every input. They are published to reproduce Sec. 3.5(3)
> and **must not be deployed as screeners** - an always-*Absent* murmur screener never
> refers anyone. The recommended deployable build is **FP32**. `resnet18/model_int8_static.onnx`
> is the one exception: it is not degenerate, and it ships as a separate download so that it
> does not inherit this warning.

This page documents the exact recipe behind every published quantized graph, what was
measured, and what was not explored. `docs/errata.md` sections B1-B3 carry the same facts
in reconciliation form.

## The recipes, verbatim

### FP32 export

```python
torch.onnx.export(
    export_model,
    dummy_input,                       # (1, 1, 2001, 64) float32
    "model_fp32.onnx",
    opset_version=17,
    dynamo=False,                      # legacy TorchScript exporter, for quantization compatibility
    input_names=["mel_spectrogram"],
    output_names=["logits"],
    dynamic_axes={"mel_spectrogram": {0: "batch_size"}, "logits": {0: "batch_size"}},
)
```

The graph takes a **log-mel spectrogram**, not a waveform: the torchlibrosa front end is
not exported, so the caller must reproduce it. `mirasol.signal.mel` is that reference
implementation, and `deployment/edge/mel.py` is a vendored torch-free copy of it whose
bit-for-bit agreement with the package version is asserted in CI.

### INT8 dynamic

```python
quantize_dynamic(
    model_input="model_fp32.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

Everything else is ONNX Runtime's default.

### INT8 static - as published

```python
quantize_static(
    model_input="model_fp32.onnx",
    model_output="model_int8_static.onnx",
    calibration_data_reader=CalibrationDataReader(num_samples=50),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
```

Every parameter that is *not* written above matters as much as the ones that are, because
it takes ONNX Runtime's default:

| Parameter | Value used | Consequence |
|---|---|---|
| `per_channel` | `False` (default) | one scale/zero-point per weight tensor - the dominant cause of the collapse, see below |
| `calibrate_method` | `MinMax` (default) | activation ranges set by absolute min/max over the calibration set, so a single outlier widens the range |
| `reduce_range` | `False` (default) | full 8-bit range |
| `quant_pre_process` | not called | no shape inference / graph cleanup before quantization |
| `nodes_to_exclude` | empty | attention and softmax nodes are quantized like everything else |
| `activation_type` | `QInt8` | ONNX Runtime recommends `QUInt8` for the CPU execution provider |
| calibration set | 50 segments | small, and on an input whose dynamic range is roughly -100..0 dB |

### FP16

```python
model_fp16 = onnxconverter_common.float16.convert_float_to_float16(onnx.load("model_fp32.onnx"))
```

## Calibration data provenance

The calibration reader loads the prepared 3-class `.npy` dataset, concatenates the four
**training** folds (every fold except `esc_fold`, so the held-out fold is never seen),
draws 50 segment indices with `numpy.random.RandomState(42).choice(..., replace=False)`,
self-concatenates each 30 s waveform to the 60 s model input length exactly as the training
dataset does, and converts each to a log-mel spectrogram with the same front end used for
export. So the calibration set is real patient audio from the training folds, seeded, and
never overlaps the evaluation fold.

## What was measured

### Prediction distributions on real segments

60 fold-0 segments, true distribution 9 Present / 4 Unknown / 47 Absent:

| Model | FP32 (P / U / A) | INT8-static (P / U / A) |
|---|---|---|
| MobileNetV3-S | 6 / 2 / 52 | **0 / 0 / 60 - constant** |
| EfficientNet-B0 | 8 / 2 / 50 | **0 / 0 / 60 - constant** |
| TinyViT | 9 / 5 / 46 | **0 / 0 / 60 - constant** |
| FastViT | 6 / 2 / 52 | 0 / 57 / 3 - collapsed to Unknown |
| ResNet-18 | 12 / 1 / 47 | 7 / 2 / 51 - healthy |
| HTS-AT | 12 / 15 / 33 | no static build; dynamic INT8 identical to FP32 |

Over the full fold-0 evaluation set (739 segments, 189 patients): MobileNetV3-S and TinyViT
predict *Absent* for all 739 segments; EfficientNet-B0 predicts *Absent* for 737 of 739
(2 *Present*, against roughly 124 genuinely *Present* segments, so effectively but not
exactly constant); FastViT predicts *Unknown* for 710 of 739. The patient-level Challenge
Scores are 0.784 / 0.784 / 0.784 / 0.311. 0.784 is simply what the project metric returns
for an all-*Absent* classifier on fold 0, which is why three architectures agree to four
decimal places - exactly the degeneracy Sec. 3.5(3) suspected.

### Recipe sweep

200 fold-0 segments / 126 patients, patient-level majority vote, project metric. One knob
mattered:

| Architecture | FP32 | published INT8 (per-tensor) | + `per_channel=True` | delta vs FP32 |
|---|---|---|---|---|
| EfficientNet-B0 | 0.883 | 0.786 (constant) | **0.862** | -2.4% |
| MobileNetV3-S | 0.887 | 0.786 (constant) | 0.812 | -8.5% |
| TinyViT | 0.872 | 0.786 (constant) | 0.746 | -14.5% |
| FastViT | 0.879 | 0.314 | 0.264 | still broken |

`QUInt8` activations, percentile calibration and a 5x larger calibration set each changed
nothing on their own. The control run reproduces the published 0.784 (0.786 on this
subsample), which is what makes the recovered figures credible.

Read against Sec. 3.5(3), this **strengthens** the mechanism claim and **qualifies** the
magnitude claim:

- Mechanism: "attention layers appear more sensitive to reduced precision" becomes a clean
  split - both CNNs recover with per-channel weights, neither transformer does.
- Magnitude: the abstract's 12-65% is a property of the per-tensor recipe. With per-channel
  weights EfficientNet-B0 loses about 2%.

### FP32 agreement of the FP16 graphs

`results/logs/static_fp16.log` holds the desktop measurements behind Sec. 3.5(2): 100.0%
prediction agreement with FP32 for EfficientNet-B0 and 99.9% for FastViT. Those numbers are
unaffected by the on-device FP16 failures - two of which were a harness bug, not a hardware
limit. See `docs/errata.md` C4.

## What the code does now

`mirasol.deployment.export`:

- **`per_channel=True` is the default.** `--per-tensor` restores the published recipe and is
  what you need to reproduce the released graphs.
- **A degeneracy gate.** After quantization the exporter runs the calibration samples
  through the INT8 graph and **fails the export** if the argmax is constant across 20 or
  more of them. That single check would have caught all four degenerate builds at build
  time. The old smoke test ran a single all-zeros tensor, which every constant classifier
  passes.
- **An export manifest.** Each output directory gets `export_manifest.json` recording the
  architecture, quantization format, source checkpoint and its parsed held-out accuracy,
  experiment name, held-out fold, exporter/torch/onnx/onnxruntime versions, sha256, and the
  FP32 reference metric the delta is measured against. None of this was recorded before,
  and the held-out fold of each published graph had to be recovered by comparing patient
  sets.

Reproducing the published per-tensor graphs:

```bash
python -m deployment.export_onnx --model_type efficientnet_b0 \
    --checkpoint <path/to.ckpt> --static --per-tensor --num_calib 50
```

Producing a usable static-INT8 graph instead:

```bash
python -m deployment.export_onnx --model_type efficientnet_b0 \
    --checkpoint <path/to.ckpt> --static           # per-channel by default
```

Patient-level evaluation of any exported graph:

```bash
python -m analysis.patient_level_int8 --model efficientnet_b0 --format int8_static
```

That script takes the format explicitly. Its predecessor silently fell back from
`model_int8_static.onnx` to `model_int8.onnx` when the static file was missing, which is how
a dynamic-INT8 HTS-AT measurement came to be reported under a static heading.

## Quantization-aware training

`results/logs/qat_results.log` ships as the record of the published QAT run. Read
`docs/errata.md` E1 before using its numbers: the reported "QAT INT8 Challenge Score" is a
fake-quant measurement reported after `tq.convert` raised `cannot pickle 'module' object`,
and the run diverged to below the three-class chance floor. Both are provable from the log.
An audit of the pre-release driver additionally reported that the fbgemm qconfig was attached
to the whole model including the torchlibrosa front end, with conv-bn fusion explicitly
skipped; that driver is not shipped here and was not re-run, so those two are not
independently confirmed.

`mirasol.training.qat` fixes all four: a `convert` failure is now fatal, the qconfig is
scoped to the backbone (leaving `spectrogram_extractor`, `logmel_extractor` and `bn0` in
FP32), conv-bn-relu is fused before `prepare_qat`, and `--qbackend` defaults to `qnnpack`
for an ARM target rather than the x86 `fbgemm`. The published numbers have not been
re-measured with the corrected configuration.

```bash
python experiments/run_qat.py --model tinyvit --checkpoint <path/to.ckpt> --qbackend qnnpack
```

## Limitations of this quantization study

Stated plainly, because the paper's conclusion "compression is no substitute for an
efficient architecture" rests on this study:

1. **Only one static recipe was explored for the published results.** Per-tensor MinMax
   QInt8, 50 calibration segments, no excluded operators, no pre-processing pass. The later
   sweep above varied four knobs; a systematic search was not run.
2. **No per-operator exclusion was tried.** Keeping softmax, LayerNorm or the attention
   matmuls in FP32 is the standard remedy for transformer quantization collapse and was not
   attempted. The FastViT result in particular should not be read as "FastViT cannot be
   quantized", only as "this recipe destroys it".
3. **No per-channel latency was measured on the target.** The recovered per-channel graphs
   were never benchmarked on the Orange Pi, which was unavailable. Per-channel quantization
   can be slower than per-tensor, so the 210 ms and "fastest and smallest" figures in the
   paper stand only for the per-tensor build, and nothing here establishes the latency of a
   *usable* static-INT8 build.
4. **QAT was run once, with a mis-scoped configuration, and diverged.** See above.
5. **Calibration set size was not tuned for the published graphs.** 50 segments on an input
   whose dynamic range is roughly -100..0 dB. A 5x larger set changed nothing in the sweep,
   but that was tested only in combination with the other knobs listed there.
6. **The evaluation fold is fold 0 throughout.** Quantization effects were never measured
   across all five folds, so every quantization delta in the paper and here is a single-fold
   quantity, unlike the FP32 numbers in Table 1.
