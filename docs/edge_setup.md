# Edge deployment

How to run the models on the target board. `deployment/edge/` is self-contained and
torch-free: copy that one directory to the device and nothing else.

> [!IMPORTANT]
> Deploy an **FP32** graph. Four of the five published static-INT8 graphs are constant
> classifiers - see `docs/quantization.md`. And note the scope of this release: the
> capture, display, button and power-management application described in Fig. S2 and
> Table S9 is **not** included. See `hardware/README.md`.

## The measured device

Everything in Table 4 was measured on this configuration:

| Item | Value |
|---|---|
| Board | Orange Pi Zero 2 W |
| SoC | Allwinner H618 |
| CPU | Cortex-A53 x4 @ 1.5 GHz |
| RAM | 4 GB DDR4 (3.8 GB usable) |
| OS | Armbian, Linux 6.6.31 aarch64 |
| Runtime | ONNX Runtime 1.24.2, CPU execution provider, 4 threads |
| Input | log-mel spectrogram, `(1, 1, 2001, 64)` float32 |
| Protocol | 3 warmup runs, 20 timed runs, per model variant |

The Cortex-A53 predates the ARM dot-product INT8 instructions, which is the hardware fact
behind Sec. 3.5(1): dynamic INT8 is slower than FP32 on this core for three of the four
architectures measured.

Two things the original benchmark did **not** record, and which the version in this release
does: the CPU frequency governor and the SoC temperature at the time of measurement, and
the Python and ONNX Runtime versions actually loaded. If you re-measure, the new benchmark
serialises all of them into its output JSON, so your run is self-describing in a way the
2026 runs were not.

## Preparing the board

```bash
# on the device
sudo apt update
sudo apt install -y python3-pip python3-venv
python3 -m venv ~/mirasol-env
source ~/mirasol-env/bin/activate
```

Set a UTF-8 locale before running anything. A minimal Armbian image often has `LANG`
unset, and a bare `print()` of a non-ASCII character then raises `UnicodeEncodeError`:

```bash
export LANG=C.UTF-8
```

The scripts in `deployment/edge/` emit ASCII only, so this is belt and braces, but the
default locale will bite other tools.

### Copy the edge directory

```bash
# from your workstation
scp -r deployment/edge/ user@orangepi:~/mirasol-edge/
```

`deployment/edge/` contains `benchmark.py`, `infer.py`, `mel.py` (a vendored, torch-free
copy of `mirasol.signal.mel`), `bench_input.npy`, `fetch_models.sh` and
`requirements-edge.txt`. Nothing else on the device is needed - in particular, the `mirasol`
package is not installed there.

### Install the runtime

```bash
# on the device
cd ~/mirasol-edge
pip install -r requirements-edge.txt
```

That pins `onnxruntime==1.24.2` exactly, plus numpy, soundfile and scipy. The pin is exact
on purpose: `quantize_static` defaults change across ONNX Runtime releases, and 1.24.2 is
the version in the Table 4 caption. librosa is deliberately **not** a dependency, so numba
and llvmlite never have to build on aarch64; `mel.py` implements the bandpass, tiling and
log-mel transform with numpy and scipy only.

If pip cannot find an `aarch64` wheel for `onnxruntime==1.24.2` under your Python version,
use a Python version for which one is published rather than relaxing the pin - a different
runtime version changes the numbers you are trying to compare against.

The Python version used for the published measurements was not recorded, since the original
benchmark script did not serialise it.

## Fetching the models

Trained graphs ship as GitHub Release assets, not in the repository (18 files,
316,643,361 B in total, one of them 109.4 MiB, above GitHub's 100 MiB blob limit).

```bash
bash fetch_models.sh --group fp32          # the recommended deployable set
bash fetch_models.sh --model mobilenetv3   # a single architecture
bash fetch_models.sh --group all
```

`fetch_models.sh` is POSIX `sh` and uses only `curl`, `sha256sum` and `tar`, so it runs on a
bare Armbian image with no `gh` and no Python. It verifies every download against
`SHA256SUMS.txt` from the same release and **refuses to continue on a mismatch**. Resulting
layout:

```
models/
  mobilenetv3/model_fp32.onnx
  efficientnet_b0/model_fp32.onnx
  resnet18/model_fp32.onnx
  ...
```

`--models-dir models` is the default for both `benchmark.py` and `infer.py`, so nothing
needs moving afterwards.

## Benchmarking

```bash
python3 benchmark.py --models-dir models --input bench_input.npy --threads 4
```

`bench_input.npy` is committed to the repository - force-added past the `*.npy` rule -
because it is the only input to Table 4 and requiring a download for it would make that
table unreproducible from a clean clone. It is purely synthetic and contains no patient
data: `numpy.random.RandomState(42)` noise at amplitude 0.01, plus 50 Hz and 120 Hz
sinusoids at amplitude 0.1 and 0.05, tiled to the model input length.

```
size    512,384 B
shape   (1, 1, 2001, 64) float32
sha256  f8a7c2d832b305de234c16f31c66e476065574915f1204d98966c78f17235d05
```

Check it before comparing timings, so you know you are timing the same input:

```bash
sha256sum bench_input.npy
```

Regenerate it with `python -m deployment.generate_bench_input --output deployment/edge/bench_input.npy`,
which prints the sha256 of the array it wrote.

What the benchmark does per variant: loads the session with
`ORT_ENABLE_ALL` graph optimisation, casts the input to the dtype the session declares,
runs 3 warmups and 20 timed inferences, and reports mean, s.d., min, max, p50, p95, p99
latency, throughput, model size, load time and peak RSS. Changes from the version that
produced Table 4, each of which affects the numbers:

- **Input dtype is cast per session.** Every FP16 row in the published JSON is an
  `INVALID_ARGUMENT: Unexpected input data type` from feeding float32 into an FP16 graph -
  a harness bug, not a hardware limit (`docs/errata.md` C4).
- **Peak RSS comes from `VmHWM`, one model per subprocess.** The published run read `VmRSS`
  inside one long-lived process, so peak RSS was monotonic across variants: TinyViT's
  195 MB is inherited from the preceding HTS-AT sessions (`docs/errata.md` C5).
- **External-data size is counted only when the graph declares external initializers.**
  Unconditionally adding a sibling `.onnx.data` is what inflated FastViT FP32 to 24.9 MB
  when the graph is 12.5 MB and self-contained (`docs/errata.md` C1).
- **`--cooldown` between variants**, and an environment block in the output JSON: ONNX
  Runtime / numpy / Python versions, `platform.machine()`, thread count,
  `thermal_zone0` temperature, `scaling_cur_freq`, and the input sha256.

For comparable timings, pin the governor and let the board settle:

```bash
# check what you have
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cat /sys/class/thermal/thermal_zone0/temp        # milli-degrees C

# pin to the highest frequency for the duration of a benchmark
sudo cpufreq-set -g performance                  # or:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

The H618 throttles under sustained load, which is why the prototype's bill of materials
includes an aluminium heatsink. Run the benchmark with the heatsink fitted, use
`--cooldown` to insert an idle gap between variants, and treat any run where
`thermal_zone0` climbs during timing as suspect. The governor and temperature of the 2026
runs were not recorded, so a re-measurement that differs by tens of milliseconds is not
necessarily a disagreement.

The benchmark runs on any Linux machine, but the numbers are then not comparable with
Table 4. `results/benchmarks/*.json` is the record of the device measurements.

## Inference on a real recording

```bash
python3 infer.py --model models/mobilenetv3/model_fp32.onnx --audio recording.wav
python3 infer.py --model models/mobilenetv3/model_fp32.onnx --audio-dir patient_dir/ --patient
```

`infer.py` implements the decision rule the paper's numbers were produced with, which the
previous version did not:

- **Three classes.** `CLASS_NAMES = ("Present", "Unknown", "Absent")`, asserted against the
  session's output dimension at load. The previous version declared two names against
  three-logit graphs and raised `IndexError` on every file, swallowed by a bare `except`
  into one error line - so it produced no prediction at all, for any released model.
- **Repeat tiling, not zero padding.** A clip shorter than 30 s is tiled by repeating
  itself, which is what training did. Zero padding is a different input.
- **Non-overlapping 30 s windows with mean-pooled logits.** The previous version truncated
  to the first 30 s.
- **`--patient` pools across auscultation locations**, mean-pooling logits over every
  recording in the directory before the argmax. That is the patient-level rule behind every
  number in Table 1 and Table 2.
- **Input dtype cast per session**, as in the benchmark.

Output is the per-class softmax of the pooled logits and the argmax label. A prediction of
*Unknown* means the acoustic quality did not permit a decision and should trigger
re-acquisition, not a referral decision.

## Expected end-to-end time

Model inference on the recommended FP32 CNN builds is 67 ms (MobileNetV3-S) to 456 ms
(EfficientNet-B0) per 30 s window. The log-mel front end in `mel.py` is numpy and scipy and
runs in the same order of magnitude. A four-location patient examination therefore stays
inside the paper's `<30 s` screening budget with a large margin, and the wall-clock cost is
dominated by acquiring the audio, not by the network.

## Caveat that bounds all of this

The models were trained and evaluated on CirCor recordings made with a research-grade
Littmann electronic stethoscope. The prototype uses a low-cost USB microphone coupled to a
stethoscope head. Differences in frequency response, SNR and motion-artifact susceptibility
may shift end-to-end performance, and no paired-recording study has been done. Numbers
measured on this device with this microphone are not the numbers in Table 1.
