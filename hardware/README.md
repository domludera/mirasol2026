# Hardware

The bill of materials and platform configuration for the prototype in Fig. S2 and
Table S9 of the paper.

> [!IMPORTANT]
> **Scope of this release.** What is published here is the **bill of materials, the
> platform configuration, and the model and benchmark pipeline** in `deployment/edge/`.
> The **device application layer is not included**: no code for USB-microphone capture, the
> OLED display, the two push buttons or the PiJuice power-management HAT exists in this
> project's source, and there are no CAD or STL files for the enclosure. Fig. S2 shows the
> assembled prototype, which was built and operated, but its front-panel software was not
> maintained as releasable source and is therefore not part of this release. Do not expect a
> turnkey device from this repository. What you can do with it is run the trained models and
> reproduce the on-device benchmark on the same hardware.

## Bill of materials (Table S9)

Single-unit retail prices converted to USD, with multi-pack items charged per unit. Excludes
shipping, taxes, and the stethoscope head to which the microphone couples.

| Component | USD |
|---|---|
| SBC (Allwinner H618, 4 GB) | $36 |
| Power-management HAT | $43 |
| 1200 mAh Li-ion battery | $17 |
| 32 GB microSD card | $21 |
| USB microphone module | $11 |
| Aluminium heatsink | $9 |
| 3D-printed PLA enclosure | $7 |
| 0.96" OLED display | $4 |
| Prototyping HAT | $4 |
| Push buttons (x2) | $1 |
| **Indicative total** | **~$153** |

The SBC alone is ~$36, which is the figure the paper uses when describing the compute
target; ~$153 is the assembled prototype. Prices are indicative of 2026 single-unit retail
and will not reproduce exactly.

Two notes on the parts list:

- The **power-management HAT plus battery is $60 of the $153**, well over a third of the
  build. A deployment that can rely on mains power or a plain USB power bank is
  substantially cheaper, and nothing in the model pipeline depends on that subsystem.
- The **heatsink is not optional** for benchmarking. The H618 throttles under sustained
  load, and every latency figure in Table 4 was measured with it fitted. See
  `docs/edge_setup.md`.

## Compute platform

| Item | Value |
|---|---|
| Board | Orange Pi Zero 2 W |
| SoC | Allwinner H618 |
| CPU | Cortex-A53 x4 @ 1.5 GHz, 64-bit ARMv8-A |
| RAM | 4 GB DDR4 (3.8 GB usable) |
| Storage | 32 GB microSD |
| Connectivity | on-board Wi-Fi and Bluetooth, **not used** - inference is fully offline |
| OS | Armbian, Linux 6.6.31 aarch64 |
| Inference runtime | ONNX Runtime 1.24.2, CPU execution provider, 4 threads |
| Model input | log-mel spectrogram, `(1, 1, 2001, 64)` float32 |

The Cortex-A53 predates the ARM dot-product INT8 instructions (`SDOT`/`UDOT`). That single
hardware fact drives the paper's quantization findings: without native INT8 acceleration,
INT8 arithmetic buys no throughput, and dynamic INT8 is actually *slower* than FP32 for
three of the four architectures measured. It is the reason the paper's conclusion is
"choose a compact FP32 model" rather than "compress a large one".

### Runtime configuration used for Table 4

```
intra_op_num_threads = 4
graph_optimization_level = ORT_ENABLE_ALL
3 warmup runs, 20 timed runs per model variant
```

Set the CPU governor before benchmarking, and let the board settle:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/class/thermal/thermal_zone0/temp     # milli-degrees C
export LANG=C.UTF-8                           # minimal Armbian images leave LANG unset
```

The governor and SoC temperature at the time of the published run were not recorded. The
benchmark script in this release serialises both, along with the runtime versions and the
input checksum, so a re-measurement is self-describing. Full procedure in
`docs/edge_setup.md`.

### Measured performance, recommended builds

From `results/benchmarks/benchmark_2026-07_6arch.json`:

| Model | Format | Latency (ms) | Size (MB) | Peak RSS (MB) |
|---|---|---|---|---|
| MobileNetV3-S | FP32 | 67 | 5.81 | 71 |
| EfficientNet-B0 | FP32 | 457 | 15.30 | 133 |
| ResNet-18 | FP32 | 697 | 42.61 | 130 |

These are the three builds worth deploying. **Do not deploy the static-INT8 graphs**: four
of the five are constant classifiers, including EfficientNet-B0's, and an always-*Absent*
murmur screener never refers anyone. See `docs/quantization.md` and `docs/errata.md` B1.
A four-location patient examination at 67 ms per 30 s window is comfortably inside the
paper's `<30 s` screening budget; wall-clock time is dominated by acquiring the audio.

## Assembly and wiring

<!-- PLACEHOLDER - to be written by the author.
     Deliberately left empty rather than guessed at. Nothing in either source tree records
     the GPIO pin assignment, the I2C address or bus used for the OLED, the button wiring or
     pull-up configuration, the PiJuice HAT stacking order, or the enclosure dimensions and
     microphone-coupling geometry, and none of it is recoverable from the photograph. An
     invented wiring diagram in a clinical-hardware repository would be worse than an
     absent one.

     To fill in:
       - stacking order: SBC -> prototyping HAT -> power HAT, and which headers pass through
       - OLED: I2C bus, address, and the pins used
       - push buttons: GPIO pins, pull-up/pull-down, and what each button does
       - USB microphone: which USB port, and how it couples to the stethoscope head
       - battery: connector and polarity, and the charge/monitoring path through the HAT
       - heatsink placement and thermal-interface material
       - enclosure: outer dimensions, mounting points, cable routing, and whether the
         printable source will be released
-->

**Not yet documented.** The wiring and assembly description is the one part of this page the
authors intend to add, and it is deliberately blank rather than approximated: no GPIO pin
assignment, I2C address, button mapping, HAT stacking order or enclosure dimension is
recorded anywhere in the project's source, and none of it can be read off the photograph
reliably. Fig. S2 (`hardware/prototype.jpg`) is the only visual record published so far.

If you are building your own, the parts list above is sufficient to source the components,
and `deployment/edge/` is sufficient to run and benchmark the models on the board once it
boots. Everything between those two points - capture, display, buttons, power - is yours to
implement for now. Please open an issue if you do; a community wiring description would be
welcome here.

## Photograph

`hardware/prototype.jpg` is Fig. S2: the prototype integrating the Orange Pi Zero 2 W, the
OLED display, the PiJuice power subsystem and the 3D-printed enclosure. EXIF metadata has
been stripped.

## Clinical status of the hardware

This is a research prototype and **not a medical device**.

The point that most limits it is acquisition, not compute. Every diagnostic number in the
paper was measured on CirCor recordings made with a research-grade Littmann electronic
stethoscope. The prototype uses an $11 USB microphone module coupled to a stethoscope head.
Frequency response, SNR and motion-artifact susceptibility all differ, and **no
paired-recording study has been performed**, so the reported accuracy characterises the
model under Littmann-quality input rather than the assembled device. A paired study on the
same patients with both acquisition paths is a prerequisite to any clinical claim about this
hardware.
