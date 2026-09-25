# experiments/ - training and export drivers

Two thin drivers replace the eight scripts that produced the paper. They build
`argv` and launch the package entry points; they never edit a tracked file.

| script | what it drives |
|---|---|
| `run_experiment.py` | `cv`, `ablation`, `sweep`, `export` subcommands (subprocess) |
| `run_qat.py` | quantization-aware training (in-process, Sec. 3.5) |

Prerequisites: `pip install -e '.[train]'` (add `.[export]` for the `export`
subcommand), a CUDA GPU for anything that trains, and a prepared dataset - see
`docs/data.md` and `src/mirasol/data/prepare.py`. Every command below is run
from the repository root.

```
python experiments/run_experiment.py --help
python experiments/run_experiment.py cv --help
python experiments/run_experiment.py --list-models
```

Add `--dry-run` to any invocation to print the exact commands without running
them. That is the fastest way to see what a subcommand will do, and it needs no
GPU and no dataset.

---

## Migration table

The four `run_5fold*` names appear in the shipped provenance logs and in the
experiment names of `results/predictions/`, so here is what each retired script
maps to. The `exp_name` column is the name in `results/predictions/`.

| retired script | equivalent invocation | `exp_name` produced |
|---|---|---|
| `run_5fold_3class.py all` | `run_experiment.py cv --task 3class --models htsat,efficientnet_b0,fastvit,tinyvit` | `<model>_3class_fold<0-4>` |
| `run_5fold_3class.py tinyvit` | `run_experiment.py cv --task 3class --models tinyvit` | `tinyvit_3class_fold<0-4>` |
| `run_5fold_newmodels.py` | `run_experiment.py cv --task 3class --models mobilenetv3,resnet18` | `mobilenetv3_3class_fold<0-4>`, `resnet18_3class_fold<0-4>` |
| `run_5fold.py all` | `run_experiment.py cv --task 2class --models htsat,efficientnet_b0,fastvit` | `<model>_binary_fold<0-4>` |
| `run_5fold.py <model>` | `run_experiment.py cv --task 2class --models <model>` | `<model>_binary_fold<0-4>` |
| `run_5fold_tinyvit.py` | `run_experiment.py cv --task 2class --models tinyvit` | `tinyvit_binary_fold<0-4>` |
| `run_class_weight_sensitivity.py` | `run_experiment.py sweep --preset class-weight --model tinyvit --fold 0` | `tinyvit_cw_2_5_1_fold0`, `tinyvit_cw_4_10_1_fold0`, `tinyvit_cw_6_15_1_fold0` |
| `run_ablation.py` | `run_experiment.py ablation --model tinyvit --fold 0` | `ablation_<condition>` |
| `automate_export_onnx.py` | `run_experiment.py export --models htsat,efficientnet_b0,fastvit,tinyvit --exp-template '{model}_{tag}' --checkpoint-strategy latest --formats fp32,int8-dynamic --validate` | (writes `exports/<model>/`) |
| `automate_static_fp16.py` | `run_experiment.py export --models efficientnet_b0,fastvit,tinyvit --exp-template '{model}_{tag}' --checkpoint-strategy latest --formats fp32,int8-static,fp16 --validate` | (writes `exports/<model>/`) |
| `automate_training_3class.py` | `run_experiment.py cv --task 3class --models htsat,efficientnet_b0,fastvit,tinyvit --folds 0 --exp-template '{model}_{tag}'` | `<model>_3class` |
| `run_qat.py` | `run_qat.py --model-type tinyvit` then `run_qat.py --model-type efficientnet_b0` | (writes `runs/<exp>/qat_<model>_fold<n>.json`) |

Notes on the mappings that are not one-for-one:

* `run_5fold.py all` ran only `htsat, efficientnet_b0, fastvit`, while
  `run_5fold_3class.py all` ran those three plus `tinyvit`. `--models all` here
  means all six registered architectures, so the retired behaviour is spelled
  out explicitly in the table rather than reproduced by `all`.
* `run_5fold_tinyvit.py` was a strict subset of `run_5fold.py tinyvit`; both map
  to the same command.
* `automate_training_3class.py` is not shipped. It trained one run per
  architecture with no fold suffix and, because it patched only `model_type` and
  `exp_name`, it cannot be re-run reliably. Its `tinyvit_3class` output is the
  Table S10 baseline row and ships in `results/predictions/`; it was in fact a
  fold-0 run (its 189 patients are exactly the fold-0 partition). The command in
  the table reproduces it under the same name.
* The two `automate_*` wrappers selected a checkpoint by modification time from
  a hardcoded `version_0` directory, hence
  `--checkpoint-strategy latest --exp-template '{model}_{tag}'` above. For new
  exports prefer the defaults (`acc` strategy, per-fold experiment names).

---

## Tasks are complete parameter sets

`--task` selects one of two labelling schemes, and each one writes **every**
parameter that depends on it. This is the fix for the defect that made Table S7
irreproducible: `run_5fold.py` patched `model_type`, `exp_name`, `esc_fold` and
`dataset_npy_name` but not `classes_num` or `loss_weight`, so with the tracked
config as committed (`classes_num = 3`, `loss_weight = [4, 10, 1]`) the
"two-class" runs built a three-output head and a three-element loss weight over
two-class data. The third logit was never trainable.

| `--task` | `classes_num` | `loss_weight` | `dataset_npy_name` | `exp` tag | paper artifact |
|---|---|---|---|---|---|
| `3class` | 3 | `[4.0, 10.0, 1.0]` | `train-circor-data-30s.npy` | `3class` | Table 1, Table 2, S1-S6, S8, S10 |
| `2class` | 2 | `[4.0, 1.0]` | `train-circor-data-30s-binary.npy` | `binary` | Table S7 |

Class order for `3class` is Present = 0, Unknown = 1, Absent = 2. For `2class`
it is Present|Unknown = 0, Absent = 1.

Each run also gets `model_type`, `exp_name`, `esc_fold`, `loss_type`,
`clip_seconds`, the derived `clip_samples` (recomputed as
`sample_rate * clip_seconds`, never inherited) and `resume_checkpoint`. The
`ablation` and `sweep` subcommands go through the same task presets, so they can
no longer inherit the fold or the class count from whatever ran previously -
which the retired `run_ablation.py` did, because it never set `esc_fold`,
`classes_num` or `loss_weight` at all.

Anything else can be overridden with repeatable `--set key=value`, for example
`--set max_epoch=5 --set batch_size=8` for a smoke test. Values are parsed as
Python literals with a fallback to `str`. Every key must already exist in
`src/mirasol/config/defaults.py`; a typo aborts the whole invocation before the
first run starts.

---

## Subcommands

### `cv` - five-fold cross-validation (Table 1, Table S7)

```
python experiments/run_experiment.py cv --task 3class --models all
python experiments/run_experiment.py cv --task 3class --models tinyvit --folds 0,1
python experiments/run_experiment.py cv --task 2class --models tinyvit
```

One unit per (architecture, fold): prepare nothing, train, select a checkpoint,
test. Default `--folds` is `0,1,2,3,4` and the default experiment name template
is `{model}_{tag}_fold{fold}`.

### `ablation` - preprocessing ablation (Table S10, upper block)

```
python experiments/run_experiment.py ablation --model tinyvit --fold 0
python experiments/run_experiment.py ablation --conditions window_10s,window_15s
```

Each condition first rebuilds its own `.npy` through
`python -m mirasol.data.prepare` (pass `--skip-prepare` if the files already
exist), then trains and tests. Conditions, matching Table S10:

| condition | `.npy` | prepare flags |
|---|---|---|
| `single_location` | `ablation-single-loc-30s.npy` | `--single_location` |
| `window_10s` | `ablation-10s.npy` | `--clip_seconds 10` |
| `window_15s` | `ablation-15s.npy` | `--clip_seconds 15` |
| `no_bandpass` | `ablation-nofilter-30s.npy` | `--no_bandpass` |
| `wide_bandpass` | `ablation-wide-30s.npy` | `--bandpass_high 1000` |

Caveat worth knowing before reading the `single_location` row: that condition
keeps one recording per patient, which changes the set of surviving patients,
and the fold split is recomputed over that set. Fold 0 then holds 187 patients
instead of 189 and shares only 56 of them with the baseline fold 0, so the
`single_location` row is not a patient-matched comparison. The window and filter
conditions do not change the patient set and their fold 0 is identical to the
baseline's.

Table S10's baseline row is not produced by this subcommand; it is the main
three-class fold-0 run (`results/predictions/tinyvit_3class.data.csv`).
`analysis/table_s10_ablation.py` computes it from a named `--baseline-exp`
rather than hardcoding the published literals.

### `sweep` - single-key sensitivity (Table S10, lower block)

```
python experiments/run_experiment.py sweep --preset class-weight --model tinyvit --fold 0
python experiments/run_experiment.py sweep --key learning_rate \
    --value lr1e4=1e-4 --value lr5e4=5e-4 --model tinyvit --fold 0
```

`--preset class-weight` sweeps `loss_weight` over the three configurations
reported in the supplement:

| label | `loss_weight` | published Challenge Score / Macro F1 / Unknown F1 |
|---|---|---|
| `cw_2_5_1` | `[2.0, 5.0, 1.0]` | 0.926 / - / 0.125 |
| `cw_4_10_1` | `[4.0, 10.0, 1.0]` | 0.918 / 0.683 / 0.300 |
| `cw_6_15_1` | `[6.0, 15.0, 1.0]` | 0.906 / - / 0.235 |

The published `[4, 10, 1]` row was read off the main fold-0 run rather than a
dedicated sweep run; the preset retrains it as `tinyvit_cw_4_10_1_fold0` so all
three rows come from identically-driven runs. `results/predictions/` therefore
contains `tinyvit_cw_2_5_1_fold0` and `tinyvit_cw_6_15_1_fold0` but no
`tinyvit_cw_4_10_1_fold0`.

### `export` - ONNX and quantized graphs (Table 4, Sec. 3.5)

```
python experiments/run_experiment.py export --models all --formats fp32,int8-dynamic
python experiments/run_experiment.py export --models tinyvit \
    --formats int8-static --per-tensor --validate
```

Formats are `fp32`, `int8-dynamic`, `int8-static` and `fp16`; the driver checks
that each requested `exports/<model>/*.onnx` was actually produced and reports
`missing-artifact` if not. `--validate` runs
`python -m mirasol.deployment.validate --patient-level` on each one.

`--per-tensor` reproduces the paper's static-INT8 recipe (`per_channel=False`,
MinMax calibration). Read `docs/quantization.md` first: four of the five
per-tensor static-INT8 graphs published for reproduction are constant
classifiers and must not be deployed as screeners. The default recipe uses
per-channel weights.

### `run_qat.py` - quantization-aware training (Sec. 3.5)

```
python experiments/run_qat.py --model-type tinyvit
python experiments/run_qat.py --model-type efficientnet_b0 --fold 2 --epochs 4 \
    --ptq-reference results/benchmarks/ptq_patient_level.json
for m in tinyvit efficientnet_b0; do
    python experiments/run_qat.py --model-type "$m" || echo "FAILED: $m"
done
```

This is the only in-process driver, which is why it is a separate script rather
than a subcommand: it is a thin CLI over `mirasol.training.qat`, which owns all
of the numerical work. The driver assembles the task preset, resolves it through
`mirasol.config.resolve` and selects the FP32 starting checkpoint with the same
shared selector as `run_experiment.py` - the library deliberately contains no
checkpoint globbing so it can never pick up an unrelated run. One architecture
per invocation, matching the library. Omit `--checkpoint` to auto-select from
`runs/<exp>/`; `--exp-name` or `--exp-template` names the source experiment. The
result record goes to `runs/<exp>/qat_<model>_fold<n>.json` unless `--output`
says otherwise.

`--qbackend` defaults to `qnnpack`, which matches the Cortex-A53 deployment
target; an audit of the pre-release driver reported the published run used the x86
`fbgemm` backend (that driver is not shipped here and was not re-run). The library reports the
final fake-quantized evaluation and the converted-INT8 evaluation as separate
fields and raises if `torch.ao.quantization.convert` or the INT8 evaluation
fails, so a fake-quant number can never appear under an INT8 heading. PTQ
reference scores come from a `--ptq-reference` JSON file instead of being
hardcoded; without it no PTQ comparison is printed. `docs/errata.md` records the
status of the QAT figure in the paper, which was produced before these changes.

---

## Where the outputs go

Everything a run produces lands under `runs/<exp_name>/`, which is gitignored:

| path | written by | contents |
|---|---|---|
| `runs/<exp>/resolved_config.json` | `mirasol.config.resolve` | the full immutable config for the run - the provenance record |
| `runs/<exp>/checkpoint/lightning_logs/version_<n>/` | Lightning | checkpoints and logs |
| `runs/<exp>/predictions.csv` | test callback | one row per evaluation segment: logits, target, recording name |

`--runs-dir` or `$MIRASOL_RUNS_DIR` moves the root (both are resolved by
`mirasol.paths.resolve_paths`, and `--runs-dir` is forwarded to
`mirasol.cli.train` so the two never disagree). ONNX exports go to
`exports/<model>/`, controlled by `--output-dir` or `$MIRASOL_EXPORTS_DIR`; also
gitignored. `--report PATH` writes a JSON status record of the invocation. No
driver writes anything under version control.

The committed dumps in `results/predictions/` keep the historical
`<exp_name>.data.csv` naming so that directory stays byte-identical to the
artifacts the published tables were computed from; a fresh run writes
`runs/<exp>/predictions.csv` and you copy it across yourself. The schema is the
same - see `results/predictions/README.md`.

Every command the driver runs is printed shell-quoted, so you can copy a line
out of a log and re-run exactly that one step by hand.

## Checkpoint selection

`--checkpoint-strategy acc` (default) picks the checkpoint whose filename
carries the highest `acc=` value, breaking ties by `version_*` index and then
modification time. `version_*` directories are ordered **numerically**, so
`version_10` no longer sorts before `version_2` as it did under the retired
scripts' `sorted(glob(...))`. If no filename carries a parsable accuracy the run
fails with a message instead of scoring every candidate 0.0 and returning an
arbitrary one; `--checkpoint-strategy latest` is the escape hatch.

Disclosure that matters for reading Table 1: `esc_fold` selects the held-out
fold, and the validation dataloader is that same fold, so the monitored `acc`
used to pick a checkpoint is measured on the test fold. Table S8 bounds the
resulting optimism; `analysis/checkpoint_selection.py` regenerates it.

## Safety rails

* The driver refuses to start if `runs/<exp>/predictions.csv` already exists for
  any requested run. Pass `--overwrite` to regenerate or `--skip-existing` to
  leave those runs alone. The retired predictions writer opened the file in
  append mode, so re-running `test` for an existing experiment silently doubled
  every row and corrupted every patient-level number aggregated from it.
* Duplicate experiment names within one invocation are rejected before anything
  runs.
* Per-run status is collected and the process exits `1` if any requested run
  produced no predictions, `2` on a configuration error, `0` only when
  everything asked for completed.
* `sys.executable` is used for every subprocess, so the drivers work in any
  virtual environment.

## Seeds and reproducibility

Two different seeds are in play. `42` is the **split** seed used by
`mirasol.data.splits` and `mirasol.data.prepare` to assign patients to folds;
`random_seed = 123777` in `config/defaults.py` is the **training** seed passed to
`pl.seed_everything`. Neither is set by the drivers, so both come from the
committed defaults unless you override them.

Class-balanced resampling (`balanced_data = True`) is active on this data path in
addition to the `loss_weight` class weights - the weights are not the only
imbalance handling. Training is not bit-deterministic
(`deterministic=False`, `torch.set_float32_matmul_precision('high')`), so a
retrained fold will not reproduce the shipped predictions exactly. `docs/reproduce.md`
lists which artifacts are reproducible from the committed CSVs alone (all of
Table 1, S7, S8, S10 and Table 4) and which require retraining.
