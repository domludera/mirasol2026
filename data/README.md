# Data

**No CirCor DigiScope file is committed to this repository.** This directory holds only
this README and `splits/`, the patient-to-fold assignment. `.gitignore` blocks everything
else under `data/`, and CI fails the build if a CirCor file appears.

`docs/data.md` is the long-form companion to this page: what is read from the dataset, what
is not, and what ODC-By requires of you.

## You probably only need 114 KB

Read this before downloading the whole dataset.

| What you want to reproduce | What you need to download |
|---|---|
| Table 1, Table 4, Tables S7, S8, S10, the class-weight sweep | **nothing** - these come from the committed `results/predictions/` and `results/benchmarks/` |
| Table 2, Tables S1-S6, Fig. S1 (clinical endpoints) | **`training_data.csv` only - 116,962 B** |
| Fig. 1 (preprocessing pipeline) | one WAV file, `9979_TV.wav` |
| Rebuilding the `.npy` datasets, retraining, ONNX export, INT8 evaluation | the full training partition, ~450 MB zipped / ~560 MB unpacked |

The clinical-endpoint analysis needs the metadata CSV and nothing else, because it joins the
committed out-of-fold predictions against three columns of it: `Outcome` (the examining
cardiologist's diagnosis) and the two murmur-grading columns. Those are CirCor's clinical
labels, which is the one thing this repository cannot ship for you.

## Getting the dataset

The **CirCor DigiScope Phonocardiogram Dataset** is distributed by PhysioNet:

- <https://physionet.org/content/circor-heart-sound/>

The metadata file alone:

```bash
mkdir -p data/circor
curl -O https://physionet.org/files/circor-heart-sound/1.0.3/training_data.csv
mv training_data.csv data/circor/
```

The full training partition (~450 MB as a zip, ~560 MB unpacked, of which ~555 MiB is
audio; check the version number on the PhysioNet page before copying these commands):

```bash
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
```

or, faster:

```bash
aws s3 sync --no-sign-request s3://physionet-open/circor-heart-sound/1.0.3/ ./circor/
```

PhysioNet requires no credentialling for this dataset, but you do accept its licence terms
by downloading it. Verify your download against the `SHA256SUMS.txt` that PhysioNet ships.

## Expected layout

The path resolver looks here by default:

```
data/
  README.md                      <- this file (committed)
  splits/
    folds.csv                    <- 941 rows, patient_id,fold (committed)
    README.md                    <- licence and provenance of folds.csv (committed)
  circor/                        <- NOT committed, you create this
    training_data.csv            <- 116,962 B, the metadata
    training_data/               <- ~560 MB: 3,163 .wav plus .hea and .tsv sidecars
      9979_AV.wav
      9979_MV.wav
      9979_TV.wav
      ...
    train-circor-data-30s.npy         <- built by you, 1.66 GiB
    train-circor-data-30s-binary.npy  <- built by you, 1.66 GiB
```

Nothing has to live here. Every script accepts explicit paths, and two environment variables
override the defaults:

```bash
export CIRCOR_ROOT=/mnt/datasets/circor        # sets both of the below
export CIRCOR_META=/mnt/datasets/circor/training_data.csv
export CIRCOR_AUDIO=/mnt/datasets/circor/training_data
```

Command-line flags (`--circor-meta`, `--audio-dir`) beat the environment, which beats the
default. If a CirCor path is needed and missing, the error names the PhysioNet URL instead of
printing a traceback.

## Building the prepared datasets

> [!WARNING]
> **The script defaults are not the configuration the paper used.** The preparation script
> defaults to `--clip_seconds 10` and `--output_name train-circor-data.npy`, while the
> training drivers expect 30 s clips in `train-circor-data-30s.npy`. Passing the arguments
> below is not optional - omitting them silently produces a dataset that trains and reports
> numbers that will not match the paper.

The paper's main 3-class dataset:

```bash
mirasol-prepare \
    --num-classes 3 \
    --raw_dir data/circor/training_data \
    --csv_file data/circor/training_data.csv \
    --output_dir data/circor \
    --output_name train-circor-data-30s.npy \
    --sample_rate 4000 \
    --clip_seconds 30 \
    --n_folds 5 \
    --seed 42
```

The 2-class dataset for Table S7 (*Unknown* merged into *Present*):

```bash
mirasol-prepare --num-classes 2 \
    --raw_dir data/circor/training_data --csv_file data/circor/training_data.csv \
    --output_dir data/circor --output_name train-circor-data-30s-binary.npy \
    --sample_rate 4000 --clip_seconds 30 --n_folds 5 --seed 42
```

The five ablation datasets for Table S10:

```bash
# single-location supervision (no multi-location restructuring)
mirasol-prepare --num-classes 3 --single_location --clip_seconds 30 \
    --output_name ablation-single-loc-30s.npy ...

# 10 s and 15 s context
mirasol-prepare --num-classes 3 --clip_seconds 10 --output_name ablation-10s.npy ...
mirasol-prepare --num-classes 3 --clip_seconds 15 --output_name ablation-15s.npy ...

# no bandpass, and a wider 25-1000 Hz bandpass
mirasol-prepare --num-classes 3 --no_bandpass --clip_seconds 30 \
    --output_name ablation-nofilter-30s.npy ...
mirasol-prepare --num-classes 3 --bandpass_low 25 --bandpass_high 1000 --clip_seconds 30 \
    --output_name ablation-wide-30s.npy ...
```

(`...` stands for the same `--raw_dir` / `--csv_file` / `--output_dir` / `--sample_rate` /
`--n_folds` / `--seed` arguments as the first command.)

`mirasol-prepare` is the console entry point for `mirasol.data.prepare`; `python -m
mirasol.data.prepare` is equivalent.

## What the preparation step does

For each recording: load, resample to 4 kHz, apply a 4th-order Butterworth bandpass
(25-400 Hz by default, causal `sosfilt`), z-score normalise per recording, then cut into
non-overlapping 30 s windows, repeat-tiling anything shorter than one window. Multi-label
recordings are restructured into separate single-label, valve-specific instances (Sec. 2.3).
Patients are then partitioned into five folds - see `docs/splits.md` for the exact recipe,
including the shuffle the supplement omits.

The output is a pickled numpy object array of five folds of dicts with keys `name`,
`target`, `waveform` (float32, 4 kHz) and `patient`. Log-mel conversion happens inside the
model, not here.

Two diagnostics printed at the end of every run, which the original script did not print:
every recording whose filename did not match `{patient}_{location}.wav` (43 of 3,163, over 17
patients) and every patient that lost all of its recordings as a result (exactly one, 50321).
That is why the cohort is 941 patients rather than 942.

Sizes to expect:

| File | Size |
|---|---|
| `train-circor-data-30s.npy` | 1.66 GiB |
| `train-circor-data-30s-binary.npy` | 1.66 GiB |
| `ablation-single-loc-30s.npy` | 0.50 GiB |
| `ablation-10s.npy` | 1.31 GiB |
| `ablation-15s.npy` | 1.43 GiB |
| `ablation-nofilter-30s.npy` | 1.66 GiB |
| `ablation-wide-30s.npy` | 1.66 GiB |

## Licence: ODC Attribution License (ODC-By 1.0)

The dataset is licensed by its authors under the **Open Data Commons Attribution License
v1.0**: <https://opendatacommons.org/licenses/by/1-0/>. You may copy, distribute, use,
modify and produce derivative databases, provided you attribute the source and keep the
attribution notice intact.

**`data/splits/folds.csv` is a Derivative Database under that licence**, not MIT-licensed
software, because it is a partition of CirCor patient identifiers. The same applies to
`results/predictions/`. Each directory carries a README saying so; `LICENSE` section 3 and
`THIRD_PARTY.md` section 6 record the scope split for the repository as a whole.

## Required citations

If you use the dataset, cite all three - this is the attribution ODC-By requires:

> Oliveira, J., Renna, F., Costa, P.D., Nogueira, M., Oliveira, C., Ferreira, C., Jorge, A.,
> Mattos, S., Hatem, T., Tavares, T., Elola, A., Rad, A.B., Sameni, R., Clifford, G.D.,
> Coimbra, M.T. The CirCor DigiScope dataset: from murmur detection to murmur
> classification. *IEEE Journal of Biomedical and Health Informatics* 26(6), 2524-2535
> (2022). <https://doi.org/10.1109/JBHI.2021.3137048>

> Reyna, M.A., Kiarashi, Y., Elola, A., Oliveira, J., Renna, F., Gu, A., Perez Alday, E.A.,
> Sadr, N., Sharma, A., Kpodonu, J., Mattos, S., Coimbra, M.T., Sameni, R., Rad, A.B.,
> Clifford, G.D. Heart murmur detection from phonocardiogram recordings: the George B. Moody
> PhysioNet Challenge 2022. *PLOS Digital Health* 2(9), e0000324 (2023).
> <https://doi.org/10.1371/journal.pdig.0000324>

> Goldberger, A.L., Amaral, L.A.N., Glass, L., Hausdorff, J.M., Ivanov, P.C., Mark, R.G.,
> Mietus, J.E., Moody, G.B., Peng, C.-K., Stanley, H.E. PhysioBank, PhysioToolkit, and
> PhysioNet: components of a new research resource for complex physiologic signals.
> *Circulation* 101(23), e215-e220 (2000). <https://doi.org/10.1161/01.CIR.101.23.e215>

The screening campaigns that produced the data were approved by the Complexo Hospitalar
HUOC/PROCAPE institutional review board (Pernambuco, Brazil). No new human-subject data were
collected for this study.
