# The dataset

Long-form companion to `data/README.md`. That file tells you how to get the data; this one
tells you what is used from it, what is not, and what the licence requires.

## The dataset is not redistributed here

This repository contains **no CirCor DigiScope audio and no CirCor DigiScope metadata
file**. `.gitignore` blocks `data/**` except `data/README.md` and `data/splits/`, and the
CI check fails the build if a CirCor file appears.

That is a deliberate policy, not an oversight. The CirCor metadata is individual-level
clinical data: `training_data.csv` holds one record per patient with age band, sex, height,
weight, pregnancy status, per-location murmur gradings, the examining cardiologist's
outcome, the recording campaign, and an "Additional ID" that cross-links repeat visits of
the same patient. It is public, but publishing a second copy of it inside an MIT-licensed
repository would both breach this project's own no-redistribution constraint and drag a
second licence (ODC-By) over part of an otherwise MIT tree.

Get it from PhysioNet:

- <https://physionet.org/content/circor-heart-sound/>

## Licence: ODC Attribution License (ODC-By 1.0)

The dataset is licensed by its authors under the **Open Data Commons Attribution License
v1.0**, <https://opendatacommons.org/licenses/by/1-0/>. In outline, ODC-By lets you copy,
distribute, use, modify and produce derivative databases, on the single condition that you
attribute the source; section 4.2 further requires that if you publicly use a Derivative
Database you keep the attribution notice intact and make the licence terms available.

Two consequences for this repository:

1. **You must attribute CirCor yourself.** Using our code to train on the dataset does not
   transfer our attribution to you. The required citations are below and in
   `THIRD_PARTY.md` section 6.
2. **Two directories here are Derivative Databases.** `results/predictions/` carries CirCor
   recording filenames (patient identifier plus auscultation location) and the murmur
   target derived from CirCor's `Murmur` column; `data/splits/folds.csv` carries CirCor
   patient identifiers partitioned into folds. Those two directories are therefore under
   **ODC-By 1.0, not MIT**, and each carries a `README.md` saying so. `LICENSE` section 3
   records the same split.

Nothing non-public is added by shipping them: the identifiers and the murmur labels are
already in the CirCor distribution. What shipping them buys is that every table in the
paper can be regenerated with no download at all, which is the main reproducibility win of
this release. Dropping the target column to be maximally conservative would force a join
against `training_data.csv` and would move Table 1 out of reach of a clean clone.

## Required attribution

Cite all three:

> Oliveira, J., Renna, F., Costa, P.D., Nogueira, M., Oliveira, C., Ferreira, C., Jorge,
> A., Mattos, S., Hatem, T., Tavares, T., Elola, A., Rad, A.B., Sameni, R., Clifford,
> G.D., Coimbra, M.T. The CirCor DigiScope dataset: from murmur detection to murmur
> classification. *IEEE Journal of Biomedical and Health Informatics* 26(6), 2524-2535
> (2022). <https://doi.org/10.1109/JBHI.2021.3137048>

> Reyna, M.A., Kiarashi, Y., Elola, A., Oliveira, J., Renna, F., Gu, A., Perez Alday,
> E.A., Sadr, N., Sharma, A., Kpodonu, J., Mattos, S., Coimbra, M.T., Sameni, R., Rad,
> A.B., Clifford, G.D. Heart murmur detection from phonocardiogram recordings: the George
> B. Moody PhysioNet Challenge 2022. *PLOS Digital Health* 2(9), e0000324 (2023).
> <https://doi.org/10.1371/journal.pdig.0000324>

> Goldberger, A.L., Amaral, L.A.N., Glass, L., Hausdorff, J.M., Ivanov, P.C., Mark, R.G.,
> Mietus, J.E., Moody, G.B., Peng, C.-K., Stanley, H.E. PhysioBank, PhysioToolkit, and
> PhysioNet: components of a new research resource for complex physiologic signals.
> *Circulation* 101(23), e215-e220 (2000). <https://doi.org/10.1161/01.CIR.101.23.e215>

The screening campaigns that produced the data were approved by the Complexo Hospitalar
HUOC/PROCAPE institutional review board (Pernambuco, Brazil). No new human-subject data
were collected for this study.

## What this project reads from the dataset

### Audio

`training_data/*.wav`, 581,673,812 B (~555 MiB) over 3,163 recordings. Needed only to
rebuild the prepared `.npy` datasets, to
retrain, and to regenerate Fig. 1 (which needs exactly one file, `9979_TV.wav`).

Recordings are named `{patient_id}_{location}.wav`, where location is one of `AV`, `MV`,
`PV`, `TV`, `Phc`. The loader matches that pattern exactly, which has one consequence
worth knowing before you rebuild: 43 of the 3,163 recordings, spread over 17 patients, use
a duplicate-location form (`50321_AV_1`, `50321_AV_2`, ...) and are skipped, leaving 3,120.
For patient 50321 all four of its recordings are skipped, so that patient disappears from
the cohort. That is why every analysis here is over **941** patients and not the 942 in the
metadata; see `docs/splits.md`. The preparation script prints every skipped file at the end
of each run.

### Metadata

`training_data.csv`, **116,962 B**. Six columns are read, and no others:

| Column | Used for |
|---|---|
| `Patient ID` | joining to recordings and to `results/predictions/` |
| `Murmur` | the training target: `Present` -> 0, `Unknown` -> 1, `Absent` -> 2 |
| `Recording locations:` | multi-location restructuring (Sec. 2.3) |
| `Outcome` | the examining cardiologist's diagnosis - clinical endpoint E1 |
| `Systolic murmur grading` | endpoint E3 and the severity strata of Table S5 |
| `Diastolic murmur grading` | endpoint E3 and the severity strata of Table S5 |

`Age` is read when present and treated as optional, so a reduced extract still works.
Nothing else in the file is touched: height, weight, pregnancy status, campaign and the
cross-linking "Additional ID" are never read, and none of them influences any published
number.

The clinical-endpoint analysis - Table 2, Tables S1-S6 and Fig. S1 - needs **only**
`training_data.csv`. That is a 114 KB download rather than the whole dataset, because those
tables are
computed by joining the committed out-of-fold predictions against `Outcome` and the two
grading columns. `Outcome` and the gradings are CirCor's clinical labels and are the one
thing this repository cannot ship for you.

### What is deliberately not used

The dataset also distributes `train_meta_data.csv`, segmentation `.tsv` files and
per-recording `.hea` headers. None is read by any script here. Earlier development copies
of the metadata (`training_data_copy.csv`, `train_meta_data_backup.csv`,
`train_meta_data_old.csv`) are not part of this release either; two of them used a
contradictory murmur encoding and would mislead anyone trying to reproduce the labels.

## Label definitions used here

Three-class task (`--num-classes 3`, the paper's main protocol):

| Class | Index | CirCor `Murmur` value |
|---|---|---|
| Present | 0 | `Present` |
| Unknown | 1 | `Unknown` |
| Absent | 2 | `Absent` |

Two-class task (`--num-classes 2`, Table S7):

| Class | Index | CirCor `Murmur` value |
|---|---|---|
| Murmur | 0 | `Present` or `Unknown` |
| Absent | 1 | `Absent` |

The index order matters and is asserted in code: it is the order the exported ONNX graphs
emit logits in, and `CLASS_NAMES = ("Present", "Unknown", "Absent")` in
`mirasol.config.defaults` is the single definition. An earlier edge-inference script
declared only two class names against three-logit graphs, which made it raise `IndexError`
on every file; the class-name tuple is now asserted against the session's output dimension
at load time.

Patient-level cohort composition, recomputed from `results/predictions/`: 179 Present,
68 Unknown, 694 Absent, total 941. The paper's 179 / 68 / 695 over 942 differs by the one
excluded patient, which is an *Absent* one.

## Clinical endpoints

Defined from the CirCor columns above, following the paper's Sec. 3.2:

| Endpoint | Definition | Prevalence |
|---|---|---|
| E1 abnormal | `Outcome == Abnormal` | 455 / 941 (48.4%) |
| E2 pathological murmur | murmur present **and** `Outcome == Abnormal` | 150 / 941 (15.9%) |
| E3 non-innocent murmur | systolic grade >= III/VI, or any diastolic murmur | 50 / 941 (5.3%) |

`analysis/clinical_endpoint.py` is the single implementation, and it restricts the
reference description to the 941 analysed patients rather than all 942 metadata rows.

## Storage footprint

| Item | Size |
|---|---|
| CirCor audio, `training_data/*.wav` (3,163 files) | 581,673,812 B (~555 MiB) |
| CirCor `training_data.csv` | 114 KB (116,962 B) |
| `train-circor-data-30s.npy` (3-class, prepared) | 1.66 GiB |
| `train-circor-data-30s-binary.npy` (2-class, prepared) | 1.66 GiB |
| `ablation-single-loc-30s.npy` | 0.50 GiB |
| `ablation-10s.npy` | 1.31 GiB |
| `ablation-15s.npy` | 1.43 GiB |
| `ablation-nofilter-30s.npy`, `ablation-wide-30s.npy` | 1.66 GiB each |
| Committed evidence in this repository (`results/`) | ~3.2 MB |

The prepared `.npy` files hold 4 kHz float32 waveforms, not spectrograms; the log-mel
transform happens inside the model. See `data/README.md` for the exact build commands.
