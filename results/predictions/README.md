# Out-of-fold prediction dumps

Committed evidence for every patient-level number in the paper. These 61 CSV files
(3,209,638 B) are the *only* thing an analysis script needs: every table and figure derived
from model predictions can be regenerated from this directory alone, with no GPU, no
checkpoints and — except for the clinical-endpoint tables, which additionally need CirCor's
`training_data.csv` for the reference labels — no dataset download.

## Licence and attribution

**This directory is not covered by the repository's MIT licence.**

The `filename` column carries CirCor DigiScope recording identifiers and the `target` column
carries the murmur label derived from CirCor's `Murmur` annotation, so this directory is a
derived database:

> Contains information from the CirCor DigiScope Phonocardiogram Dataset made available
> under the Open Data Commons Attribution License (ODC-By 1.0).

Both identifiers and labels are already public in CirCor, so nothing here is newly disclosed;
scoping the licence is a compliance obligation, not a privacy one. `data/splits/folds.csv`
carries the same notice. See `LICENSE` and `THIRD_PARTY.md`.

Cite the dataset when you use these files:

- Oliveira, J., Renna, F., Costa, P.D., et al. *The CirCor DigiScope dataset: from murmur
  detection to murmur classification.* IEEE J. Biomed. Health Inform. **26**(6), 2524–2535
  (2022). doi:10.1109/JBHI.2021.3137048
- Reyna, M.A., Kiarashi, Y., Elola, A., et al. *Heart murmur detection from phonocardiogram
  recordings: the George B. Moody PhysioNet Challenge 2022.* PLOS Digit. Health **2**(9),
  e0000324 (2023). doi:10.1371/journal.pdig.0000324

## Column schema

**There is no header row.** Every file is machine-written, comma-separated, one row per
evaluated audio segment, in the order the test dataloader emitted it.

Three-class files (5 columns):

| # | Column | Type | Meaning |
|---|--------|------|---------|
| 0 | `logit_present` | float | raw score for *Present* |
| 1 | `logit_unknown` | float | raw score for *Unknown* |
| 2 | `logit_absent`  | float | raw score for *Absent* |
| 3 | `target`        | float | ground-truth class index, written as `0.0`/`1.0`/`2.0` |
| 4 | `filename`      | str   | CirCor segment identifier, `{patient_id}_{location}[-{window}].wav` |

Two-class files (4 columns) drop the `logit_unknown` column: `logit_murmur`,
`logit_absent`, `target`, `filename`, with class 0 = *Present* ∪ *Unknown* (the merged murmur
class) and class 1 = *Absent*.

Details that matter for reuse:

- **The three columns are unnormalised logits, not probabilities.** Training used
  `loss_type = "weighted_clip_ce"`, and no softmax is applied before the dump. Across the 61
  files the values span −112.409 to 68.494 and rows do not sum to 1. `argmax` is unaffected,
  but any calibration, ROC or threshold-sweep analysis must apply a softmax first. The
  continuous screening score used for Table S6 and Fig. S1(a) is
  `softmax(pooled_logits)[Present] + softmax(pooled_logits)[Unknown]`.
- **Class order is Present = 0, Unknown = 1, Absent = 2** for the three-class task, matching
  `mirasol.data.prepare` and `CLASS_NAMES` in `mirasol.config.defaults`.
- `target` is the patient-level murmur label repeated on every segment of that patient. It is
  verified constant within each patient in all 61 files, so grouping with `first` is safe.
- `filename` values are CirCor's own de-identified pseudonyms — the file names as distributed
  by PhysioNet. They are not derived from any patient-identifying information. `patient_id`
  is the text before the first underscore; `location` is one of `AV`, `MV`, `PV`, `TV`, `Phc`
  (30 rows repo-wide use `Phc`); the optional `-{window}` suffix is the index of a 30 s
  window when one recording is long enough to yield more than one.
- **Patient level is reached by mean-pooling the segment logits, then `argmax`** — never by
  majority vote over segments. Every published patient-level number uses mean pooling.
- No file contains a duplicated row. (`toolbox/preds_target_to_csv.py` opened its output in
  append mode, so re-running `test` for an existing experiment name could have doubled a
  file; it did not happen to any file shipped here, and the callback in
  `mirasol.training.predictions` now writes with mode `w`.)

## Fold structure

The five `*_fold{k}` files of one experiment family are the five held-out folds, disjoint at
the patient level and exhaustive over the 941 retained patients. All ten five-fold families
(six architectures × three-class, four architectures × two-class) agree on **exactly the same
patient partition**, with 0 patients assigned differently — this partition is what
`data/splits/folds.csv` records.

| Fold | Patients | Segments | Present / Unknown / Absent (patients) |
|------|----------|----------|----------------------------------------|
| 0 | 189 | 739 | 34 / 14 / 141 |
| 1 | 188 | 747 | 36 / 16 / 136 |
| 2 | 188 | 756 | 42 / 10 / 136 |
| 3 | 188 | 753 | 34 / 14 / 140 |
| 4 | 188 | 726 | 33 / 14 / 141 |
| **total** | **941** | **3,721** | **179 / 68 / 694** |

CirCor's public training partition holds 942 patients labelled 179 *Present* / 68 *Unknown* /
695 *Absent*; one *Absent* patient (50321) is dropped during dataset preparation, which is
the single difference. See `docs/splits.md`.

## What each file feeds

| Files | Count | Paper artifact |
|---|---|---|
| `{mobilenetv3,fastvit,efficientnet_b0,tinyvit,resnet18,htsat}_3class_fold{0..4}.data.csv` | 30 | Table 1 (all seven metrics), Table 2, Tables S1–S6, Fig. S1 |
| `{efficientnet_b0,fastvit,tinyvit,htsat}_binary_fold{0..4}.data.csv` | 20 | Table S7 (two-class murmur task) |
| `tinyvit_3class.data.csv` | 1 | Table S10 baseline row |
| `ablation_single_location.data.csv`, `ablation_window_{10s,15s}.data.csv`, `ablation_{no,wide}_bandpass.data.csv` | 5 | Table S10 ablation rows |
| `tinyvit_3class_fold0.data.csv` | (also above) | class-weight sweep reference `[4,10,1]` (Supp. §5) |
| `tinyvit_cw_2_5_1_fold0.data.csv`, `tinyvit_cw_6_15_1_fold0.data.csv` | 2 | class-weight sweep `[2,5,1]` and `[6,15,1]` (Supp. §5) |
| `{efficientnet_b0,fastvit,tinyvit,htsat}_3class.data.csv` | 4 | fold-0 FP32 baselines for the quantization deltas of Sec. 3.5; these are the four runs whose checkpoints were exported to ONNX (`results/logs/int8_quantization.log`, lines 516/522/528/534) |

Not shipped: `{0,1,2,3}.wa_model.ckpt.data.csv` (205 KB), the outputs of a weight-averaging
code path that is not part of the released pipeline and produces no published number.

## Verification performed before committing

- **Table 1 reproduces exactly.** Mean-pooling logits per patient, `argmax`, then the
  project's Challenge Score and scikit-learn metrics over the five folds reproduces all
  42 cells of Table 1 (6 architectures × 7 metrics); the largest absolute deviation from the
  printed value is 0.0005, i.e. rounding. The `±` values in Table 1 are **population**
  standard deviations over the five folds (`ddof = 0`); using the sample s.d. gives values
  about 12 % larger and will not match the paper.
- **Table S10 reproduces exactly** at patient level from the six single-split files:
  baseline 0.873 / 0.678 / 0.907 / 0.316, single-location 0.904 / 0.661 / 0.945 / 0.125,
  10 s 0.852 / 0.588 / 0.891 / 0.125, 15 s 0.857 / 0.561 / 0.905 / 0.000, no filter
  0.857 / 0.558 / 0.902 / 0.000, 25–1000 Hz 0.857 / 0.557 / 0.899 / 0.000.
- **The class-weight sweep reproduces exactly**: `[4,10,1]` 0.918 / 0.683 / 0.300,
  `[2,5,1]` 0.926 / 0.632 / 0.125, `[6,15,1]` 0.906 / 0.642 / 0.235.
- Every `filename` parses as `{digits}_{letters}[-{digits}].wav`; 0 unparsed rows in 46,375.

Two things a reproducer needs to know, and which the paper does not state:

1. The Table S10 baseline row and the class-weight sweep's `[4,10,1]` reference are **two
   different training runs** of the same configuration (TinyViT, fold 0):
   `tinyvit_3class.data.csv` gives Challenge Score 0.907 and is the S10 baseline;
   `tinyvit_3class_fold0.data.csv` gives 0.918 and is the sweep reference. Both are correct
   as published; they are simply not the same run.
2. `ablation_single_location.data.csv` covers **187 patients, not the 189 of fold 0**, and
   those 187 patients are spread across all five folds of the main partition
   (56 / 46 / 27 / 21 / 37). Keeping one recording per patient drops the patients whose
   chosen location has no matching audio file, which shortens the patient list that
   `random.Random(42).shuffle` is applied to and therefore re-derives a *different* five-fold
   partition. The single-location row of Table S10 is consequently evaluated on a different
   held-out patient set from the other rows. See `docs/errata.md`.
