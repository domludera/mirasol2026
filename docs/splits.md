# Cross-validation folds

The paper promises the data splits, and this page plus `data/splits/folds.csv` is that
promise. Read the first section before trying to reconstruct the folds from the
supplement's description, which is incomplete.

## Use the shipped file

```
data/splits/folds.csv        941 rows + header,  patient_id,fold
```

That file is **ground truth** for every number in the paper. It is the partition the
published runs actually used, recovered from and cross-checked against the patient sets in
`results/predictions/`. If you want to reproduce a per-fold number, join on this file
rather than rerunning the recipe.

## The recipe, with the step the supplement omits

The supplement says:

> Patients are assigned to five folds by deterministic round-robin over sorted patient
> identifiers with seed 42, so the splits are reconstructible without shipping index files.

The implementation has **three** steps, not two:

```python
patients = sorted(patient_map.keys())   # 1. sort the surviving patient IDs (as strings)
rng = random.Random(42)                 # 2. seed a fresh Mersenne Twister
rng.shuffle(patients)                   #    and shuffle
for i, pid in enumerate(patients):      # 3. round-robin
    folds[i % 5].extend(patient_map[pid])
```

Verified: this recipe reproduces the published partition exactly, with **0 mismatches**
against the patient sets in `results/predictions/`. Following the prose literally - sort,
then round-robin, with no shuffle - places **742 of the 941** patients in the wrong fold.
This is recorded in `docs/errata.md` D1.

Three further details matter if you insist on regenerating rather than reading the CSV:

1. **The shuffle depends on CPython's `random` module.** `random.Random(42).shuffle` uses
   the Mersenne Twister and Python's own Fisher-Yates loop. Any other implementation - a
   different Python runtime, `numpy.random.shuffle`, a future change to CPython's
   `shuffle` - gives a different permutation from the same seed. This is exactly why the
   CSV ships.
2. **The IDs are sorted as strings, not integers.** `sorted()` on `["9979", "10000"]`
   yields `["10000", "9979"]`. Sorting numerically changes the permutation.
3. **The list being shuffled is the surviving patients, not the metadata rows.** It has 941
   entries, not the 942 in `training_data.csv` - see below. Shuffling 942 IDs gives a
   different partition even with the same seed.

`mirasol.data.splits` implements the recipe, emits `data/splits/folds.csv`, and
`tests/test_splits.py` asserts that the recipe output, the committed CSV, and the patient
partition implied by `results/predictions/` all agree, with zero mismatches.

## Fold sizes

Recomputed from `results/predictions/`:

| Fold | Patients | Segments (3-class, 30 s) |
|---|---|---|
| 0 | 189 | 739 |
| 1 | 188 | 747 |
| 2 | 188 | 756 |
| 3 | 188 | 753 |
| 4 | 188 | 726 |
| **total** | **941** | **3,721** |

Patient counts are 189 / 188 / 188 / 188 / 188 because 941 is not divisible by five and
round-robin gives the remainder to fold 0. Segment counts differ per fold because patients
contribute a variable number of location-specific instances after multi-location
restructuring (Sec. 2.3) and because longer recordings yield more than one 30 s window.

Patient-level class composition over all five folds: 179 Present, 68 Unknown, 694 Absent.

## Why 941 patients and not 942

The public partition has 942 patients and 3,163 recordings. Every number in the paper is
over 941 patients.

Recordings are matched by filename as `{patient_id}_{location}.wav`, with location in
`AV`, `MV`, `PV`, `TV`, `Phc`. Some recordings use a duplicate-location form instead -
`50321_AV_1`, `50321_AV_2`, `50321_MV_1`, `50321_MV_2` - which that pattern does not match.
**43 of the 3,163 recordings, spread over 17 patients, are skipped this way**, leaving
3,120. For 16 of those patients enough recordings survive that the patient stays in the
cohort; **patient 50321 loses all four of its recordings and therefore disappears
entirely**, which is the one patient the supplement says was "dropped during dataset
preparation".

The 17 affected patients and their skipped-file counts: 49748 (4), 49952 (2), 50032 (2),
50115 (4), 50204 (2), 50210 (2), 50216 (2), 50249 (2), 50250 (2), 50263 (2), 50321 (4),
50348 (2), 50656 (2), 50690 (2), 50743 (3), 50782 (4), 50802 (2).

The excluded patient is an *Absent* one: the paper reports 179 Present / 68 Unknown /
695 Absent over 942, and the committed predictions give 179 / 68 / 694 over 941.

This behaviour is preserved exactly - it is what produced the published cohort - but it is
no longer silent. `mirasol.data.prepare` additionally globs `{pid}_{loc}*.wav` for
diagnostics and prints every unmatched file and every fully-dropped patient at the end of
each run.

## Both seeds

| Seed | Value | Governs | Stated in the paper? |
|---|---|---|---|
| Split seed | **42** | `random.Random(42).shuffle` over the patient list | yes |
| Training seed | **123777** | `pl.seed_everything`: weight init, SpecAugment, Mixup, the class-balanced sampling queue | no |

`mirasol.config.defaults` carries a comment at each so the two cannot be confused.

## Runs are not bitwise reproducible

`pl.Trainer` was **not** constructed with `deterministic=True`, and the class-balanced
sampler redraws its queue - with replacement - at the end of every epoch. Retraining with
the same seeds on the same data therefore reproduces per-fold metrics to roughly a
percentage point, not bitwise. The five-fold means in Table 1 are correspondingly stable;
individual fold numbers are not.

Two consequences worth planning for:

- The two "baseline" TinyViT fold-0 runs in the supplement differ for exactly this reason
  (`docs/errata.md` F1): Challenge Score 0.907 for one, 0.918 for the other, same
  configuration.
- If you retrain, compare five-fold means against Table 1, not single folds. To reproduce a
  single published fold number exactly, use the committed predictions.

## No leakage across folds

All recordings from one patient are assigned to the same fold, so no patient appears in
both the training and the evaluation half of any fold. Two independent checks hold on the
shipped artifacts:

- the five patient sets in `results/predictions/*_3class_fold{0..4}.data.csv` are pairwise
  disjoint and their union has exactly 941 members;
- `data/splits/folds.csv` assigns each of those 941 patients to exactly one fold, matching
  the file it appears in.

Both are asserted in `tests/test_splits.py`.

## One exception: the single-location ablation

The ablation condition that removes multi-location restructuring uses a **different**
partition, because it changes the surviving patient list and the split is a shuffle over
that list. It evaluates 187 patients, of which 56 are in the main fold 0. See
`docs/errata.md` F2 before comparing that row to the baseline.
