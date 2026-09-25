# `data/splits/` — the five-fold patient partition

## Licence and attribution

**This directory is not covered by the repository's MIT licence.** `folds.csv` lists CirCor
DigiScope patient identifiers, so it is a derived database:

> Contains information from the CirCor DigiScope Phonocardiogram Dataset made available
> under the Open Data Commons Attribution License (ODC-By 1.0).

It contains **no labels and no demographics** — only CirCor's public pseudonymous patient IDs
and our own fold index. Cite Oliveira et al. (2022), doi:10.1109/JBHI.2021.3137048, and
Reyna et al. (2023), doi:10.1371/journal.pdig.0000324. See `LICENSE`, `THIRD_PARTY.md` and
`results/predictions/README.md`.

## `folds.csv`

Header row `patient_id,fold`, then 941 rows, one per retained patient. `patient_id` is the
CirCor identifier as a string; `fold` is the held-out fold index 0–4. Rows are ordered by
`patient_id` **lexicographically**, i.e. exactly as Python's `sorted()` orders these strings,
so the file can be compared to the generating recipe without re-sorting. (Three IDs are
four digits — `2530`, `9979`, `9983` — and therefore sort last, not first.)

Fold sizes: 189, 188, 188, 188, 188.

## Provenance

The partition is taken from the shipped out-of-fold predictions, which are ground truth:
each `results/predictions/{model}_{task}_fold{k}.data.csv` is one held-out fold, and all ten
five-fold families (six architectures three-class, four architectures two-class) imply
**exactly the same** patient partition — 941 patients, 0 assigned differently, folds disjoint
and exhaustive.

`mirasol.data.splits` regenerates the same table from the recipe in
`mirasol/data/prepare.py`:

```python
patients = sorted(patient_map.keys())   # lexicographic, on the ID strings
rng = random.Random(42)
rng.shuffle(patients)                   # <- CPython Mersenne Twister
folds[i % 5].append(patients[i])
```

Cross-check of recipe against the prediction-derived ground truth, over all 941 patients:

| Reconstruction | Agreements | Disagreements |
|---|---|---|
| `sorted()` on the ID strings → `Random(42).shuffle` → `i % 5` | **941** | **0** |
| numeric sort (`sorted(ids, key=int)`) → `Random(42).shuffle` → `i % 5` | 188 | 753 |
| sorted round-robin with the shuffle omitted | 199 | 742 |

Two properties of that recipe are load-bearing and are easy to get wrong:

1. **The shuffle is required.** A sorted round-robin with no shuffle reproduces only 199 of
   941 assignments.
2. **The sort must be lexicographic on the ID strings.** Sorting the same IDs numerically
   changes the input order of `shuffle` and reproduces only 188 of 941 assignments.

Because step 2 depends on `random.Random(42).shuffle`, the partition is reproducible only on
a CPython Mersenne Twister; `folds.csv` is shipped so that reproduction never depends on that.
`tests/test_splits.py` asserts the CSV against both the recipe and the partition implied by
`results/predictions/`. Full discussion in `docs/splits.md`.

## Why 941 and not 942

CirCor's public training partition, as distributed in `training_data.csv`, holds 942 patients
and 3,163 recording entries, labelled 179 *Present*, 68 *Unknown*, 695 *Absent*. One *Absent*
patient, `50321`, is dropped during dataset preparation: it appears in no fold and in none of
the 61 prediction dumps, and the retained label counts are 179 / 68 / 694. Its metadata row is
the one row whose `Recording locations:` field repeats a location (`AV+AV+MV+MV`), so it is
also the row for which the `{patient}_{location}.wav` naming convention used by
`mirasol/data/prepare.py` cannot address its recordings one-to-one. Recording-level skips of
metadata entries with no matching audio file do not remove their patients. See
`docs/data.md`.

## Scope

This partition applies to the three-class and two-class runs and to the window-length and
bandpass ablations. It does **not** apply to the single-location ablation: keeping one
recording per patient drops the patients whose selected location has no audio file, which
shortens the list passed to `shuffle` and re-derives a different partition — the 187 patients
in `results/predictions/ablation_single_location.data.csv` are spread across all five folds
here (56 / 46 / 27 / 21 / 37). See `docs/errata.md`.
