# Real-data benchmark: where the metrics actually land

**Protocol update (2026-09-05).** New runs use the full shared background for
SHAP and LIME and save per-run SHAP context. The checked-in JSON and tables below
predate this change: TreeSHAP used training-path counts, while LinearSHAP could
subsample the 200-row reference internally. These historical numbers do not
validate the updated pipeline and must be regenerated after the remaining
correctness and benchmark-protocol fixes.

New benchmark output also marks method/subgroup diagnostics as `descriptive`,
with no active quality thresholds. The historical verdict column below reflects
the old report policy and is not the current classification of these diagnostics.

**Split/preprocessing update (2026-09-06).** New runs keep Adult's official
training/test files separate. Its holdout is fixed across seeds; seeds still
vary the model and explanation sampling. Diabetes uses `GroupShuffleSplit` to
hold out 30% of unique patients per seed (the row proportion may differ), with
no patient shared between training and test. This is a patient holdout, not a
temporal validation protocol.

Numeric medians, means, standard deviations and categorical vocabularies are
fitted only on training rows. All-missing training numeric columns use zero;
constant training columns retain unit scale;
unseen test categories map to an all-zero block under the existing drop-first
encoding convention. Models and explanation backgrounds use training rows;
instance and subgroup diagnostics use test rows. All models share the same
prepared split for a given dataset/seed. Each exported run context records the
protocol, row counts, encoded dimension, split-index hashes and (for Diabetes)
patient counts and overlap. Top-level dataset dimensions now describe raw
features; encoded dimensions can vary between patient splits.

The tables and checked-in JSON below used pooled preprocessing and random row
splits, including recombination of Adult's official files. They remain historical
results and cannot be used as evidence for the corrected protocol. A full rerun
and comparison are still required before updating empirical conclusions.

**Refresh-output update (2026-09-21).** The runner now validates the frozen
configuration and historical baseline before loading data, then verifies the
source-archive hashes before model fitting. It preserves
`experiments/benchmark_results.json` and publishes a
completed refresh under `experiments/results/real/` only after every run has
finished:

- `raw_runs.json` (schema 1) contains one record per dataset/model/seed, with
  unrounded aggregate metrics, `metric_counts`, feature names, background
  positions within the training partition, and the split/SHAP context.
- `summary.json` (schema 1) contains the frozen config, budgets, aggregation,
  dataset metadata, run counts, and summaries derived from the same run records.
- `environment.json` (schema 1) contains commit/worktree state, command, Python,
  platform, installed packages, runner/config hashes, and verified source hashes.
- `change_report.md` applies the pre-run comparison tolerance to Adult, Diabetes,
  and pooled medians. A material-change flag requires explanation; it is not an
  automatic regression failure.

Within `raw_runs.json`:

- `runs[].samples`: positions within the test partition, perturbation seeds,
  per-instance metric values, SHAP values, LIME coefficients and contributions.
  Sensitivity and stability now cover every explained instance, with the same
  per-instance budget. Each sensitivity estimate uses outer seed + position within the sampled batch.
- `runs[].stability_runs`: a list of records, one per explained instance, each
  containing five LIME contribution vectors and their actual seeds (outer seed
  through outer seed + 4). LIME repeats the full explanation batch at the same
  budget and reuses the displayed contribution matrix as its first repeat.
  `evaluation_counts` records actual sample coverage and LIME calls per run.
- `runs[].subgroups`: sampled test positions, segmentation feature and cutpoints,
  memberships, group sizes and mean absolute attribution vectors. The first
  sorted group is the flip-rate reference. These are group-level diagnostics,
  not additional independent per-instance measurements.
Budget and aggregation metadata live in `summary.json`.

Each raw metric is `{value, status}`. Finite values retain full precision;
`nan`, `positive_infinity`, `negative_infinity` and `not_computed` retain their
status with a JSON `null` value. The file is strict JSON, without NaN/Infinity
literals. Within-run means omit NaN but preserve infinities, so an infinite
infidelity failure is no longer silently removed. Per-metric counts separately
record total, finite, NaN and each infinity. Sample counts for sensitivity and
stability equal the actual number of explained samples; subgroup counts refer
to one aggregate diagnostic.

**Coverage update (2026-09-06).** Schema 2 evaluated sensitivity and stability
only for the first instance; schema 3 averages their per-instance values over
all explained instances. This changes the meaning of run aggregates and the
shape of `stability_runs` (single object to list). Each instance still uses six
sensitivity perturbations and five LIME repeats; five full-batch LIME calls
include the displayed explanation, with no extra call for its first repeat.
The default explanation sample count remains four: this fixes uneven coverage
but does not establish dataset-wide representativeness. Historical results
must be rerun before comparison under the new protocol.

**Sampling update (2026-09-06).** Explained instances are now uniformly sampled
without replacement from the test partition using the outer seed, instead of
always taking its first rows. Sampling occurs once per dataset/seed so all model
families receive the same instances; a dedicated RNG keeps the draw independent
of background size and model iteration. Run `sampling` metadata records the
method, seed, requested/available/actual counts and selected test positions.
`sample_position` is the row's position within the explanation batch;
`test_position` identifies its position in the full test partition. Both sample
metrics and LIME repeat records carry this mapping. Perturbation seeds use the
batch position, while subgroup positions continue to refer to the full test set.

Run the frozen explanation sample count:

```bash
python -m experiments.benchmark_real_data --n-explain 4
```

The frozen refresh count is four. Counts must be positive integers and are
validated before data loading; the canonical runner rejects a value that differs
from `experiments/config.json`.
Sampling units are test rows (encounters for Diabetes), not unique patients;
multiple encounters from a held-out patient may be sampled. Random selection
removes fixed-prefix selection but does not guarantee class balance or establish
patient-level independence. Historical fixed-prefix results remain pending a
rerun under this recorded sampling protocol.

Summary P10/median/P90 describe **finite run aggregates**, with total/finite and
non-finite run counts both pooled and per dataset. They are not confidence
intervals; runs sharing patients, holdouts or model fits must not be treated as
independent samples. Summaries can be recomputed from `runs[].metrics`, and run
means/counts from the per-instance and subgroup records. Split hashes plus the
protocol/seed identify the partition; row positions refer to that partition and
require the same source data/order. Patient identifiers are not exported.

A companion to the synthetic calibration. It runs the full trust-metric battery
on **two real UCI datasets** — Adult Income (48k × 84 features) and Diabetes
130-US (102k × 62 features) — across three model families (Random Forest,
Gradient Boosting, Logistic Regression) × 8 seeds, with numeric features
z-scored. It answers two questions the synthetic study cannot:

1. **Do the report's hand-picked defaults match reality?**
2. **Are the metrics' typical values transferable across datasets?**

Reproduce with `python -m experiments.benchmark_real_data --n-explain 4`
(data auto-downloads into `experiments/data/`, which is gitignored).

## Results (pooled over 2 datasets × 3 models × 8 seeds)

| Metric | dir | Adult | Diabetes | pooled | P10 | P90 | default (good/warn) | verdict@median |
|---|---|---|---|---|---|---|---|---|
| removal-effect correlation | ↑ | 0.427 | 0.479 | 0.454 | 0.374 | 0.519 | 0.5 / 0.2 | warn |
| comprehensiveness ratio | ↑ | 63.8 | 10.6 | 28.5 | 7.2 | 522.9 | `> 1` gate | good |
| LIME infidelity (normalized) | ↓ | 0.748 | 0.725 | 0.726 | 0.000 | 1.149 | 0.5 / 1.0 | warn |
| max-sensitivity | ↓ | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | 0.5 / 2.0 | good |
| run-to-run rank stability (all-d) | ↑ | 0.926 | 0.910 | 0.921 | 0.878 | 1.000 | 0.9 / 0.7 | good |
| run-to-run rank stability (**top-k**) | ↑ | 1.000 | 1.000 | 1.000 | 0.655 | 1.000 | 0.9 / 0.7 | good |
| run-to-run sign stability | ↑ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9 / 0.7 | good |
| SHAP vs LIME sign disagreement | ↓ | 0.319 | 0.325 | 0.325 | 0.000 | 0.394 | 0.2 / 0.5 | warn |
| SHAP vs LIME rank agreement (all-d) | ↑ | 0.804 | 0.867 | 0.843 | 0.606 | 0.977 | 0.7 / 0.4 | good |
| SHAP vs LIME rank agreement (**top-k**) | ↑ | 0.688 | 0.688 | 0.688 | 0.348 | 1.000 | 0.7 / 0.4 | warn |
| SHAP vs LIME top-k overlap | ↑ | 0.750 | 0.833 | 0.792 | 0.642 | 1.000 | 0.66 / 0.33 | good |
| SHAP vs LIME magnitude disagreement | ↓ | 0.494 | 0.516 | 0.494 | 0.035 | 0.687 | 1.0 / 1.5 | good |
| cross-segment rank stability | ↑ | 0.984 | 0.991 | 0.989 | 0.936 | 0.999 | 0.7 / 0.4 | good |
| top-k flip rate | ↓ | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.34 / 0.67 | **bad** |

(*"verdict@median" = what the current report default would say about the pooled
median. ↑ higher-is-better, ↓ lower-is-better.*)

## Findings

1. **Correcting LIME's feature units changes the rank results materially.** The
   pooled all-feature run stability is 0.92 rather than the previous 0.46, and
   SHAP–LIME rank agreement is 0.84 rather than 0.46. The earlier values mixed
   standardized LIME coefficients with original-coordinate perturbations.

2. **Some defaults remain too loose or saturating.** Comprehensiveness explodes on
   high-dimensional data (median 28.5, P90 523) because "remove 3 random
   features" does almost nothing when 80 features are noise — so the default
   `> 1` gate is trivially met and the ratio is meaningless as a *graded* number.
   Max-sensitivity defaults (0.5 / 2.0) are far looser than observed values
   (pooled median 0.0, P90 0.032 in standardized neighbourhoods).

3. **Transferability is partial.** Removal-effect correlation (0.43 vs 0.48),
   sign stability, sign disagreement, cross-segment rank stability, and
   top-k overlap transfer well across the two datasets. But comprehensiveness
   (64 vs 11) and top-k flip rate (1.0 vs 0.0) are strongly dataset-specific — a
   single threshold cannot serve both.

4. **Scale handling must be explicit.** This benchmark standardizes numeric
   features before fitting. The library now converts LIME slopes back to original
   feature units and defines sensitivity neighbourhoods in background-standardized
   coordinates, so heterogeneous raw scales no longer silently corrupt those
   two calculations.

## Resolution

The implementation changes motivated by these findings are:

* **Rank stability/agreement.** `cross_run_stability` and
  `explainer_disagreement` return a **`topk_rank_corr`** (Spearman over the
  top-k features only). The metric function returns both; `report.py` scores the
  top-k result while preserving the full result for inspection.
  Here the top-k versions read **1.00** (stability) and **0.69** (agreement).

* **Comprehensiveness** is now a `> 1` "not noise" **gate** in `report.py`,
  not a graded score (its absolute size saturates on high-dimensional data).

* **LIME infidelity** is now **normalized** by the mean squared model-output
  change, making it a scale-free fraction (≈1 = "no better than
  predicting zero change"). In this rerun its medians are similar across the two
  datasets (Adult 0.75 vs Diabetes 0.72).

Remaining known limitation:

* **Top-k flip rate is coarse and dataset-specific** (1.0 on Adult vs 0.0 on
  Diabetes — with three segments it can only be 0 / 0.5 / 1.0). It is kept as a
  coarse "does the story flip across segments" signal, not a finely-gradable
  score; the thresholds treat 0 as good and 1 as bad.
