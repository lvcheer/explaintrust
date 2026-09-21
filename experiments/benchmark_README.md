# Real-data benchmark: where the metrics actually land

**Reviewed refresh (2026-09-21).** The generated table below comes from the
completed bundle under `experiments/results/real/`. These runs use the full
shared background for SHAP and LIME, save per-run SHAP context, and mark
method/subgroup diagnostics as `descriptive` with no active quality thresholds.
The top-level `experiments/benchmark_results.json` remains the immutable
historical comparison baseline.

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

The preserved historical baseline used pooled preprocessing and random row
splits, including recombination of Adult's official files. The refreshed bundle
uses the corrected protocol above. Because several protocol changes occurred
together and the historical file has no raw runs, the two bundles are not
directly comparable at the run level.

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
status with a JSON `null` value. A `not_computed` subgroup value also carries
structured error metadata. The file is strict JSON, without NaN/Infinity
literals. Within-run means omit NaN but preserve infinities, so an infinite
infidelity failure is no longer silently removed. Per-metric counts separately
record total, finite, NaN, each infinity, and not-computed observations. Sample
counts for sensitivity and stability equal the actual number of explained
samples; subgroup counts refer to one aggregate diagnostic.

**Coverage update (2026-09-06).** Schema 2 evaluated sensitivity and stability
only for the first instance; schema 3 averages their per-instance values over
all explained instances. This changes the meaning of run aggregates and the
shape of `stability_runs` (single object to list). Each instance still uses six
sensitivity perturbations and five LIME repeats; five full-batch LIME calls
include the displayed explanation, with no extra call for its first repeat.
The default explanation sample count remains four: this fixes uneven coverage
but does not establish dataset-wide representativeness. The reviewed comparison
and its limitations are recorded in `experiments/results/real/change_report.md`.

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
patient-level independence. The fixed-prefix values remain only in the
historical baseline; the refreshed bundle uses the recorded sampled positions.

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

<!-- BEGIN AUTO:real-results-table -->
| Metric | dir | Adult | Diabetes | pooled | P10 | P90 | finite runs | default (good/warn) | verdict@median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| removal-effect correlation | ↑ | 0.4497 | 0.4763 | 0.4636 | 0.395 | 0.5248 | 48/48 | 0.5 / 0.2 | warn |
| comprehensiveness ratio | ↑ | 45.5251 | 13.6115 | 20.6734 | 8.8529 | 8.112e+09 | 48/48 | 1 / 1 | good |
| LIME infidelity (normalized) | ↓ | 0.7353 | 0.656 | 0.683 | 0 | 0.9901 | 48/48 | 0.5 / 1 | warn |
| max-sensitivity | ↓ | 0.0001 | 0 | 0 | 0 | 0.0424 | 48/48 | 0.5 / 2 | good |
| run-to-run rank stability (all-d) | ↑ | 0.9215 | 0.9119 | 0.9186 | 0.8894 | 0.9999 | 48/48 | 0.9 / 0.7 | good |
| run-to-run rank stability (top-k) | ↑ | 0.8562 | 0.875 | 0.875 | 0.6962 | 1 | 48/48 | 0.9 / 0.7 | warn |
| run-to-run sign stability | ↑ | 1 | 1 | 1 | 0.8917 | 1 | 48/48 | 0.9 / 0.7 | good |
| SHAP vs LIME sign disagreement | — | 0.293 | 0.2748 | 0.2809 | 0 | 0.3536 | 48/48 | not scored | descriptive |
| SHAP vs LIME rank agreement (all-d) | — | 0.8796 | 0.8857 | 0.8796 | 0.6173 | 0.9999 | 48/48 | not scored | descriptive |
| SHAP vs LIME rank agreement (top-k) | — | 0.75 | 0.75 | 0.75 | 0.375 | 1 | 48/48 | not scored | descriptive |
| SHAP vs LIME top-k overlap | — | 0.8333 | 0.8333 | 0.8333 | 0.6667 | 1 | 48/48 | not scored | descriptive |
| SHAP vs LIME magnitude disagreement | — | 0.3891 | 0.4384 | 0.4123 | 0.0041 | 0.5882 | 48/48 | not scored | descriptive |
| cross-segment rank stability | — | 0.954 | 0.9874 | 0.9794 | 0.9485 | 0.9928 | 47/48 (1 n/c) | not scored | descriptive |
| top-k flip rate | — | 1 | 0 | 0.5 | 0 | 1 | 47/48 (1 n/c) | not scored | descriptive |

`n/c` means an explicitly not-computed run; descriptive metrics are not scored.
<!-- END AUTO:real-results-table -->

(*"verdict@median" = what the current report default would say about the pooled
median. ↑ higher-is-better, ↓ lower-is-better.*)

## Findings

1. **Expanded coverage exposes top-k instability.** Full-rank stability remains
   near its historical level, but pooled top-k stability is now 0.88 rather than
   1.00. Recomputing the refreshed raw runs with only each run's first explained
   instance restores a median of 1.00, so the material change is consistent with
   evaluating all four sampled instances instead of only the first.

2. **Comprehensiveness is heavy-tailed, not a graded effect size.** Its pooled
   median is 20.67 and P90 is 8.112e+09. Five sample-level ratios exceed `1e9`
   because the random-removal denominator is zero or nearly zero. The `> 1`
   default remains only a "not noise" gate; the absolute ratio should not rank
   datasets or explanation quality. Max-sensitivity remains much smaller than
   its defaults, with pooled median 0 and P90 0.0424.

3. **Transferability is partial and descriptive diagnostics differ by dataset.**
   Removal-effect correlation is similar on Adult and Diabetes (0.45 vs 0.48),
   while comprehensiveness differs (45.53 vs 13.61) and top-k flip rate is 1.0
   vs 0.0. The latter is a coarse descriptive subgroup diagnostic, not evidence
   for a universal quality threshold. One Adult subgroup run is explicitly
   not computed and remains counted rather than silently discarded.

4. **The refresh is not directly comparable run by run.** Training-only
   preprocessing, corrected holdouts, sampled explanation rows, explicit
   interventional backgrounds, and four-instance stability coverage all differ
   from the historical protocol. Twelve of 42 prespecified comparisons are
   flagged as material changes, but the historical file lacks raw runs needed
   to isolate causes or claim improved explanation quality.

## Resolution

The implementation changes motivated by these findings are:

* **Rank stability/agreement.** `cross_run_stability` and
  `explainer_disagreement` return a **`topk_rank_corr`** (Spearman over the
  top-k features only). The metric function returns both; `report.py` scores the
  top-k stability result while preserving full-rank stability for inspection;
  method agreement remains descriptive. Here the pooled top-k values are
  **0.88** (stability) and **0.75** (agreement).

* **Comprehensiveness** is now a `> 1` "not noise" **gate** in `report.py`,
  not a graded score (its absolute size saturates on high-dimensional data).

* **LIME infidelity** is now **normalized** by the mean squared model-output
  change, making it a scale-free fraction (≈1 = "no better than
  predicting zero change"). In this rerun its medians are similar across the two
  datasets (Adult 0.74 vs Diabetes 0.66).

Remaining known limitation:

* **Top-k flip rate is coarse and dataset-specific** (1.0 on Adult vs 0.0 on
  Diabetes — with three segments it can only be 0 / 0.5 / 1.0). It is kept as a
  descriptive "does the story flip across segments" signal, not a scored or
  finely gradable quality measure.
