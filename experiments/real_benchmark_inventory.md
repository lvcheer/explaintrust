# Real-data benchmark pre-run inventory

Date: 2026-09-21  
Branch: `experiment/reproducible-benchmark-refresh`  
Scope: Adult Income and Diabetes 130-US benchmark entry points, source data, protocol, historical outputs, and pre-run gaps. No benchmark was executed during this inventory.

Implementation update (2026-09-21): the output/configuration blockers recorded
below were subsequently addressed in the runner. They are retained here as the
pre-change audit trail; the full benchmark remains unexecuted.

## Current entry points

| Purpose | Entry point | Current behavior |
|---|---|---|
| Real-data benchmark | `python -m experiments.benchmark_real_data --n-explain 4` | Validates the frozen contract, loads both UCI datasets, evaluates 3 models × 8 seeds, preserves the historical JSON, writes four artifacts under `experiments/results/real/`, and prints a summary table. |
| Dataset/split diagnostic | `python -m experiments.real_datasets` | Loads both datasets, prepares one split for each, and prints dimensions and split context. It is a diagnostic helper, not a benchmark result producer. |

The frozen entry point in `experiments/config.json` is the module form shown above and is the canonical command for the refresh.

## Frozen run matrix

| Dimension | Frozen value |
|---|---|
| Datasets | Adult, Diabetes 130-US |
| Models | Random Forest, Gradient Boosting, Logistic Regression |
| Seeds | 0–7 |
| Dataset × model × seed runs | 48 |
| Explained test rows per run | 4, uniformly sampled without replacement |
| Model-run instance explanations | 192 |
| Background | Up to 200 uniformly sampled training rows |
| Subgroup sample | Up to 2,000 test rows |
| Metrics | 14 aggregate fields, with per-instance records where applicable |
| Top-k | 3 |

Sampled test positions are shared across model families for a given dataset and seed. Background positions are also reproducible and shared because the background RNG is reinitialized from the same outer seed for each model.

## Source data inventory

Both source archives are already cached under the ignored `experiments/data/` directory, so a reviewed run does not require a new download.

| Dataset | Cached archive | Verified SHA-256 | Rows | Features used by current raw loader | Current split |
|---|---|---|---:|---:|---|
| Adult | `experiments/data/adult.zip` | `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` | 48,842 | 12 | Official 32,561/16,281 train/test files; fixed across seeds |
| Diabetes 130-US | `experiments/data/diabetes130.zip` | `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a` | 101,766 | 39 | `GroupShuffleSplit`, 30% of patient groups held out per seed |

For Diabetes, `patient_nbr` is retained only as a split group and is excluded from model features. The current split context records train/test patient counts and overlap. Adult's sampling-weight and redundant education fields, plus the Diabetes identifiers and outcome-adjacent administrative fields listed in `real_datasets.py`, are excluded.

## Current leakage controls

Direct source review and the benchmark tests establish the following controls:

- Adult retains the official train/test partition instead of recombining files and resplitting rows.
- Diabetes uses patient-group holdout; a patient cannot occur in both partitions.
- Numeric imputation, means, scales, and categorical vocabularies are fitted on training rows only.
- Unseen test categories map to an all-zero block under the drop-first encoding.
- Models and explanation backgrounds use training rows; explained instances and subgroup diagnostics use test rows.
- Each exported schema-3 run is designed to include split-index hashes, patient overlap where applicable, background positions, explained test positions, raw metric statuses, SHAP context, and internal perturbation/stability seeds.

The focused benchmark suite completed with `23 passed` and no failures. This verifies the split, sampling, coverage, and non-finite serialization logic with fixtures; it is not a substitute for inspecting the future full-run artifacts.

## Historical checked-in result

`experiments/benchmark_results.json` is the only checked-in real-data result file. Its SHA-256 is:

`f27fb8aec0c522e179517a12fe9787e7e8131189af4190aa44a96e618bf0f071`

This matches `experiments/baseline_manifest.json` and the tag `baseline/pre-benchmark-refresh-2026-09-20`. The file has no `schema_version`, no raw `runs`, no `run_contexts`, no environment, and no aggregation metadata. It contains only 14 summary metrics for 48 historical runs. The old run pooled preprocessing, used random row splits, and did not use the current explanation/background protocol; its numbers are comparison baselines only.

| Historical metric | Adult median | Diabetes median | Pooled median |
|---|---:|---:|---:|
| Removal-effect correlation | 0.4271 | 0.4789 | 0.4543 |
| Comprehensiveness | 63.7952 | 10.6341 | 28.5470 |
| LIME infidelity | 0.7483 | 0.7246 | 0.7256 |
| Max sensitivity | 0.0000 | 0.0000 | 0.0000 |
| Run stability, all features | 0.9259 | 0.9096 | 0.9207 |
| Run stability, top-k | 1.0000 | 1.0000 | 1.0000 |
| Run sign stability | 1.0000 | 1.0000 | 1.0000 |
| SHAP/LIME sign disagreement | 0.3194 | 0.3249 | 0.3249 |
| SHAP/LIME rank agreement, all features | 0.8042 | 0.8665 | 0.8427 |
| SHAP/LIME rank agreement, top-k | 0.6875 | 0.6875 | 0.6875 |
| SHAP/LIME top-k overlap | 0.7500 | 0.8333 | 0.7917 |
| SHAP/LIME magnitude disagreement | 0.4940 | 0.5160 | 0.4940 |
| Cross-segment rank stability | 0.9835 | 0.9914 | 0.9890 |
| Cross-segment top-k flip rate | 1.0000 | 0.0000 | 1.0000 |

The historical medians can support a clearly labelled coarse comparison, but the absence of raw historical run values prevents paired run-level change analysis.

## Current runner output contract

The current source is designed to write schema 3 with:

- one nested run record per dataset/model/seed;
- per-instance metric values, SHAP values, LIME coefficients and contributions;
- five LIME stability repeats per explained instance;
- subgroup membership, group counts, mean absolute attributions and diagnostics;
- finite/non-finite status envelopes and counts;
- split, sampling, background and SHAP context;
- pooled and per-dataset finite-run percentile summaries.

This is substantially more traceable than the checked-in historical JSON, but it does not yet satisfy the frozen Wednesday acceptance criteria by itself.

## Pre-run blockers and gaps

### Blocking

1. **The runner overwrites the historical baseline.** It writes schema 3 directly to `experiments/benchmark_results.json`. The baseline must remain untouched and recoverable without relying only on Git history.
2. **There is no separate environment artifact.** The intended output lacks the executing commit, dirty flag, Python/platform details, dependency versions, runner/config hashes, command, generation time, and verified data-archive hashes.
3. **The output layout does not match the refresh contract.** There are no separate raw, summary, environment, and change-report files under a refreshed result directory.
4. **The comparison tolerance is not frozen.** The plan requires flagging changes beyond a predetermined tolerance, but `config.json` currently defines no absolute or relative comparison rule. Choosing it after seeing refreshed results would introduce hindsight bias.

### Important but non-blocking for execution

1. `CURRENT_DEFAULTS` manually mirrors report policy. Its numeric values currently agree with the corresponding report thresholds, but there is no automated semantic mapping check; top-k stability reuses the same stability boundary.
2. Historical `datasets` dimensions (`Adult: 48,842 × 84`, `Diabetes: 101,766 × 62`) describe the older encoded matrices, while current metadata intentionally reports raw rows/features and records encoded dimension per split. These fields must not be compared as if they had the same meaning.
3. The default of four explained rows per model run is traceable but sparse. It is the frozen protocol, not evidence of dataset-wide representativeness.
4. Summary P10/median/P90 values are descriptive run percentiles, not confidence intervals; shared data and repeated patient populations prevent treating all runs as independent observations.

## Recommended output boundary for the refresh

Preserve `experiments/benchmark_results.json` as the immutable historical baseline and write the reviewed rerun to a separate real-data namespace:

- `experiments/results/real/raw_runs.json`
- `experiments/results/real/summary.json`
- `experiments/results/real/environment.json`
- `experiments/results/real/change_report.md`

Before any full run, the comparison tolerance and the exact output contract should be added to the frozen configuration and validated by the runner. This is the next prerequisite step; the benchmark itself should remain paused until that change is reviewed.
