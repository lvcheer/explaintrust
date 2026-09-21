# Reproducible benchmark refresh protocol

Status: frozen for the benchmark refresh started on 2026-09-20.

## Purpose

This refresh replaces the checked-in synthetic and real-data results that were
generated before the current explanation and split protocols. Its purpose is to
produce one traceable source for public tables, figures, and findings. It does
not establish that an explanation is correct or certify a model as trustworthy.

The pre-refresh source revision is
`02f5ee541e647940072649d11ed2d09002ee1fca`. Checksums for the checked-in
result artifacts at that revision are recorded in `baseline_manifest.json`.

## Frozen scope

The machine-readable companion to this document is `config.json` (schema 2).
It records the runner constants and metric defaults audited on 2026-09-20. The
runners do not yet load that file; adding a drift check is a later engineering
step and must not silently alter this frozen protocol. The refresh uses the code
paths and metric definitions already present at the source revision above.
During the run:

- do not add explainers, datasets, models, or metrics;
- do not change metric definitions, thresholds, or perturbation budgets;
- do not tune parameters after inspecting refreshed results;
- preserve non-finite and unavailable values with an explicit status;
- treat method-disagreement and subgroup metrics as descriptive diagnostics;
- stop and revise this protocol before continuing if a correctness defect is
  found. Record the change and restart all affected runs.

## Synthetic calibration experiment

The entry point is `python -m experiments.calibrate_thresholds`.

- Dataset: three repository-defined engineered regimes: clean (the usual x0/x2
  relationship is broken by replacing x2 with independent noise), collinear
  (the native x0/x2 proxy relationship), and heterogeneous (four features with
  an x0-by-x1 interaction).
- Seeds: 0 through 19. Seeds 0--9 fit boundaries; seeds 10--19 are held out.
- Rows per generated dataset: 900.
- Split: unstratified 70/30 `train_test_split`, keyed by the outer seed.
- Models: `RandomForestClassifier(n_estimators=60, max_depth=4,
  random_state=seed)` for all regimes except LIME infidelity, which uses
  `LogisticRegression(max_iter=1000)`.
- Explanations: four test instances per seed, top-k of three, and a 100-row
  training reference shared by SHAP and LIME where they are compared. Both the
  explained instances and background are the first rows of their partitions.
- SHAP: interventional TreeSHAP when an explicit background is supplied.
- Prediction target: the scalar model output selected by `scalar_predictor`;
  SHAP, LIME, and perturbation diagnostics use the same output within a run.

Frozen synthetic budgets:

| Component | Budget or rule |
|---|---|
| LIME for infidelity and method disagreement | 1,000 samples |
| LIME smooth sensitivity condition | 2,000 samples |
| LIME stability conditions | 2,000 samples (nominal good), 20 samples (stress) |
| Infidelity | 200 Gaussian perturbations, normalized by output-change MSE |
| Comprehensiveness | 20 random top-k-size removal sets |
| Max-sensitivity | 6 perturbations within radius 0.1 in background-standardized coordinates |
| Run-to-run stability | 5 LIME runs |
| Feature selection | top-k = 3 |

Frozen synthetic outputs comprise removal-effect correlation,
comprehensiveness ratio, normalized LIME infidelity, max-sensitivity, run rank
and sign stability, four SHAP/LIME disagreement diagnostics (sign, full-rank,
top-k overlap, and relative magnitude), and two subgroup diagnostics (rank and
top-k flip). The disagreement and subgroup values are descriptive; their
engineered contrast labels do not turn them into universal quality tests.

The experiment is a diagnostic calibration study. Only shuffled-attribution
conditions are direct corruption controls; other nominal good/bad labels are
stress-test assumptions.

The runner must not overwrite the historical `calibration.json` during the
refresh. It writes `raw_runs.json`, `summary.json`, and `environment.json` under
`experiments/results/synthetic/`; the first file groups observations by seed,
model, scenario, and condition.

## Real-data benchmark

The entry point is
`python -m experiments.benchmark_real_data --n-explain 4`.

- Datasets: UCI Adult (id 2; 48,842 rows) and Diabetes 130-US Hospitals
  (id 296; 101,766 rows). The locally cached source archives are pinned in
  `config.json` by URL and SHA-256.
- Models: Random Forest (60 trees, depth 5), Gradient Boosting (60 estimators,
  depth 3), and Logistic Regression (`max_iter=1000`). All applicable model
  random states use the outer seed.
- Seeds: 0 through 7.
- Adult split: official training/test holdout, fixed across seeds.
- Diabetes split: 30% patient-group holdout by seed; no patient may occur in
  both training and test partitions.
- Preprocessing: fit on training data only and apply unchanged to test data.
- Explained rows: four test rows sampled uniformly without replacement by a
  dedicated RNG keyed by seed. All model families use the same sampled rows for
  a dataset and seed.
- Background: at most 200 training rows sampled without replacement by seed
  and shared by SHAP and LIME for a run.
- Subgroup diagnostic: at most 2,000 test rows; the segmentation feature is
  selected by the frozen balancing rule in the runner and divided at empirical
  0.33 and 0.66 quantiles.
- Explanation settings: SHAP `method="auto"` with an explicit background;
  LIME uses 1,000 samples. Stability reuses the displayed LIME batch and adds
  runs at seed offsets 1 through 4.

Frozen real-data budgets:

| Component | Budget or rule |
|---|---|
| Explained test rows | 4 per dataset/model/seed run |
| Background | at most 200 training rows |
| Subgroup sample | at most 2,000 test rows |
| Infidelity | 200 Gaussian perturbations, normalized by output-change MSE |
| Comprehensiveness | 20 random top-k-size removal sets |
| Max-sensitivity | 6 perturbations within radius 0.1 in background-standardized coordinates |
| Run-to-run stability | 5 LIME runs at outer-seed offsets 0–4 |
| Feature selection | top-k = 3 |

The real-data output includes 14 metrics: removal correlation,
comprehensiveness, normalized infidelity, sensitivity, full and top-k stability
rank, stability sign, five method-disagreement values, and two subgroup values.
Method-disagreement and subgroup values are descriptive. Run values are means
that omit NaN but preserve infinity; P10/median/P90 summarize finite
dataset/model/seed run aggregates and are not confidence intervals.

All instance-level values and run contexts must be retained.

Adult uses its official 32,561-row training file and 16,281-row test file.
Diabetes holds out 30% of the 71,518 patient groups per seed. Numeric imputation
and scaling plus categorical vocabularies are fitted only on training rows;
unseen test categories map to an all-zero encoded block. The two holdouts answer
different generalisation questions and must not be described as equivalent or
as temporal validation.

## Reproducibility and traceability requirements

Every refreshed result bundle must contain or be accompanied by:

- the full Git commit hash and a clean/dirty worktree flag;
- Python, operating-system, and installed-package information;
- the complete frozen configuration and its SHA-256 digest;
- dataset split context and hashes already emitted by the real-data runner;
- one record per dataset/model/seed run and the underlying sample records;
- explicit counts for finite, NaN, positive-infinity, negative-infinity, and
  not-computed values where applicable.

Two executions from clean environments at the same commit and configuration
must reproduce deterministic selections and key metrics. Any unavoidable
numeric tolerance must be declared before comparing the runs.

## Acceptance criteria

The refresh is complete only when:

1. synthetic and real-data outputs were generated under this frozen protocol;
2. each public summary number can be traced to raw run records;
3. a change report explains differences from the baseline rather than silently
   replacing it;
4. README, experiment documentation, article data/figures, and demo examples
   are generated or checked against the same summary source;
5. the full test suite passes and a clean-environment reproduction succeeds.
