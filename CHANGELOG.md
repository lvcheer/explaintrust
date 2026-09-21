# Changelog

All notable changes are recorded here. This project follows
[Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) structure.

## [Unreleased]

### Fixed

- Added classification/regression performance metrics, confusion matrices and ROC
  curves, model-specific controls including an MLP, configurable quality-analysis
  budgets, and a Chinese/English presentation switch that preserves training state.

- The app now uses an explicit five-stage workflow with downstream invalidation,
  synthetic/real/uploaded data selection, feature dictionaries, configurable
  splits and models, and SHAP/LIME selection. Quality analysis reuses the displayed
  explanation. Added optional XGBoost support and direct raw-margin prediction
  for class-specific SHAP/LIME output alignment.

- Benchmarks now select explained test rows by reproducible uniform sampling
  without replacement, shared across models for each dataset/seed. The new
  `--n-explain` option validates the requested count before loading data. Exports
  retain requested/actual counts and batch-to-test position mappings, including
  when the request exceeds the holdout size.

- Benchmark sensitivity and LIME repeat stability now cover every explained
  instance. Repeats use the same full batch and budget, reusing the displayed
  explanation as the first repeat. Schema 3 exports per-instance repeats,
  actual seed sequences and evaluation counts; run means cover all instances.

- Benchmark JSON now retains per-instance metrics and attribution vectors,
  per-seed run aggregates, LIME repeat vectors and subgroup provenance, with
  explicit budgets, scope and finite/missing/infinite counts. Infinite failures
  survive run aggregation; strict JSON preserves non-finite status separately
  from null values. Summary percentiles remain finite-run descriptive statistics.

- Real-data benchmarks now preserve Adult's official holdout and split Diabetes
  by patient before fitting imputation, scaling or categorical vocabulary.
  Per-run exports record split provenance and patient overlap; dataset loaders
  now return raw features and split metadata instead of globally encoded arrays.
  The historical benchmark remains immutable; the reviewed refreshed bundle,
  change report and generated public table now document the corrected protocol.

- Reports now separate scored faithfulness/sensitivity/reproducibility checks
  from descriptive method disagreement and subgroup heterogeneity. Descriptive
  values, missingness and inapplicability remain visible without determining
  the overall verdict. Exports identify roles and the decision policy; retired
  descriptive threshold overrides are rejected. Multiple scored failures now
  produce `CHECKS FAILED` rather than a universal `UNRELIABLE` label.
- SHAP now honors the complete supplied background: tree calls use
  interventional mode and linear calls disable implicit reference subsampling.
  Tree calls without a background retain training-path behavior; contradictory
  explicit mode/background combinations are rejected. Optional returned context
  and app exports record the actual reference, mode and selected-class base value.
  Experiment and article generators use the aligned protocol. Reviewed synthetic,
  real-data and article result bundles now drive their public tables, numerical
  claims and figures through a write/check consistency command; historical
  top-level experiment JSON files remain preserved comparison baselines.
- Binary `class_index=0` now selects the class-0 decision margin and native
  single-output SHAP attributions, with matching LIME directions. Per-class
  SHAP outputs and already-selected KernelSHAP outputs are not negated again.
  Invalid class indices, multiclass models, and raw classifier labels are
  rejected explicitly; regression outputs remain unchanged.
- App stability checks now repeat LIME for every explained instance at the
  main explanation's sample budget. Per-instance spreads follow the selected
  feature table and JSON export; the report averages per-instance stability
  scores and records the covered instances, budgets, and actual seeds.
  The headless demo also uses the same LIME budget for both checks.
- Low-dimensional app analyses now select a proper feature subset (top-1 for
  two features, top-2 for three). All-feature removal/set comparisons and
  one-feature rank correlations no longer count as passing or failing evidence.
  Metric dictionaries carry `n_features`; reports accept it explicitly and
  expose `not_applicable` verdicts with null JSON values and exclusion reasons.
- Max-sensitivity rejects non-finite inputs/attributions, invalid shapes and
  ineffective perturbation budgets instead of returning a misleading zero.
  Reports distinguish infinite infidelity failures from unavailable values and
  retain known failures when other checks are missing.
- KernelSHAP now applies the requested seed to its actual sampling RNG and
  restores the caller's NumPy RNG state, including on failure. Concurrent
  KernelSHAP calls through the library are serialized to prevent RNG races.

## [0.1.0] - 2026-09-04

### Added

- Initial explainers, diagnostic metrics, trust report, Streamlit demo,
  synthetic experiments, and real-data benchmark.
- JSON report export with method context and preprocessing audit metadata.
- Configurable, validated report thresholds.
- Held-out synthetic calibration and refreshed real-data benchmark outputs.
- GitHub Actions tests for Python 3.9 and 3.12.

### Changed

- SHAP, LIME, and perturbation metrics now use a consistent model-output space.
- LIME coefficients are returned in original feature units.
- Infidelity normalization and max-sensitivity now match their documented
  baselines and formulas.
- Rank-based comparisons use attribution magnitude; subgroup consistency is no
  longer described as direct distribution-shift validation.
- Positive overall verdicts were replaced with the more limited “no issues
  detected” wording; unavailable checks produce insufficient evidence.

### Security

- Removed unsafe pickle/joblib model upload from the Streamlit application.

[Unreleased]: https://github.com/lvcheer/explaintrust/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lvcheer/explaintrust/releases/tag/v0.1.0
