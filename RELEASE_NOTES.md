# explaintrust 0.2.0

Release date: 21 September 2026

This release completes the first fully refreshed and reproducible explaintrust
evidence bundle. The code is unchanged from `0.2.0rc1`, which passed GitHub CI,
clean wheel and source-distribution installs, and public TestPyPI installation
verification before promotion.

## User-visible changes

- Trust reports now distinguish scored faithfulness, sensitivity, and
  reproducibility checks from descriptive explainer-disagreement and subgroup
  diagnostics. Passing means that no configured check failed; it is not a
  certificate of correctness, causality, or fairness.
- Reports preserve failed evidence when other checks are unavailable and expose
  good, warn, bad, unavailable, and not-applicable states with decision-policy
  metadata.
- SHAP and LIME handling now records model-output space, selected class,
  explainer background, budgets, seeds, and sample coverage. Binary
  `class_index=0` follows the class-0 decision direction consistently.
- The Streamlit application uses a five-stage bilingual workflow, and the
  repository provides a headless reference pipeline, a project homepage, and a
  two-page English technical brief.
- CI tests Python 3.9 and 3.12, checks public result consistency, builds both
  distributions, and clean-installs the wheel for an import/report smoke test.

## Experiment protocol changes

- Adult keeps its official holdout; Diabetes is split by patient before
  training-only imputation, scaling, or vocabulary fitting.
- Explained rows are sampled reproducibly without replacement and shared across
  models for each dataset and seed. Stability and sensitivity now cover every
  explained instance rather than only the first row.
- SHAP and LIME use explicit training backgrounds, and LIME repeats use the same
  batch and budget as the displayed explanation.
- Raw runs now retain per-instance metrics, attribution and repeat vectors,
  seed sequences, split provenance, missing/non-finite status, environment
  metadata, and configuration digests. Public tables, figures, and headline
  claims are checked against the canonical summaries.

## Why historical numbers are not directly comparable

The refreshed benchmark changes holdout construction, preprocessing scope,
explained-row sampling, background handling, stability coverage, and export
semantics together. The preserved historical benchmark contains no raw runs,
so these effects cannot be isolated retrospectively. Twelve of 42 prespecified
historical comparisons are materially different; these are screening flags,
not hypothesis tests or evidence that explanation quality improved.

Comprehensiveness is also heavy-tailed because the random-removal denominator
can be zero or nearly zero. Its pooled median is `20.67`, while P90 is
`8.112e+09`; use it as a not-noise gate rather than a graded effect size.

## Known limitations

- Scope is numeric tabular binary classification and regression with SHAP and
  LIME. Multiclass, image, text, LLM, survival, and counterfactual explanations
  are outside this release.
- Real-data evidence covers only Adult and Diabetes and is not universal
  validation. One Adult subgroup run is explicitly not computed.
- Default thresholds are versioned policy choices, not universally calibrated
  cutoffs. Method disagreement and subgroup heterogeneity remain descriptive.
- The software, article, and technical brief have not been peer reviewed.
- XGBoost support is optional; the core release verification does not require
  that extra.

## Installation

From the tagged repository checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/demo.py
```

Review the full change history in `CHANGELOG.md`, the reproducibility boundary
in `experiments/public_results_inventory.md`, and citation metadata in
`CITATION.cff`.
