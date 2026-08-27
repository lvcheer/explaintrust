# Threshold calibration experiment

A controlled study that measures each report metric in regimes where we *know*
whether the explanation is trustworthy, to see (a) which metrics actually
discriminate, and (b) whether the thresholds can be fitted from data. The
result is a **diagnostic**, not a claim about real-world distributions.

## Why calibrate?

`explaintrust/report.py` maps every metric to a verdict (`good` / `warn` /
`bad`) using two thresholds per metric. Those thresholds were originally chosen
by hand — the weakest link in an otherwise defensible pipeline. This study asks
whether the data can choose better boundaries, and in doing so it exposes which
metrics *can't* be calibrated on these axes.

## Method

For each metric we construct two regimes with known ground truth, then measure
the metric's distribution in each:

| Metric | "good" regime (trustworthy) | "bad" regime (untrustworthy) |
|---|---|---|
| removal-effect correlation | real SHAP on clean data | SHAP magnitudes shuffled across features |
| comprehensiveness ratio | real SHAP on clean data | shuffled SHAP |
| LIME infidelity | real LIME on a *linear* model | shuffled LIME weights |
| max-sensitivity | smooth LIME (2000 samples) | piecewise TreeSHAP |
| run-to-run stability | LIME, 2000 samples | LIME, 20 samples |
| SHAP vs LIME disagreement | collinearity removed (ρ = 0), low-dim | high collinearity (ρ ≈ 0.9), low-dim |
| cross-segment stability | random halves (homogeneous) | split by x0 (heterogeneous interaction) |

We sweep 20 seeds, mostly with a `RandomForestClassifier(n_estimators=60,
max_depth=4)` on `n = 900`, and aggregate each metric over ~80 instance-level
samples per regime.

**Threshold rule.** Two boundaries per metric so that ~80% of known-good samples
are judged `good` and ~80% of known-bad samples are judged `bad`:

* *higher-is-better*: `good = P20(good)`, `warn = P80(bad)`
* *lower-is-better*: `good = P80(good)`, `warn = P20(bad)`

The `pass(good)`/`flag(bad)` columns report how well the fitted thresholds
recover the ground-truth labels on the same data.

## Results

| Metric | dir | med(good) | med(bad) | good | warn | pass | flag | Result |
|---|---|---|---|---|---|---|---|---|
| SHAP removal-effect correlation | ↑ | 0.616 | −0.030 | 0.426 | 0.249 | .80 | .78 | ✅ separates |
| SHAP comprehensiveness (top-k vs random) | ↑ | 2.343 | 0.476 | 1.441 | 2.033 | .80 | .74 | ⚠️ direction right, tails overlap |
| LIME local fidelity (infidelity) | ↓ | 0.002 | 1.302 | 0.004 | 0.986 | .80 | .80 | ✅ separates |
| Max sensitivity | ↓ | 0.029 | 0.135 | 0.035 | 0.074 | .80 | .80 | ✅ separates |
| Run-to-run rank stability | ↑ | 0.910 | 0.548 | 0.865 | 0.635 | .80 | .80 | ✅ separates |
| Run-to-run sign stability | ↑ | 1.000 | 0.667 | 1.000 | 1.000 | 1.00 | .65 | ⚠️ coarse |
| SHAP vs LIME sign disagreement | ↓ | 0.250 | 0.000 | 0.500 | 0.000 | .99 | .03 | ❌ inverted |
| SHAP vs LIME rank agreement | ↑ | 0.800 | 1.000 | 0.400 | 1.000 | .96 | .03 | ❌ inverted |
| SHAP vs LIME top-3 overlap | ↑ | 1.000 | 1.000 | 0.667 | 1.000 | 1.00 | .00 | ❌ saturated |
| Cross-segment rank stability | ↑ | 1.000 | 0.900 | 1.000 | 1.000 | .90 | .50 | ⚠️ weak |
| Top-3 flip rate across segments | ↓ | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 | .00 | ❌ saturated |

(*↑ = higher-is-better, ↓ = lower-is-better.*)

## Findings

1. **Four metrics separate cleanly.** Removal-effect correlation, LIME
   infidelity, max-sensitivity, and run-to-run rank stability all show a strong,
   correctly-directed signal. This *confirms* those definitions behave as
   intended.

2. **Absolute thresholds are regime-specific, so they are not auto-baked.**
   Infidelity was calibrated on a *linear* model (where LIME's weights are the
   true gradient), so its "good" bar lands near 0.004 — but a real
   random-forest LIME reads ≈ 0.1. Baking that in would flag every tree-model
   explanation as untrustworthy. The fitted values are a *scale reference*, not
   universal defaults.

3. **The disagreement metrics are blind to the collinearity disagreement —
   the study's key negative result.** Collinearity makes SHAP and LIME *split*
   the x0/x2 credit (a *magnitude* disagreement), while they still agree on the
   *set* and *ranking* of important features. So the rank/sign/top-k agreement
   metrics read **higher** under collinearity — they say "good" exactly when the
   explanation is most misleading. The metric that *would* catch this,
   `per_feature_gap` (normalized |a−b| per feature), is already computed by
   `explainer_disagreement` but is **not scored** in the report.

4. **Set- and sign-based metrics saturate.** Sign stability is 1.0 until LIME
   is under-sampled to 20; top-k flip rate stays 0.0 because the top-3 *set*
   never changes (only its order); top-k overlap saturates at 1.0. These are
   coarse detectors that only fire on dramatic changes.

## Recommendation

Add a **magnitude-disagreement** metric to the scorecard by scoring the
already-computed `per_feature_gap` (e.g., its mean over the top-k features).
This is the missing piece the study exposes: it is what actually separates the
collinear "SHAP and LIME tell different stories" regime from the clean one, and
it is the natural next experiment axis for the disagreement family.

## Applying the results

The current `explaintrust/report.py` thresholds remain the documented defaults;
this study is the first step toward real calibration, not the final word. To
reproduce or tweak:

```bash
python3 experiments/calibrate_thresholds.py   # regenerates calibration.json
```

## Caveats

1. **Synthetic ground truth.** The regimes are engineered with the synthetic
   datasets' known generative process. Real data has no such labels, so this is
   a calibration of the *metric scale*, not of *real-world usefulness*.
2. **One model family.** Only `RandomForestClassifier` is swept here (plus a
   linear model for infidelity). Other model families may shift distributions.
3. **Classification only.** The report's thresholds are in log-odds space for
   binary classifiers; regression needs a separate sweep.
4. **Percentiles are a convention.** The 80/20 rule is transparent and
   reproducible, but it is a choice. With real labeled data you should replace
   it with a decision-theoretic objective (e.g., a maximum acceptable
   false-trust rate).
5. **Necessary, not sufficient.** Calibrated thresholds do not make an
   explanation trustworthy; they make the *verdict* about it less arbitrary.
