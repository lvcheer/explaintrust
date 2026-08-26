# Threshold calibration experiment

This directory turns the report's "sensible defaults" into **calibrated
defaults** on synthetic data with known ground truth. It is a first pass — a
defensible v0.2 baseline, not a claim about real-world distributions.

## Why calibrate?

`explaintrust/report.py` maps every metric to a verdict (`good` / `warn` /
`bad`) using two thresholds per metric. Those thresholds were originally chosen
by hand. Choosing them *by hand* is the weakest link in an otherwise defensible
pipeline: the metrics are implemented from published definitions, but the
boundaries that turn a number into a verdict were vibes.

## Method

For each metric we construct two regimes in which we **know** the ground truth,
then measure the metric's distribution in each:

| Metric | "good" regime (trustworthy) | "bad" regime (untrustworthy) |
|---|---|---|
| removal-effect correlation | real SHAP attribution | SHAP magnitudes shuffled across features |
| comprehensiveness ratio | real SHAP attribution | shuffled SHAP |
| LIME infidelity | real LIME weights | shuffled LIME weights |
| max-sensitivity | deterministic TreeSHAP | undersampled (noisy) LIME |
| run-to-run stability | LIME, many samples (2000) | LIME, few samples (200) |
| SHAP vs LIME disagreement | collinearity removed (ρ = 0) | high collinearity (ρ = 0.85) |
| cross-segment stability | in-distribution holdout | drifted inputs (`x1_drift`) |

We sweep 20 seeds with a `RandomForestClassifier(n_estimators=60, max_depth=4)`
on the collinear dataset (`n = 900`), and aggregate each metric over ~80
instance-level samples per regime.

**Threshold rule.** Two boundaries per metric so that roughly 80% of known-good
samples are judged `good` and 80% of known-bad samples are judged `bad`:

* *higher-is-better*: `good = P20(good)`, `warn = P80(bad)`
* *lower-is-better*: `good = P80(good)`, `warn = P20(bad)`

The two validation columns (`good_pass_rate`, `bad_flag_rate`) report how well
the chosen thresholds recover the ground-truth labels on the same data.

## Results

| Metric | dir | med(good) | med(bad) | good | warn | pass(good) | flag(bad) | Result |
|---|---|---|---|---|---|---|---|---|
| SHAP removal-effect correlation | ↑ | 0.616 | −0.030 | 0.426 | 0.249 | .80 | .78 | ✅ separates |
| SHAP comprehensiveness (top-k vs random) | ↑ | 2.343 | 0.476 | 1.441 | 2.033 | .80 | .74 | ⚠️ direction right, tails overlap |
| LIME local fidelity (infidelity) | ↓ | 0.002 | 1.302 | 0.004 | 0.986 | .80 | .80 | ✅ separates |
| Max sensitivity | ↓ | 0.029 | 0.135 | 0.035 | 0.074 | .80 | .80 | ✅ separates |
| Run-to-run rank stability | ↑ | 0.910 | 0.625 | 0.865 | 0.708 | .80 | .80 | ✅ separates |
| Run-to-run sign stability | ↑ | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | .45 | ❌ no signal |
| SHAP vs LIME sign disagreement | ↓ | 0.200 | 0.300 | 0.420 | 0.200 | .80 | .11 | ⚠️ weak |
| SHAP vs LIME rank agreement | ↑ | 0.715 | 0.751 | 0.469 | 0.857 | .80 | .12 | ❌ inverted |
| SHAP vs LIME top-3 overlap | ↑ | 0.667 | 1.000 | 0.667 | 1.000 | .93 | .05 | ❌ inverted |
| Cross-segment rank stability | ↑ | 0.988 | 0.988 | 0.952 | 0.988 | .85 | .10 | ❌ no signal |
| Top-3 flip rate across segments | ↓ | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 | .00 | ❌ no signal |

(*↑ = higher-is-better, ↓ = lower-is-better. "pass(good)"/"flag(bad)" are the
validation rates under the fitted thresholds.*)

## Findings

1. **Four metrics separate cleanly** — removal-effect correlation, LIME
   infidelity, max-sensitivity, and run-to-run rank stability all show a strong,
   correctly-directed signal on their synthetic axes. This *confirms* the
   metric definitions behave as intended.

2. **Absolute thresholds are regime-specific, so they are *not* auto-baked.**
   For example, infidelity was calibrated on a *linear* model (where LIME's
   weights are the true gradient), so its "good" bar lands near 0.004 — but a
   real random-forest LIME reads ≈ 0.1. Baking that number in would flag every
   tree-model explanation as untrustworthy. The calibrated values are a useful
   *scale reference*, not universal defaults.

3. **Five metrics showed no (or inverted) signal on these axes.** Sign
   stability is trivially 1.0 in both regimes (LIME's top-feature signs do not
   flip even at 50 samples); the disagreement and cross-segment metrics are
   dominated by the seven noise features, and — as the article itself notes —
   SHAP already resists `x2`, so decoupling `x2` does not flip the top-k. These
   metrics need a *better experimental axis* (not a different threshold).

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
2. **One model family.** Only `RandomForestClassifier` is swept here. Tree vs
   linear vs gradient-boosting models may shift the distributions (the linear
   path is exercised by the test suite but not by this sweep).
3. **Classification only.** The report's thresholds are in log-odds space for
   binary classifiers; regression needs a separate sweep.
4. **Percentiles are a convention.** The 80/20 rule is transparent and
   reproducible, but it is a choice. With real labeled data you should replace
   it with a decision-theoretic objective (e.g., target a maximum acceptable
   false-trust rate).
5. **Necessary, not sufficient.** Calibrated thresholds do not make an
   explanation trustworthy; they make the *verdict* about it less arbitrary.
