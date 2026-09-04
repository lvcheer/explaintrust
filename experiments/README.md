# Threshold calibration experiment

A controlled study that measures each report metric in engineered nominally
"good" and "bad" regimes, to see (a) which metrics discriminate those regimes,
and (b) whether thresholds can be fitted from data. Shuffled-attribution regimes
are genuine corruption controls; other labels encode stress-test assumptions,
not ground truth that an explanation is trustworthy. The result is a
**diagnostic**, not a claim about real-world distributions.

## Why calibrate?

`explaintrust/report.py` maps every metric to a verdict (`good` / `warn` /
`bad`) using two thresholds per metric. Those thresholds were originally chosen
by hand — the weakest link in an otherwise defensible pipeline. This study asks
whether the data can choose better boundaries, and in doing so it exposes which
metrics *can't* be calibrated on these axes.

## Method

For each metric we construct two contrastive regimes, then measure the metric's
distribution in each:

| Metric | nominal "good" regime | nominal "bad" / stress regime |
|---|---|---|
| removal-effect correlation | real SHAP on clean data | SHAP magnitudes shuffled across features |
| comprehensiveness ratio | real SHAP on clean data | shuffled SHAP |
| LIME infidelity | real LIME on a *linear* model | shuffled LIME weights |
| max-sensitivity | smooth LIME (2000 samples) | piecewise TreeSHAP |
| run-to-run stability | LIME, 2000 samples | LIME, 20 samples |
| SHAP vs LIME disagreement | collinearity removed (ρ = 0), low-dim | high collinearity (ρ ≈ 0.9), low-dim |
| cross-segment stability | random halves (homogeneous) | split by x0 (heterogeneous interaction) |

We sweep 20 seeds, mostly with a `RandomForestClassifier(n_estimators=60,
max_depth=4)` on `n = 900`. Samples from seeds 0–9 form the calibration half;
seeds 10–19 are held out for evaluation.

**Threshold rule.** Fit two percentile boundaries using only the calibration
half:

* *higher-is-better*: `good = P20(good)`, `warn = P80(bad)`
* *lower-is-better*: `good = P80(good)`, `warn = P20(bad)`

The `pass(good)`/`flag(bad)` columns report performance on the held-out seeds,
not on the samples used to choose the thresholds.

## Results

| Metric | dir | med(good) | med(bad) | good | warn | pass | flag | Result |
|---|---|---|---|---|---|---|---|---|
| SHAP removal-effect correlation | ↑ | 0.636 | 0.018 | 0.430 | 0.225 | .80 | .72 | ✅ separates |
| SHAP comprehensiveness (top-k vs random) | ↑ | 2.142 | 0.642 | 1.730 | 2.533 | .62 | .75 | ⚠️ tails overlap |
| LIME local fidelity (infidelity) | ↓ | 0.000 | 1.914 | 0.0001 | 1.433 | .80 | .82 | ✅ separates |
| Max sensitivity | ↓ | 0.0003 | 0.0057 | 0.0006 | 0.0036 | .90 | .80 | ✅ separates |
| Run-to-run rank stability | ↑ | 0.824 | 0.552 | 0.824 | 0.797 | .50 | .90 | ⚠️ unstable good boundary |
| Run-to-run sign stability | ↑ | 1.000 | 0.667 | 1.000 | 1.000 | 1.00 | .60 | ⚠️ coarse |
| SHAP vs LIME sign disagreement | ↓ | 0.250 | 0.250 | 0.250 | 0.000 | .80 | .12 | ❌ no separation |
| SHAP vs LIME rank agreement | ↑ | 0.800 | 0.800 | 0.800 | 1.000 | .80 | .20 | ❌ no separation |
| SHAP vs LIME top-3 overlap | ↑ | 1.000 | 1.000 | 0.667 | 1.000 | 1.00 | .00 | ❌ saturated |
| SHAP vs LIME magnitude disagreement | ↓ | 0.868 | 0.548 | 1.103 | 0.332 | .78 | .15 | ❌ inverted |
| Cross-segment rank stability | ↑ | 1.000 | 0.800 | 1.000 | 1.000 | .80 | .70 | ⚠️ coarse but directional |
| Top-3 flip rate across segments | ↓ | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 | .00 | ❌ saturated |

(*↑ = higher-is-better, ↓ = lower-is-better.*)

## Findings

1. **Three metrics separate on held-out seeds.** Removal-effect correlation,
   LIME infidelity, and max-sensitivity retain a strong, correctly directed
   signal. Run-to-run rank stability identifies noisy explanations, but its
   fitted `good` boundary passes only half of held-out good cases.

2. **Absolute thresholds are regime-specific, so they are not auto-baked.**
   Infidelity was calibrated on a *linear* model (where LIME's weights recover
   the gradient), so its "good" boundary lands near 0.0001. Nonlinear models
   occupy a different regime. Baking that boundary into the package would turn
   a controlled reference case into a false universal rule.

3. **The disagreement metrics are blind to this collinearity regime — the
   study's key negative result.** On held-out seeds, sign and rank medians are
   identical between regimes, top-k overlap saturates, and magnitude
   disagreement is inverted. These diagnostics describe differences between
   explainers; they do not identify proxy reliance as inherently invalid.

4. **Magnitude disagreement is descriptive, not a trust label.** Its held-out
   median is 0.87 in the nominally good regime and 0.55 in the bad regime. The
   metric captures a per-feature relative gap that rank/sign/top-k cannot, but
   this experiment provides no basis for interpreting a larger gap as less
   trustworthy.

5. **Set- and sign-based metrics saturate.** Sign stability is 1.0 until LIME
   is under-sampled to 20; top-k flip rate stays 0.0 because the top-3 *set*
   never changes (only its order); top-k overlap saturates at 1.0. These are
   coarse detectors that only fire on dramatic changes.

## Outcome

The actionable result is a reusable **held-out calibration harness** and clear
negative evidence against auto-adopting most fitted thresholds. The report keeps
documented defaults, including magnitude disagreement (good ≤ 1.0, warn ≤ 1.5),
but these are configurable diagnostics rather than validated trust boundaries.

## Applying the results

The current `explaintrust/report.py` thresholds remain the documented defaults;
this study is the first step toward real calibration, not the final word. To
reproduce or tweak:

```bash
python3 experiments/calibrate_thresholds.py   # regenerates calibration.json
```

## Caveats

1. **Synthetic controls and assumptions.** The regimes use a known generative
   process, but not every good/bad label is ground truth for explanation quality.
   This checks metric scale and selected stress cases, not real-world usefulness.
2. **One model family.** Only `RandomForestClassifier` is swept here (plus a
   linear model for infidelity). Other model families may shift distributions.
3. **Classification only.** The sweep covers binary classifiers. The explained
   output is model dependent (for example, probability for a random forest and
   raw margin for gradient boosting); regression needs a separate sweep.
4. **Percentiles are a convention.** The 80/20 rule is transparent and
   reproducible, but it is a choice. With real labeled data you should replace
   it with a decision-theoretic objective (e.g., a maximum acceptable
   false-trust rate).
5. **Necessary, not sufficient.** Calibrated thresholds do not make an
   explanation trustworthy; they make the *verdict* about it less arbitrary.
