# Threshold calibration experiment

**Refresh completed (2026-09-21).** The reviewed run uses interventional
TreeSHAP with the full explicit background shared with LIME. The generated table
below comes from `experiments/results/synthetic/summary.json`; the top-level
`calibration.json` remains an immutable historical baseline from the earlier
training-path protocol.

A controlled study that measures each report metric in engineered nominally
"good" and "bad" regimes, to see (a) which metrics discriminate those regimes,
and (b) whether thresholds can be fitted from data. Shuffled-attribution regimes
are genuine corruption controls; other labels encode stress-test assumptions,
not ground truth that an explanation is trustworthy. The result is a
**diagnostic**, not a claim about real-world distributions.

## Why calibrate?

The original report mapped every metric to a verdict (`good` / `warn` /
`bad`) using thresholds. Those thresholds were originally chosen
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

<!-- BEGIN AUTO:synthetic-results-table -->
| Metric | dir | med(good) | med(stress) | good | warn | pass | flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SHAP removal-effect correlation | ↑ | 0.6829 | 0.0182 | 0.4279 | 0.24 | 0.85 | 0.78 |
| SHAP comprehensiveness (top-k vs random) | ↑ | 2.2065 | 0.5621 | 1.7216 | 2.5893 | 0.65 | 0.78 |
| LIME local fidelity (infidelity) | ↓ | 0 | 1.9136 | 0.0001 | 1.4334 | 0.80 | 0.82 |
| Max sensitivity | ↓ | 0.0003 | 0.0057 | 0.0006 | 0.0032 | 0.90 | 0.90 |
| Run-to-run rank stability | ↑ | 0.8236 | 0.5515 | 0.824 | 0.7966 | 0.50 | 0.90 |
| Run-to-run sign stability | ↑ | 1 | 0.6667 | 1 | 1 | 1.00 | 0.60 |
| SHAP vs LIME sign disagreement | ↓ | 0.25 | 0.25 | 0.25 | 0 | 0.80 | 0.17 |
| SHAP vs LIME rank agreement | ↑ | 0.8 | 0.8 | 0.8 | 1 | 0.85 | 0.23 |
| SHAP vs LIME top-3 overlap | ↑ | 1 | 1 | 0.6667 | 1 | 1.00 | 0.00 |
| SHAP vs LIME magnitude disagreement (top-3) | ↓ | 0.8238 | 0.547 | 1.0047 | 0.2609 | 0.70 | 0.23 |
| Cross-segment rank stability | ↑ | 1 | 0.8 | 1 | 1 | 0.90 | 0.70 |
| Top-3 flip rate across segments | ↓ | 0 | 0 | 0 | 0 | 1.00 | 0.00 |
<!-- END AUTO:synthetic-results-table -->

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
   median is 0.82 in the nominally good regime and 0.55 in the stress regime. The
   metric captures a per-feature relative gap that rank/sign/top-k cannot, but
   this experiment provides no basis for interpreting a larger gap as less
   trustworthy.

5. **Set- and sign-based metrics saturate.** Sign stability is 1.0 until LIME
   is under-sampled to 20; top-k flip rate stays 0.0 because the top-3 *set*
   never changes (only its order); top-k overlap saturates at 1.0. These are
   coarse detectors that only fire on dramatic changes.

## Outcome

The actionable result is a reusable **held-out calibration harness** and clear
negative evidence against auto-adopting most fitted thresholds. The current
report scores only faithfulness, sensitivity and run-to-run stability. Method
disagreement and subgroup heterogeneity retain their numerical values as
descriptive diagnostics and do not affect the overall verdict. Historical
disagreement/subgroup thresholds in the tables above are no longer applied.

## Applying the results

The current `explaintrust/report.py` thresholds remain the documented defaults;
this study is the first step toward real calibration, not the final word. To
reproduce or tweak:

```bash
python -m experiments.calibrate_thresholds
```

The runner validates `experiments/config.json`, preserves the historical
`calibration.json`, and writes the refreshed per-seed observations, summary,
and environment metadata under `experiments/results/synthetic/`.

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
