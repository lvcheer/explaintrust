# Real-data benchmark: where the metrics actually land

A companion to the synthetic calibration. It runs the full trust-metric battery
on **two real UCI datasets** — Adult Income (48k × 84 features) and Diabetes
130-US (102k × 62 features) — across three model families (Random Forest,
Gradient Boosting, Logistic Regression) × 8 seeds, with numeric features
z-scored. It answers two questions the synthetic study cannot:

1. **Do the report's hand-picked defaults match reality?**
2. **Are the metrics' typical values transferable across datasets?**

Reproduce with `python3 experiments/benchmark_real_data.py`
(data auto-downloads into `experiments/data/`, which is gitignored).

## Results (pooled over 2 datasets × 3 models × 8 seeds)

| Metric | dir | Adult | Diabetes | pooled | P10 | P90 | default (good/warn) | verdict@median |
|---|---|---|---|---|---|---|---|---|
| removal-effect correlation | ↑ | 0.425 | 0.477 | 0.454 | 0.371 | 0.506 | 0.5 / 0.2 | warn |
| comprehensiveness ratio | ↑ | 62.6 | 10.6 | 28.5 | 7.2 | 428.7 | `> 1` gate | good |
| LIME infidelity (normalized) | ↓ | 0.774 | 0.868 | 0.827 | 0.080 | 1.398 | 0.5 / 1.0 | warn |
| max-sensitivity | ↓ | 0.008 | 0.022 | 0.011 | 0.000 | 0.816 | 0.5 / 2.0 | good |
| run-to-run rank stability (all-d) | ↑ | 0.416 | 0.475 | 0.460 | 0.274 | 1.000 | 0.9 / 0.7 | **bad** |
| run-to-run rank stability (**top-k**) | ↑ | 1.000 | 1.000 | 1.000 | 0.940 | 1.000 | 0.9 / 0.7 | **good** |
| run-to-run sign stability | ↑ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9 / 0.7 | good |
| SHAP vs LIME sign disagreement | ↓ | 0.328 | 0.322 | 0.326 | 0.000 | 0.379 | 0.2 / 0.5 | warn |
| SHAP vs LIME rank agreement (all-d) | ↑ | 0.419 | 0.518 | 0.456 | 0.328 | 0.974 | 0.7 / 0.4 | warn |
| SHAP vs LIME rank agreement (**top-k**) | ↑ | 0.562 | 0.875 | 0.750 | 0.250 | 1.000 | 0.7 / 0.4 | **good** |
| SHAP vs LIME top-k overlap | ↑ | 0.667 | 0.750 | 0.750 | 0.583 | 0.833 | 0.66 / 0.33 | good |
| SHAP vs LIME magnitude disagreement | ↓ | 0.779 | 0.605 | 0.727 | 0.405 | 1.024 | 1.0 / 1.5 | good |
| cross-segment rank stability | ↑ | 0.984 | 0.991 | 0.989 | 0.936 | 0.999 | 0.7 / 0.4 | good |
| top-k flip rate | ↓ | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.34 / 0.67 | **bad** |

(*"verdict@median" = what the current report default would say about the pooled
median. ↑ higher-is-better, ↓ lower-is-better.*)

## Findings

1. **Several defaults are too strict for real data — the biggest is run-to-run
   rank stability.** The default demands ≥ 0.9, but real LIME rank stability is
   ≈ 0.46 on 84-feature Adult and 0.48 on 62-feature Diabetes. The synthetic
   10-feature calibration (0.91) was optimistic: **rank-based metrics degrade
   with feature dimensionality**, because dozens of one-hot noise features
   shuffle their ranks between runs. Same story for SHAP-vs-LIME rank agreement
   (real ≈ 0.46 vs default 0.7). As shipped, the report would flag essentially
   *every* real-data explanation as unreliable on stability grounds.

2. **Other defaults are too loose / saturating.** Comprehensiveness explodes on
   high-dimensional data (median 28.5, P90 428) because "remove 3 random
   features" does almost nothing when 80 features are noise — so the default
   `> 1.5` is trivially met and the ratio is meaningless as a *graded* number.
   Max-sensitivity defaults (0.5 / 2.0) are far looser than real values (~0.01).

3. **Transferability is partial.** Removal-effect correlation (0.43 vs 0.48),
   sign stability, sign disagreement, cross-segment rank stability, and
   top-k overlap transfer well across the two datasets. But comprehensiveness
   (62 vs 10), LIME infidelity (0.23 vs 0.03), magnitude disagreement (0.78 vs
   0.61), and top-k flip rate (1.0 vs 0.0) are strongly dataset-specific — a
   single threshold cannot serve both.

4. **Feature standardization is mandatory.** On raw Adult, capital-gain
   (0–99999) blew LIME infidelity up to ~1.7×10⁷ and collapsed the distribution
   segmentation into one degenerate bin. After z-scoring, infidelity lands at
   ≈ 0.23 and the segmentation behaves. The metrics implicitly assume
   unit-scale features (the synthetic datasets were N(0,1)).

## Resolution

All three findings are now addressed in the library:

* **Dimensionality (rank stability/agreement).** `cross_run_stability` and
  `explainer_disagreement` return a **`topk_rank_corr`** (Spearman over the
  top-k features only), and `report.py` scores that instead of the all-features
  rank correlation. On real data the top-k versions read **1.00** (stability)
  and **0.75** (agreement) — "good" — while the all-features versions read 0.46.

* **Comprehensiveness** is now a `> 1` "not noise" **gate** in `report.py`,
  not a graded score (its absolute size saturates on high-dimensional data).

* **LIME infidelity** is now **normalized** by the variance of the model's
  output change, making it a scale-free fraction (≈1 = "no better than
  predicting zero change"). This fixed the transferability: Adult 0.77 vs
  Diabetes 0.87, instead of the raw 0.23 vs 0.03 (a 9× gap).

Remaining known limitation:

* **Top-k flip rate is coarse and dataset-specific** (1.0 on Adult vs 0.0 on
  Diabetes — with three segments it can only be 0 / 0.5 / 1.0). It is kept as a
  coarse "does the story flip across segments" signal, not a finely-gradable
  score; the thresholds treat 0 as good and 1 as bad.
