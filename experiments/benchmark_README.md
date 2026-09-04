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
| removal-effect correlation | ↑ | 0.427 | 0.479 | 0.454 | 0.374 | 0.519 | 0.5 / 0.2 | warn |
| comprehensiveness ratio | ↑ | 63.8 | 10.6 | 28.5 | 7.2 | 522.9 | `> 1` gate | good |
| LIME infidelity (normalized) | ↓ | 0.748 | 0.725 | 0.726 | 0.000 | 1.149 | 0.5 / 1.0 | warn |
| max-sensitivity | ↓ | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | 0.5 / 2.0 | good |
| run-to-run rank stability (all-d) | ↑ | 0.926 | 0.910 | 0.921 | 0.878 | 1.000 | 0.9 / 0.7 | good |
| run-to-run rank stability (**top-k**) | ↑ | 1.000 | 1.000 | 1.000 | 0.655 | 1.000 | 0.9 / 0.7 | good |
| run-to-run sign stability | ↑ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9 / 0.7 | good |
| SHAP vs LIME sign disagreement | ↓ | 0.319 | 0.325 | 0.325 | 0.000 | 0.394 | 0.2 / 0.5 | warn |
| SHAP vs LIME rank agreement (all-d) | ↑ | 0.804 | 0.867 | 0.843 | 0.606 | 0.977 | 0.7 / 0.4 | good |
| SHAP vs LIME rank agreement (**top-k**) | ↑ | 0.688 | 0.688 | 0.688 | 0.348 | 1.000 | 0.7 / 0.4 | warn |
| SHAP vs LIME top-k overlap | ↑ | 0.750 | 0.833 | 0.792 | 0.642 | 1.000 | 0.66 / 0.33 | good |
| SHAP vs LIME magnitude disagreement | ↓ | 0.494 | 0.516 | 0.494 | 0.035 | 0.687 | 1.0 / 1.5 | good |
| cross-segment rank stability | ↑ | 0.984 | 0.991 | 0.989 | 0.936 | 0.999 | 0.7 / 0.4 | good |
| top-k flip rate | ↓ | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.34 / 0.67 | **bad** |

(*"verdict@median" = what the current report default would say about the pooled
median. ↑ higher-is-better, ↓ lower-is-better.*)

## Findings

1. **Correcting LIME's feature units changes the rank results materially.** The
   pooled all-feature run stability is 0.92 rather than the previous 0.46, and
   SHAP–LIME rank agreement is 0.84 rather than 0.46. The earlier values mixed
   standardized LIME coefficients with original-coordinate perturbations.

2. **Some defaults remain too loose or saturating.** Comprehensiveness explodes on
   high-dimensional data (median 28.5, P90 523) because "remove 3 random
   features" does almost nothing when 80 features are noise — so the default
   `> 1` gate is trivially met and the ratio is meaningless as a *graded* number.
   Max-sensitivity defaults (0.5 / 2.0) are far looser than observed values
   (pooled median 0.0, P90 0.032 in standardized neighbourhoods).

3. **Transferability is partial.** Removal-effect correlation (0.43 vs 0.48),
   sign stability, sign disagreement, cross-segment rank stability, and
   top-k overlap transfer well across the two datasets. But comprehensiveness
   (64 vs 11) and top-k flip rate (1.0 vs 0.0) are strongly dataset-specific — a
   single threshold cannot serve both.

4. **Scale handling must be explicit.** This benchmark standardizes numeric
   features before fitting. The library now converts LIME slopes back to original
   feature units and defines sensitivity neighbourhoods in background-standardized
   coordinates, so heterogeneous raw scales no longer silently corrupt those
   two calculations.

## Resolution

The implementation changes motivated by these findings are:

* **Rank stability/agreement.** `cross_run_stability` and
  `explainer_disagreement` return a **`topk_rank_corr`** (Spearman over the
  top-k features only). The metric function returns both; `report.py` scores the
  top-k result while preserving the full result for inspection.
  Here the top-k versions read **1.00** (stability) and **0.69** (agreement).

* **Comprehensiveness** is now a `> 1` "not noise" **gate** in `report.py`,
  not a graded score (its absolute size saturates on high-dimensional data).

* **LIME infidelity** is now **normalized** by the mean squared model-output
  change, making it a scale-free fraction (≈1 = "no better than
  predicting zero change"). In this rerun its medians are similar across the two
  datasets (Adult 0.75 vs Diabetes 0.72).

Remaining known limitation:

* **Top-k flip rate is coarse and dataset-specific** (1.0 on Adult vs 0.0 on
  Diabetes — with three segments it can only be 0 / 0.5 / 1.0). It is kept as a
  coarse "does the story flip across segments" signal, not a finely-gradable
  score; the thresholds treat 0 as good and 1 as bad.
