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
| comprehensiveness ratio | ↑ | 62.6 | 10.6 | 28.5 | 7.2 | 428.7 | 1.5 / 1.0 | good |
| LIME infidelity | ↓ | 0.234 | 0.027 | 0.097 | 0.012 | 1.037 | 0.1 / 0.5 | good |
| max-sensitivity | ↓ | 0.008 | 0.022 | 0.011 | 0.000 | 0.816 | 0.5 / 2.0 | good |
| run-to-run rank stability | ↑ | 0.416 | 0.475 | 0.460 | 0.274 | 1.000 | 0.9 / 0.7 | **bad** |
| run-to-run sign stability | ↑ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9 / 0.7 | good |
| SHAP vs LIME sign disagreement | ↓ | 0.328 | 0.322 | 0.326 | 0.000 | 0.379 | 0.2 / 0.5 | warn |
| SHAP vs LIME rank agreement | ↑ | 0.419 | 0.518 | 0.456 | 0.328 | 0.974 | 0.7 / 0.4 | warn |
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

## Proposed direction (not yet baked into `report.py`)

* **Dimensionality-aware stability/agreement.** The rank-based defaults (0.9 /
  0.7) are untenable at d ≈ 60–80. Either (a) lower them to the real-data
  medians (rank stability good ≈ 0.5, rank agreement good ≈ 0.5), or (b) make
  the verdict depend on the feature count, or (c) prefer top-k overlap and sign
  stability — which are robust to d — as the primary stability/agreement checks.

* **Keep the "not noise" checks qualitative.** Comprehensiveness should stay a
  `> 1` gate, not a graded score, on high-dimensional data.

* **Split thresholds by dataset/scale where transferability fails** (infidelity,
  flip rate), or report a per-dataset reference distribution alongside the
  verdict.

The current `report.py` defaults are unchanged pending these decisions; this
document is the evidence base for the next revision.
