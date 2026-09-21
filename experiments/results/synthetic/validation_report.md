# Synthetic benchmark validation report

Date: 2026-09-21  
Branch: `experiment/reproducible-benchmark-refresh`  
Audited result snapshot: `fc2a2837ea423852205418d54a3b4821f6a5bea1`  
Reproducibility report commit: `ae153df50c89a55058a2c1bb0bb4625cc7dd8eb7`

## Decision

The synthetic benchmark **passes implementation-integrity checks but only partially passes scientific calibration checks**. The run is deterministic, finite, train/test separated, and internally aligned on prediction target and explanation background. However, only 4 of the 12 proposed metric boundaries form an ordered three-level decision band; 5 are reversed and 3 are degenerate. Therefore, these synthetic thresholds must remain experimental and must not replace package defaults or be presented as generally validated cut-offs.

Evidence strength is high for statements about this exact run and its implementation, moderate for synthetic scenario discrimination, and low for generalization beyond the engineered datasets.

## Evidence audited

The audit used the committed `raw_runs.json`, `summary.json`, `environment.json`, and `reproducibility_report.json`, plus direct source review and focused tests. The source artifact hashes at audit time were:

| Artifact | SHA-256 |
|---|---|
| `raw_runs.json` | `b5703d4dc46c4499b6fc31bc088878574eed99f84af773dff070e2c35e316927` |
| `summary.json` | `500473d9228667d051d76cf8e30879e31191f98df4a2d211e2cf26285b959636` |
| `environment.json` | `3354cba5d5fb94c2d8250f07e3550a247857a44663d2c605652849a63c03db7d` |
| `reproducibility_report.json` | `fe26933436327e20be23c7c4b9f8c10887962e8124b904da7a2a19e7839c2f72` |

## Acceptance checks

| Check | Result | Evidence and interpretation |
|---|---|---|
| Same seed and environment reproduce exactly | Pass | Two complete runs produced identical canonical raw observations and summary metrics. This is exact reproducibility for the audited environment, not evidence of cross-platform bitwise identity. |
| Different seeds yield reasonable variation | Partial pass | 10/12 nominal conditions and 11/12 stress conditions vary across seeds. Nominal sign stability and both top-3 flip-rate conditions are constant; the former is a ceiling effect and the latter has no discriminatory signal. |
| Training and test data are isolated | Pass, with provenance gap | Direct source review shows model fitting and background selection use `Xtr`, while evaluated instances use `Xte`. Reconstruction for seeds 0–19 gives 630 train rows, 270 test rows, and zero overlap for every seed. Raw artifacts do not yet store split-index hashes, so the artifact alone cannot independently prove this property. |
| Non-finite outcomes remain explicit | Pass for this run | All 1,320 observations are finite. The raw schema preserves `finite`, `nan`, `positive_infinity`, and `negative_infinity` statuses, and focused tests verify invalid/non-finite metric behavior. The summary currently removes non-finite values before calibration; this did not affect the audited run but is a future traceability risk. |
| SHAP/LIME target and background alignment | Pass, with provenance gap | The runner passes the first 100 training rows as the background and uses `scalar_predictor` for metric evaluation. Focused alignment/class-selection tests passed, and spot checks on seeds 0, 7, and 19 reconstructed the selected model output with maximum absolute errors from `3.70e-09` to `5.38e-09`. Raw runs do not store the background digest, selected class, or output space. |
| Focused regression tests | Pass | `92 passed, 2 skipped`; both skips require the optional `xgboost` dependency. No focused test failed. |

The reconstructed positive-class rates were 0.4794–0.5571 in training and 0.4852–0.5630 in test, so the unstratified splits did not create a degenerate class partition in this run.

## Metric-level findings

“Nominal better” compares the seed-level mean of the nominal and stress condition in the metric’s declared direction. It is a paired descriptive check, not a hypothesis test.

| Metric | Paired seed result | Across-seed behavior | Boundary | Audit disposition |
|---|---:|---|---|---|
| SHAP removal-effect correlation | 20/20 nominal better | 20 unique values in both conditions | Ordered | Provisional candidate |
| SHAP comprehensiveness | 19/20 nominal better | 20 unique values in both conditions | Reversed | Signal present; threshold unusable |
| LIME infidelity | 20/20 nominal better | 20 unique values in both conditions | Ordered | Provisional candidate |
| Max sensitivity | 18/20 nominal better | 20 unique values in both conditions | Ordered | Provisional candidate; scenario is algorithm-confounded |
| Run-to-run rank stability | 18/20 nominal better | 19 unique values in both conditions | Ordered | Provisional candidate |
| Run-to-run sign stability | 12/20 nominal better, 8 ties | Nominal fixed at 1.0 | Degenerate (`1.0 = 1.0`) | Ceiling effect; threshold unusable |
| SHAP/LIME sign disagreement | 4/20 nominal better, 5 ties | Variable in both conditions | Reversed | Current control does not validate metric |
| SHAP/LIME rank agreement | 9/20 nominal better, 5 ties | Variable in both conditions | Reversed | Mixed signal; threshold unusable |
| SHAP/LIME top-3 overlap | 5/20 nominal better, 7 ties | Coarse, four unique seed means | Reversed | Current control does not validate metric |
| SHAP/LIME magnitude disagreement | 3/20 nominal better | 20 unique values in both conditions | Reversed | Direction is contradicted by this control |
| Cross-segment rank stability | 11/20 nominal better, 9 ties | Only two values: 0.8 and 1.0 | Degenerate (`1.0 = 1.0`) | Weak, saturated signal |
| Top-3 flip rate across segments | 0/20 nominal better, 20 ties | Fixed at 0.0 in both conditions | Degenerate (`0.0 = 0.0`) | No discriminatory signal |

The four ordered candidates are removal-effect correlation, LIME infidelity, max sensitivity, and run-to-run rank stability. “Provisional” means only that the engineered nominal/stress conditions produce a usable ordered band in this run; it does not establish a universal threshold.

## Major concerns

### 1. Eight proposed boundaries cannot support three-way verdicts

Five boundaries are reversed and three are equal. With the current verdict order, a reversed or equal pair makes the warning interval empty or logically ambiguous. The reported pass/flag rates for those metrics should not be interpreted as evidence that the corresponding threshold is usable.

### 2. Observation-level calibration overstates the effective sample size

Metrics with four explained instances per seed are calibrated as 40 observations per condition, although observations produced by the same fitted model and split are correlated. Seeds 0–9 versus 10–19 do prevent direct seed leakage, but the effective independent sample is closer to 10 seeds per half. Uncertainty intervals and seed-level aggregation are absent.

### 3. Several controls do not isolate a single quality construct

The sensitivity scenario labels high-sample LIME as nominal and TreeSHAP as stress. This compares explainer families as well as sensitivity, so it cannot establish that one algorithm is generally “good” and the other “bad.” The disagreement control shows that collinearity does not consistently increase all four SHAP/LIME disagreement measures. The distribution experiment uses top-3 among only four features, which is too coarse here: the flip rate remains zero in every condition and seed.

### 4. Result artifacts do not carry all causal provenance

The code and tests establish the intended split, background, output and internal repetition seeds, but `raw_runs.json` does not store split hashes, background hashes, selected class/output space, or the five internal seeds used in stability runs. A future reader cannot reconstruct these properties from the result artifact alone.

### 5. Non-finite filtering is silent in the summary layer

Raw serialization correctly preserves non-finite status, but `propose()` drops non-finite values before dividing calibration and test halves. If a future run contains failures, this can change sample membership and hide how many measurements were excluded. The current all-finite run is unaffected.

## Required follow-up before threshold promotion

1. Add a calibration validity gate: if boundaries are reversed or equal, emit `non_separating` instead of good/warn/bad performance.
2. Calibrate and report at the seed/model level, with uncertainty intervals; retain per-instance observations only as nested evidence.
3. Redesign the disagreement and distribution controls, especially the four-feature top-3 flip experiment, and separate algorithm choice from the sensitivity construct.
4. Record split, background, output-space and internal-seed provenance in each run artifact.
5. Report non-finite counts and keep calibration/test seed membership fixed even when an observation is unavailable.
6. Validate any surviving thresholds on the planned real-dataset benchmark before considering changes to package defaults.

## Overall assessment

The refreshed synthetic runner is trustworthy as a reproducible experimental instrument. Its results support four provisional metric candidates and clearly falsify or weaken several other proposed controls. That negative evidence is useful: the correct outcome is to preserve the raw results, avoid promoting the current threshold table, and revise the experimental design in a later explicitly approved step.
