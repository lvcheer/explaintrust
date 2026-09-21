# explaintrust

## Testing post-hoc explanations before acting on them

Version 0.2.0 | Technical brief | 21 September 2026

explaintrust is a Python toolkit for evaluating numeric tabular SHAP and LIME
explanations. It asks whether an explanation tracks the model, remains stable
under perturbation, reproduces across stochastic runs, and changes across
observed groups. The output is a structured trust report, not a certificate of
correctness or causality.

## The problem

Attribution plots are easy to read and easy to over-interpret. A ranked bar
chart can look coherent even when removing the highest-ranked features barely
changes the prediction, when a new random seed changes the ranking, or when the
same model behaves differently across subgroups. SHAP and LIME also return
different mathematical objects: SHAP values are baseline-relative
contributions, while LIME coefficients are local slopes. Comparing them without
aligning model output and units creates measurement artifacts.

explaintrust keeps measurement and interpretation separate. Faithfulness,
sensitivity, and reproducibility checks contribute to the overall verdict.
Method disagreement and subgroup heterogeneity are reported as descriptive
evidence because a difference does not identify which explanation is better.

## Evaluation map

| Evidence question | Metrics | Role |
|---|---|---|
| Do important features move the prediction? | SHAP removal-effect correlation and comprehensiveness; normalized LIME infidelity | Scored |
| Does the explanation survive perturbation and reruns? | Max-sensitivity; top-k rank and sign stability | Scored |
| Do two explainers tell different stories? | SHAP-LIME sign, rank, top-k, and magnitude differences | Descriptive |
| Does the story differ across selected groups? | Cross-segment rank stability and top-k flip rate | Descriptive |

## Evaluation protocol

The controlled synthetic study uses engineered nominal and stress regimes with
held-out seeds. The real-data benchmark covers Adult and Diabetes with 48 run
aggregates across datasets, models, and seeds. Adult retains its official
holdout; Diabetes is split by patient. Preprocessing is fitted on training data,
SHAP and LIME share explicit training backgrounds, and each run samples four
test instances. Raw runs, summaries, configuration digests, environment
metadata, and a historical change report are stored together.

## Three findings

### 1. Only three synthetic metrics separated the engineered regimes

The held-out rule required a good-condition pass rate of at least 0.80 and a
stress-condition flag rate of at least 0.75. Exactly three metrics met both
criteria: SHAP removal-effect correlation, LIME infidelity, and max-sensitivity.
The negative results matter: method agreement and subgroup diagnostics did not
justify new quality thresholds.

### 2. Broader coverage exposed top-k variability

Across 48 Adult/Diabetes run aggregates, the pooled top-k stability median was
`0.875`. Restricting the refreshed raw runs to the first explained instance
restores `1.000`. The change is consistent with evaluating four sampled
instances instead of a single fixed-prefix instance.

### 3. Comprehensiveness is a gate, not an effect size

The pooled median is `20.67`, while P90 is `8.112e+09`. Five sample-level ratios
exceed `1e9` because the random-removal denominator is zero or nearly zero. The
ratio can test whether top-ranked features beat random removal, but its absolute
size should not rank datasets or explanation quality.

<!-- PAGE BREAK -->

## From metrics to a trust report

The report assigns good, warn, bad, unavailable, or not-applicable states to six
scored checks. Its overall conclusion summarizes those checks and preserves
failed evidence when another metric is missing. Descriptive SHAP-LIME and
subgroup diagnostics keep their values but remain outside the overall verdict.
Each JSON export records metric roles, inclusion flags, decision-policy
defaults, sample coverage, and explainer context.

The defaults are versioned policy choices, not universally calibrated cutoffs.
Applications should override them only with domain evidence. A passing report
means that no configured check failed under the recorded setup; it does not
show that the explanation is causal, fair, or suitable for a decision.

## Reproducibility architecture

The experiment configuration freezes datasets, models, seeds, explanation
budgets, thresholds, and output contracts. Runners preserve historical baseline
files and write refreshed raw, summary, and environment bundles under
`experiments/results/`. A single command regenerates public tables, article
claims, compatibility data, and figures from canonical summaries:

```bash
python -m experiments.build_all_results
python -m experiments.build_all_results --check
```

CI runs the check on Python 3.9 and 3.12, builds wheel and source archives, and
clean-installs the wheel before testing import metadata and a minimal report.
The same consistency contract checks the headline values in the bilingual
README, project page, and this brief.

## Limits and interpretation

- Current scope is numeric tabular binary classification and regression with
  SHAP and LIME. Multiclass, image, text, LLM, and counterfactual explanations
  are outside scope.
- The real-data evidence comes from Adult and Diabetes. It is not universal
  validation, and one Adult subgroup run is explicitly not computed.
- Training-only preprocessing, corrected holdouts, sampled explanation rows,
  explicit backgrounds, and broader stability coverage changed together. The
  historical file has no raw runs, so individual causes cannot be isolated.
- Twelve of 42 prespecified historical comparisons were materially different.
  These flags are screening results, not hypothesis tests or proof that the
  refreshed explanations improved.
- The software and article have not been peer reviewed.

## Run the reference pipeline

```bash
git clone https://github.com/lvcheer/explaintrust.git
cd explaintrust
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/demo.py
```

- Live demo: https://lvcheer-explaintrust-appstreamlit-app-x7k48h.streamlit.app/
- Repository: https://github.com/lvcheer/explaintrust
- Project page source: https://github.com/lvcheer/explaintrust/blob/main/article/index.qmd

## Citation

lvcheer (2026). *explaintrust* (version 0.2.0). MIT License.
https://github.com/lvcheer/explaintrust. Machine-readable citation metadata is
provided in `CITATION.cff`.
