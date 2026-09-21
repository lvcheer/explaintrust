# Public result locations

Audit date: 2026-09-21.

The source-to-target contract is frozen in
`experiments/public_result_map.json` (schema 1). The
`python -m experiments.build_all_results` entry point consumes that map in
write and check modes. It currently controls the two benchmark Markdown tables
and all article-derived claims, data, and figures, and it checks six manually
reviewed result contracts. Workflow integration remains planned. This inventory
explains the boundary in human-readable form.

This inventory covers every tracked source that stores, renders, or makes a
reader-facing claim about experiment results. It distinguishes the two reviewed
benchmark families from the separate synthetic example used by the article.
That distinction matters: article-example numbers must not be generated from a
benchmark summary, and benchmark results must not be inferred from the article
example.

## Canonical stored results

| Surface | Result family | Current content | Required action |
|---|---|---|---|
| `experiments/results/synthetic/raw_runs.json` | Refreshed synthetic calibration | Reviewed per-seed, model, scenario, condition, and observation records | Canonical raw source for the refreshed synthetic study. |
| `experiments/results/synthetic/summary.json` | Refreshed synthetic calibration | Reviewed frozen config, derived thresholds, and held-out summaries | Canonical source for the synthetic Markdown table. |
| `experiments/results/synthetic/environment.json` | Refreshed synthetic calibration | Commit, dirty flag, Python/platform, runner/config digests, command, and installed packages | Keep beside the exact raw and summary outputs from that execution. |
| `experiments/results/real/raw_runs.json` | Refreshed Adult/Diabetes benchmark | Reviewed 48-run bundle with sample, subgroup, split, background, and status evidence | Canonical raw source; public summaries must remain traceable to these runs. |
| `experiments/results/real/summary.json` | Refreshed Adult/Diabetes benchmark | Reviewed run-level medians and percentiles under the frozen protocol | Canonical source for the real-data Markdown table. |
| `experiments/results/real/environment.json` | Refreshed Adult/Diabetes benchmark | Exact execution context, source hashes, and config/runner identities | Keep unchanged beside the reviewed raw and summary files. |
| `experiments/results/real/change_report.md` | Historical comparison | Prespecified comparison plus reviewed technical explanations for all material changes | Canonical interpretation source; do not infer causal explanations from the summary alone. |
| `experiments/calibration.json` | Synthetic calibration | Historical thresholds, held-out medians, pass rates, and flag rates | Preserve as the immutable pre-refresh baseline; never use it to generate refreshed claims. |
| `experiments/benchmark_results.json` | Adult/Diabetes benchmark | Historical dataset/model/seed summaries from the earlier split and explanation protocol | Preserve as the immutable pre-refresh comparison baseline; never overwrite it with refreshed output. |
| `article/figures/article_results.json` | Article-only synthetic example | Canonical seeded configuration, environment, per-instance measurements, article summaries, OJS source data, and endpoint profiles | Regenerate with `article/scripts/generate_figures.py`. Keep separate from both benchmark result families. |
| `article/figures/conversion.json` | Article-only synthetic example | Compatibility projection for the OJS interactive | Generate from `article_results.json`; never edit independently. |

The two top-level experiment JSON files remain immutable historical baselines;
the reviewed files below `experiments/results/` are the current experimental
sources. Article examples remain a third, separate result family.

## Reader-facing tables and numerical claims

| Surface | Values or claims shown | Source that should control it | Required action |
|---|---|---|---|
| `experiments/README.md` — Results and Findings | Generated refreshed table plus manually reviewed synthetic findings | Refreshed synthetic summary | Keep the table generated; check surrounding qualitative and negative-result prose against the same summary. |
| `experiments/benchmark_README.md` — Results, Findings, and Resolution | Generated refreshed table plus manually reviewed real-data findings | Refreshed real-data summary and change report | Keep the table generated; retain protocol history and use the reviewed change report for interpretation. |
| `article/index.qmd` — sections 1–4 | Removal correlation, comprehensiveness, method-agreement, sign-disagreement, stability, and collinearity example numbers | `article/figures/article_results.json` | Generate marked claims from the article bundle. Do not source these claims from the Adult/Diabetes benchmark. |
| `README.md` — English and Chinese quickstart sections | Matching bilingual status: reviewed outputs drive generated public tables while top-level JSON remains historical | Reviewed synthetic and real-data bundle status | Keep this prose manual; check mode rejects stale or mismatched status wording. |
| `explaintrust/report.py` — module documentation | Qualitative claim that three metrics separate engineered regimes while several do not | Refreshed synthetic summary | Keep the checked claim qualitative. Numeric defaults remain a decision policy, not fitted benchmark output. |
| `CHANGELOG.md` — Unreleased | Records completed refreshes, protocol changes, preserved baselines, and non-comparability | Refresh completion and release history | Keep the checked status without copying result tables into the changelog. |
| `article/README.md` | Documents the completed numerical refresh and reproduction commands | Article-example regeneration status | Keep the status synchronized with the generated bundle and consistency check. |

## Figures and rendered outputs

| Surface | Generator/source | Required action |
|---|---|---|
| `article/figures/conversion_flip.png` | `article_results.json` through `build_all_results` | Regenerate and visually verify its displayed correlation values against `conversion.json`. |
| `article/figures/endpoints.png` | `article_results.json` through `build_all_results` | Regenerate and visually verify labels, mean-absolute attribution scale, and empirical-correlation wording. |
| `article/_site/` | Quarto render of `article/index.qmd`; ignored by Git | Re-render for visual QA. It is a delivery artifact, not a canonical result source. |
| Deployed article/GitHub Pages, if published | Published `article/_site/` | Publish only from the reviewed render tied to the result commit. |

`article/scripts/generate_figures.py` is therefore part of the result-producing
pipeline even though it is not itself a result surface. Its console messages are
diagnostic only.

## Demo and decision-policy values

| Surface | What users see | Classification | Required action |
|---|---|---|---|
| `app/streamlit_app.py` | Live-computed model performance, metric values, coverage counts, subgroup tables, and exported report JSON | Per-session computation, not a checked-in benchmark snapshot | A checked contract keeps wording diagnostic and forbids references to stored benchmark result files. |
| `explaintrust/report.py::DEFAULT_THRESHOLDS` | Six good/warn boundaries included in reports and visible through the app/export | Versioned decision-policy defaults, not validated universal thresholds | Keep distinct from fitted synthetic boundaries. Any change requires a separate policy decision and tests. |
| `experiments/benchmark_real_data.py::CURRENT_DEFAULTS` | Defaults printed beside benchmark medians | Checked mirror of the report decision policy | Check mode detects drift from `DEFAULT_THRESHOLDS`; do not silently fit these values to the refreshed test results. |
| `examples/demo.py` | Live example metrics and report printed when run | Computed example, not a stored benchmark result | Use as a smoke test; do not copy its one-run numbers into public findings. |

The Streamlit app contains no checked-in benchmark conclusion. Its displayed
numbers vary with the selected data, model, seed, and analysis budget.

## Result-producing entry points

| Entry point | Output currently written or displayed |
|---|---|
| `python -m experiments.build_all_results [--check]` | Writes both benchmark tables and all article-derived claims/data/figures, or checks them for source drift without modifying files. |
| `python -m experiments.calibrate_thresholds` | Validates the frozen config; preserves historical `calibration.json`; writes `experiments/results/synthetic/{raw_runs,summary,environment}.json`; and prints the synthetic result table. |
| `python -m experiments.benchmark_real_data --n-explain 4` | Preserves historical `experiments/benchmark_results.json`; writes `experiments/results/real/{raw_runs,summary,environment}.json` plus `change_report.md`; and prints the real-data summary table. |
| `python article/scripts/generate_figures.py` | Runs the seeded article example, writes `article_results.json`, and refreshes every article-derived target through `build_all_results`. |
| `quarto render article` | Builds the ignored `article/_site/` delivery artifact. |

## Frozen generation mapping

| Mapping ID | Canonical source | Target | Boundary |
|---|---|---|---|
| `synthetic_results_table` | `results/synthetic/summary.json` | `experiments/README.md` | Generated Markdown block; findings remain reviewed prose. |
| `real_results_table` | `results/real/summary.json` plus report policy | `experiments/benchmark_README.md` | Generated Markdown block; material-change explanations come from `change_report.md`. |
| `article_numeric_claims` | `article_results.json` | `article/index.qmd` | Generated marked article-example claims, separate from both benchmarks. |
| `article_conversion_data` | `article_results.json` | `article/figures/conversion.json` | Generated compatibility projection for the OJS interactive. |
| `article_conversion_figure` | `article_results.json` | `conversion_flip.png` | Generated static figure carrying its source-bundle digest. |
| `article_endpoints_figure` | `article_results.json` | `endpoints.png` | Generated static figure carrying its source-bundle digest. |

The Streamlit app and `examples/demo.py` intentionally have no generated
benchmark snapshot: they compute per-session/example values. Check mode must
instead reject embedded benchmark medians or fitted experimental thresholds.
CI will run `python -m experiments.build_all_results --check`; the release
workflow will run write mode followed by check mode. Contributor guidance will
name check mode for changes to result-producing code.

## Explicit exclusions

The following must not become canonical sources for refreshed claims:

- `review/2026-09-05/PROJECT_REVIEW.zh-CN.md` and `probe-results.json`: frozen
  historical review evidence tied to an earlier commit;
- tests and test fixtures: expected values validate code behaviour but are not
  empirical study results;
- `experiments/data/`: cached source data, not derived results;
- `article/.quarto/`: local Quarto cache;
- `dist/` and `*.egg-info/`: rebuildable packaging artifacts;
- console logs, screenshots, notebooks, or manually copied tables.

`CITATION.cff` is release metadata rather than a result source. At release time,
its version/date must point to the commit containing the reviewed results.

## Current consistency-check coverage

`python -m experiments.build_all_results --check` now verifies:

1. both experiment Markdown tables against their corresponding summaries;
2. qualitative result claims in `README.md`, both experiment READMEs,
   `article/README.md`, `CHANGELOG.md`, and `explaintrust/report.py` through
   narrowly scoped assertions;
3. article claims, compatibility JSON, and static figures against the canonical
   `article_results.json` bundle (implemented);
4. the real-data runner's displayed defaults against
   `explaintrust.report.DEFAULT_THRESHOLDS`;
5. that the Streamlit and headless demos remain live computations without
   references to stored benchmark outputs.

CI, release, and contributor-command integration remain the next workflow step.
Ignored Quarto/build outputs remain delivery artifacts rather than inputs.
