# Public result locations

Audit date: 2026-09-20.

This inventory covers every tracked source that stores, renders, or makes a
reader-facing claim about experiment results. It distinguishes the two planned
benchmark refreshes from the separate synthetic example used by the article.
That distinction matters: article-example numbers must not be generated from a
benchmark summary, and benchmark results must not be inferred from the article
example.

## Canonical stored results

| Surface | Result family | Current content | Required action |
|---|---|---|---|
| `experiments/results/synthetic/raw_runs.json` | Refreshed synthetic calibration | Per-seed, model, scenario, condition, and observation records; created by the refreshed runner | Treat as the canonical raw source after the first reviewed run. |
| `experiments/results/synthetic/summary.json` | Refreshed synthetic calibration | Frozen config plus derived thresholds and held-out summaries; created by the refreshed runner | Generate only from the same in-memory observations written to `raw_runs.json`. |
| `experiments/results/synthetic/environment.json` | Refreshed synthetic calibration | Commit, dirty flag, Python/platform, runner/config digests, command, and installed packages | Keep beside the exact raw and summary outputs from that execution. |
| `experiments/calibration.json` | Synthetic calibration | Historical thresholds, held-out medians, pass rates, and flag rates | Preserve as the pre-refresh baseline. Replace only with reviewed output from the frozen synthetic protocol, or migrate it explicitly into the new raw/summary layout. |
| `experiments/benchmark_results.json` | Adult/Diabetes benchmark | Historical dataset/model/seed summaries from the earlier split and explanation protocol | Preserve as the pre-refresh baseline. Replace only after refreshed run records, split context, non-finite statuses, and summaries pass validation. |
| `article/figures/conversion.json` | Article-only synthetic example | Per-feature attributions plus before/after rank-correlation and sign-disagreement values | Regenerate with `article/scripts/generate_figures.py`. Keep separate from both benchmark result families. |

These JSON files are currently the nearest available machine-readable sources,
but the two experiment files are explicitly historical rather than valid output
for the frozen refresh protocol.

## Reader-facing tables and numerical claims

| Surface | Values or claims shown | Source that should control it | Required action |
|---|---|---|---|
| `experiments/README.md` — Results and Findings | Complete synthetic table: medians, fitted boundaries, held-out pass/flag rates, and qualitative findings | Refreshed synthetic summary | Generate the table; review every prose claim, including negative results, against the same summary. |
| `experiments/benchmark_README.md` — Results, Findings, and Resolution | Adult/Diabetes/pooled medians, P10/P90, defaults, verdicts, and prose comparisons | Refreshed real-data summary | Generate the table; retain protocol history while removing obsolete empirical prose. |
| `article/index.qmd` — sections 1–4 | Removal correlation, comprehensiveness, method-agreement, sign-disagreement, stability, and collinearity example numbers | Article-only generated data plus explicitly saved outputs for article examples | Reconcile every number after regenerating the article examples. Do not source these claims from the Adult/Diabetes benchmark. |
| `README.md` — English and Chinese quickstart sections | No benchmark table; both language sections state that historical experiment tables await regeneration | Refresh completion status | Remove both warnings only after the synthetic and real-data result bundles are reviewed. |
| `explaintrust/report.py` — module documentation | Qualitative claim that three metrics separate engineered regimes while several do not | Refreshed synthetic summary | Recheck the claim after the synthetic rerun. Numeric defaults remain a decision policy, not fitted benchmark output. |
| `CHANGELOG.md` — Unreleased | States that saved historical results require regeneration | Refresh completion and release history | Record completion, protocol changes, and non-comparability without copying a new result table into the changelog. |
| `article/README.md` | States that article figures and numbers use the earlier protocol | Article-example regeneration status | Remove the pending-refresh notice only after figure generation and prose reconciliation succeed. |

## Figures and rendered outputs

| Surface | Generator/source | Required action |
|---|---|---|
| `article/figures/conversion_flip.png` | `article/scripts/generate_figures.py`, using the same run that writes `conversion.json` | Regenerate and visually verify its displayed correlation values against `conversion.json`. |
| `article/figures/endpoints.png` | `article/scripts/generate_figures.py`, separate clean/collinear synthetic profiles | Regenerate and visually verify labels, attribution scale, and empirical-correlation wording. |
| `article/_site/` | Quarto render of `article/index.qmd`; ignored by Git | Re-render for visual QA. It is a delivery artifact, not a canonical result source. |
| Deployed article/GitHub Pages, if published | Published `article/_site/` | Publish only from the reviewed render tied to the result commit. |

`article/scripts/generate_figures.py` is therefore part of the result-producing
pipeline even though it is not itself a result surface. Its console messages are
diagnostic only.

## Demo and decision-policy values

| Surface | What users see | Classification | Required action |
|---|---|---|---|
| `app/streamlit_app.py` | Live-computed model performance, metric values, coverage counts, subgroup tables, and exported report JSON | Per-session computation, not a checked-in benchmark snapshot | Confirm wording remains diagnostic and that the app does not embed refreshed benchmark constants. |
| `explaintrust/report.py::DEFAULT_THRESHOLDS` | Six good/warn boundaries included in reports and visible through the app/export | Versioned decision-policy defaults, not validated universal thresholds | Keep distinct from fitted synthetic boundaries. Any change requires a separate policy decision and tests. |
| `experiments/benchmark_real_data.py::CURRENT_DEFAULTS` | Defaults printed beside benchmark medians | Manual mirror of the report decision policy | Consistency checking must detect drift from `DEFAULT_THRESHOLDS`; do not silently fit these values to the refreshed test results. |
| `examples/demo.py` | Live example metrics and report printed when run | Computed example, not a stored benchmark result | Use as a smoke test; do not copy its one-run numbers into public findings. |

The Streamlit app contains no checked-in benchmark conclusion. Its displayed
numbers vary with the selected data, model, seed, and analysis budget.

## Result-producing entry points

| Entry point | Output currently written or displayed |
|---|---|
| `python -m experiments.calibrate_thresholds` | Validates the frozen config; preserves historical `calibration.json`; writes `experiments/results/synthetic/{raw_runs,summary,environment}.json`; and prints the synthetic result table. |
| `python -m experiments.benchmark_real_data --n-explain 4` | Writes `experiments/benchmark_results.json` and prints the real-data summary table. |
| `python article/scripts/generate_figures.py` | Writes `conversion.json`, `conversion_flip.png`, and `endpoints.png`. |
| `quarto render article` | Builds the ignored `article/_site/` delivery artifact. |

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

## Consistency-check coverage required later

Automation must eventually verify:

1. both experiment Markdown tables against their corresponding summaries;
2. qualitative result claims in `README.md`, `article/README.md`,
   `CHANGELOG.md`, and `explaintrust/report.py` through explicit status markers
   or narrowly scoped assertions;
3. article interactive/static values against `conversion.json` and regenerated
   figures;
4. the real-data runner's displayed defaults against
   `explaintrust.report.DEFAULT_THRESHOLDS`;
5. that ignored render/build outputs are regenerated from the reviewed sources
   rather than treated as inputs.
