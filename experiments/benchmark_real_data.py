"""Anchor report thresholds on real tabular data.

Runs the full trust-metric battery on two real UCI datasets (Adult, Diabetes
130-US) across three model families and several seeds, and reports each metric's
empirical distribution. This answers two questions the synthetic calibration
cannot:

1. **Where do the metrics actually land on realistic data?** — i.e. are the
   report's hand-picked defaults achievable, too strict, or too loose?
2. **Are the metrics' typical values transferable across datasets?** — if a
   metric's median is ~the same on Adult and on Diabetes, one threshold can
   serve both; if not, thresholds are dataset-specific.

Output: ``experiments/results/real/`` contains separate raw, summary,
environment, and historical-change artifacts. The historical
``experiments/benchmark_results.json`` is never overwritten.

Run from the repo root:

    python -m experiments.benchmark_real_data --n-explain 4
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from explaintrust import (
    lime_attributions,
    scalar_predictor,
    shap_attributions,
    to_contribution_scale,
)
from explaintrust.metrics import (
    comprehensiveness_ratio,
    cross_run_stability,
    cross_segment_stability,
    explainer_disagreement,
    infidelity,
    max_sensitivity,
    removal_effect_correlation,
)

# Make the repo root importable so ``experiments`` resolves whether run as
# ``python experiments/benchmark_real_data.py`` or imported from the repo root.
HERE = Path(__file__).parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from experiments.real_datasets import load_adult, load_diabetes, prepare_split  # noqa: E402
SEEDS = range(8)
N_EXPLAIN = 4
BG = 200
TOPK = 3
LIME_SAMPLES = 1000
DIST_N = 2000  # subsample size for the subgroup-consistency check
TEST_FRACTION = 0.3
COMPREHENSIVENESS_RANDOM_DRAWS = 20
INFIDELITY_PERTURBATIONS = 200
INFIDELITY_STRATEGY = "gaussian"
INFIDELITY_NORMALIZED = True
SENSITIVITY_PERTURBATIONS = 6
SENSITIVITY_RADIUS = 0.1
STABILITY_SEED_OFFSETS = list(range(5))
RF_N_ESTIMATORS = 60
RF_MAX_DEPTH = 5
GB_N_ESTIMATORS = 60
GB_MAX_DEPTH = 3
LR_MAX_ITER = 1000
DATA_ARCHIVES = {"adult": "adult.zip", "diabetes": "diabetes130.zip"}

MODELS = {
    "RandomForest": lambda s: RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=s, n_jobs=-1,
    ),
    "GradientBoosting": lambda s: GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH, random_state=s,
    ),
    "LogisticRegression": lambda s: LogisticRegression(max_iter=LR_MAX_ITER, random_state=s),
}

# Current hand-picked defaults in explaintrust/report.py (kept here for the
# comparison table; update if report.py changes).
CURRENT_DEFAULTS = {
    "removal_corr": ("higher", 0.5, 0.2),
    "comprehensiveness": ("higher", 1.0, 1.0),  # now a >1 "not noise" gate
    "infidelity": ("lower", 0.5, 1.0),          # normalized against zero-change MSE
    "sensitivity": ("lower", 0.5, 2.0),
    "stability_rank": ("higher", 0.9, 0.7),
    "stability_rank_topk": ("higher", 0.9, 0.7),
    "stability_sign": ("higher", 0.9, 0.7),
    "disagreement_sign": ("descriptive", None, None),
    "disagreement_rank": ("descriptive", None, None),
    "disagreement_rank_topk": ("descriptive", None, None),
    "disagreement_topk": ("descriptive", None, None),
    "disagreement_magnitude": ("descriptive", None, None),
    "distribution_rank": ("descriptive", None, None),
    "distribution_flip": ("descriptive", None, None),
}

METRICS = list(CURRENT_DEFAULTS.keys())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_config(path: Path | None = None):
    path = HERE / "config.json" if path is None else Path(path)
    raw = path.read_bytes()
    config = json.loads(raw)
    real = config["real_data"]
    expected = {
        "seeds": list(SEEDS),
        "explanations_per_dataset_model_seed": N_EXPLAIN,
        "background_max_rows": BG,
        "subgroup_max_rows": DIST_N,
        "top_k": TOPK,
        "metrics": METRICS,
    }
    errors = [
        f"real_data.{key}: config={real.get(key)!r}, runner={value!r}"
        for key, value in expected.items() if real.get(key) != value
    ]
    model_expected = {
        ("RandomForest", "class"): "sklearn.ensemble.RandomForestClassifier",
        ("RandomForest", "n_estimators"): RF_N_ESTIMATORS,
        ("RandomForest", "max_depth"): RF_MAX_DEPTH,
        ("RandomForest", "random_state"): "outer seed",
        ("RandomForest", "n_jobs"): -1,
        ("GradientBoosting", "class"): "sklearn.ensemble.GradientBoostingClassifier",
        ("GradientBoosting", "n_estimators"): GB_N_ESTIMATORS,
        ("GradientBoosting", "max_depth"): GB_MAX_DEPTH,
        ("GradientBoosting", "random_state"): "outer seed",
        ("LogisticRegression", "class"): "sklearn.linear_model.LogisticRegression",
        ("LogisticRegression", "max_iter"): LR_MAX_ITER,
        ("LogisticRegression", "random_state"): "outer seed",
    }
    models = real["models"]
    if set(models) != set(MODELS):
        errors.append(f"real_data.models: config={sorted(models)!r}, runner={sorted(MODELS)!r}")
    errors.extend(
        f"real_data.models.{model}.{key}: config={models.get(model, {}).get(key)!r}, runner={value!r}"
        for (model, key), value in model_expected.items()
        if models.get(model, {}).get(key) != value
    )
    budgets = real["perturbation_budgets"]
    budget_expected = {
        "comprehensiveness_random_draws": COMPREHENSIVENESS_RANDOM_DRAWS,
        "infidelity_perturbations": INFIDELITY_PERTURBATIONS,
        "infidelity_strategy": INFIDELITY_STRATEGY,
        "infidelity_normalized": INFIDELITY_NORMALIZED,
        "sensitivity_perturbations": SENSITIVITY_PERTURBATIONS,
        "sensitivity_radius": SENSITIVITY_RADIUS,
        "stability_seed_offsets": STABILITY_SEED_OFFSETS,
    }
    errors.extend(
        f"real_data.perturbation_budgets.{key}: config={budgets.get(key)!r}, runner={value!r}"
        for key, value in budget_expected.items() if budgets.get(key) != value
    )
    if real["explainers"].get("lime_samples") != LIME_SAMPLES:
        errors.append(
            "real_data.explainers.lime_samples: "
            f"config={real['explainers'].get('lime_samples')!r}, runner={LIME_SAMPLES!r}"
        )
    if set(real["datasets"]) != set(DATA_ARCHIVES):
        errors.append(
            f"real_data.datasets: config={sorted(real['datasets'])!r}, "
            f"runner={sorted(DATA_ARCHIVES)!r}"
        )
    if config.get("schema_version") != 3:
        errors.append(f"schema_version: config={config.get('schema_version')!r}, runner=3")

    contract = real["output_contract"]
    artifacts = contract["artifacts"]
    expected_artifacts = {"raw_runs", "summary", "environment", "change_report"}
    if set(artifacts) != expected_artifacts:
        errors.append(f"real_data.output_contract.artifacts: config={sorted(artifacts)!r}")
    historical_path = contract["historical_baseline"]
    output_paths = [artifact["path"] for artifact in artifacts.values()]
    if historical_path in output_paths:
        errors.append("real-data outputs must not overwrite the historical baseline")
    if len(output_paths) != len(set(output_paths)):
        errors.append("real-data output paths must be unique")
    if any(
        Path(output).is_absolute() or ".." in Path(output).parts
        or Path(output).parts[:3] != ("experiments", "results", "real")
        for output in output_paths
    ):
        errors.append("real-data outputs must remain under experiments/results/real")
    for key in ("preserve_historical_baseline", "publish_only_after_complete_run", "strict_json"):
        if contract.get(key) is not True:
            errors.append(f"real_data.output_contract.{key} must be true")
    for name in ("raw_runs", "summary", "environment"):
        if artifacts.get(name, {}).get("schema_version") != 1:
            errors.append(f"real_data.output_contract.artifacts.{name}.schema_version must be 1")
    if artifacts.get("change_report", {}).get("format") != "markdown":
        errors.append("real_data.output_contract.artifacts.change_report.format must be markdown")
    if errors:
        raise ValueError("frozen config does not match the real-data runner:\n- " + "\n- ".join(errors))

    repo_root = HERE.parent
    comparison = real["historical_comparison"]
    baseline_path = repo_root / comparison["baseline_path"]
    baseline_sha256 = _sha256(baseline_path)
    if baseline_sha256 != comparison["baseline_sha256"]:
        raise ValueError(
            f"historical baseline hash mismatch: {baseline_sha256} != "
            f"{comparison['baseline_sha256']}"
        )
    if comparison["baseline_path"] != historical_path:
        raise ValueError("output contract and comparison must name the same historical baseline")

    return config, hashlib.sha256(raw).hexdigest(), json.loads(baseline_path.read_text())


def _source_context(config: dict) -> dict:
    real = config["real_data"]
    repo_root = HERE.parent
    sources = {}
    for name, archive in DATA_ARCHIVES.items():
        archive_path = HERE / "data" / archive
        actual = _sha256(archive_path)
        expected_hash = real["datasets"][name]["archive_sha256"]
        if actual != expected_hash:
            raise ValueError(f"{name} archive hash mismatch: {actual} != {expected_hash}")
        sources[name] = {
            "path": str(archive_path.relative_to(repo_root)),
            "source_url": real["datasets"][name]["source_url"],
            "sha256": actual,
        }
    return sources


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=HERE.parent, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _environment_snapshot(config_sha256: str, sources: dict) -> dict:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "dirty": bool(_git_output("status", "--porcelain")),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "config": {"path": "experiments/config.json", "sha256": config_sha256},
        "runner": {"path": "experiments/benchmark_real_data.py", "sha256": _sha256(Path(__file__))},
        "datasets": sources,
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def _segment_feature(X, quantiles=(0.33, 0.66), min_rows=5) -> int:
    """Pick the feature whose quantile-binning gives the most balanced segments.

    Avoids skewed features (e.g. capital-gain, 0 for ~90% of rows) whose
    quantiles collapse and leave a single degenerate segment.
    """
    best_j, best_min = 0, -1
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        q = np.quantile(col, quantiles)
        if q[0] == q[-1]:
            continue
        counts = np.bincount(np.digitize(col, q))
        if len(counts) < 2 or counts.min() < min_rows:
            continue
        if counts.min() > best_min:
            best_min, best_j = counts.min(), j
    return best_j


def _run_metrics(model, X_explain, X_bg, names, X_dist, seed, *, test_positions=None):
    if test_positions is None:
        test_positions = np.arange(len(X_explain))
    pred = scalar_predictor(model)
    S, shap_context = shap_attributions(
        model, X_explain, X_background=X_bg, method="auto", return_context=True,
    )
    L = lime_attributions(model, X_explain, X_bg, feature_names=names, num_samples=LIME_SAMPLES, seed=seed)
    C = to_contribution_scale(L, X_explain, X_bg)

    n = len(X_explain)
    removals = [removal_effect_correlation(pred, X_explain[i], S[i], X_bg) for i in range(n)]
    comps = [comprehensiveness_ratio(
        pred, X_explain[i], S[i], X_bg, top_k=TOPK,
        n_random=COMPREHENSIVENESS_RANDOM_DRAWS, seed=seed + i,
    ) for i in range(n)]
    infids = [infidelity(
        pred, X_explain[i], L[i], X_bg,
        n_perturbations=INFIDELITY_PERTURBATIONS,
        strategy=INFIDELITY_STRATEGY, normalize=INFIDELITY_NORMALIZED,
        seed=seed + i,
    ) for i in range(n)]

    def explain_single(x):
        return shap_attributions(model, x.reshape(1, -1), X_background=X_bg, method="auto")[0]

    sensitivities = [
        max_sensitivity(
            explain_single, X_explain[i], X_bg,
            n_perturbations=SENSITIVITY_PERTURBATIONS,
            radius=SENSITIVITY_RADIUS, seed=seed + i,
        )
        for i in range(n)
    ]

    # Repeat the same full batch at the displayed LIME budget; reuse its first
    # contribution matrix so stability describes the explanations above.
    stability_seeds = [seed + offset for offset in STABILITY_SEED_OFFSETS]
    repeated_contributions = [C]
    for repeat_seed in stability_seeds[1:]:
        attr = lime_attributions(model, X_explain, X_bg, feature_names=names,
                                 num_samples=LIME_SAMPLES, seed=repeat_seed)
        repeated_contributions.append(to_contribution_scale(attr, X_explain, X_bg))
    stabilities = [
        cross_run_stability(
            lambda seed: repeated_contributions[seed][i],
            n_runs=len(STABILITY_SEED_OFFSETS), top_k=TOPK,
        )
        for i in range(n)
    ]

    ds = {"sign": [], "rank": [], "ranktopk": [], "topk": [], "mag": []}
    for i in range(n):
        d = explainer_disagreement(S[i], C[i], top_k=TOPK)
        ds["sign"].append(d["sign_disagreement"])
        ds["rank"].append(d["rank_corr"])
        ds["ranktopk"].append(d["topk_rank_corr"])
        ds["topk"].append(d["topk_overlap"])
        ds["mag"].append(d["magnitude_disagreement"])

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_dist), min(DIST_N, len(X_dist)), replace=False)
    Xd = X_dist[idx]
    Ad = shap_attributions(model, Xd, X_background=X_bg, method="auto")
    seg_feat = _segment_feature(Xd)
    cutpoints = np.quantile(Xd[:, seg_feat], [0.33, 0.66])
    seg = np.digitize(Xd[:, seg_feat], cutpoints)
    dist = cross_segment_stability(Xd, seg, Ad, top_k=TOPK)

    sample_values = {
        "removal_corr": removals, "comprehensiveness": comps, "infidelity": infids,
        "disagreement_sign": ds["sign"], "disagreement_rank": ds["rank"],
        "disagreement_rank_topk": ds["ranktopk"], "disagreement_topk": ds["topk"],
        "disagreement_magnitude": ds["mag"],
        "sensitivity": sensitivities,
        "stability_rank": [s["rank_corr"] for s in stabilities],
        "stability_rank_topk": [s["topk_rank_corr"] for s in stabilities],
        "stability_sign": [s["sign_agreement"] for s in stabilities],
    }
    subgroup = {"distribution_rank": dist["rank_corr"],
                "distribution_flip": dist["topk_flip_rate"]}
    values = {**sample_values, **{k: [v] for k, v in subgroup.items()}}
    result = {key: _mean(value) for key, value in values.items()}
    result.update({
        "shap_context": shap_context,
        "metric_counts": {key: _counts(value) for key, value in values.items()},
        "samples": [
            {"sample_position": i, "test_position": int(test_positions[i]), "perturbation_seed": seed + i,
             "metrics": {key: _measurement(value[i]) for key, value in sample_values.items()},
             "shap": S[i].tolist(), "lime_coefficients": L[i].tolist(),
             "lime_contributions": C[i].tolist()}
            for i in range(n)
        ],
        "stability_runs": [
            {"sample_position": i, "test_position": int(test_positions[i]), "runs": [
                {"seed": repeat_seed, "contributions": contributions[i].tolist()}
                for repeat_seed, contributions in zip(stability_seeds, repeated_contributions)
            ]} for i in range(n)
        ],
        "evaluation_counts": {"explained_samples": n, "sensitivity_samples": n,
                              "sensitivity_perturbations_per_sample": SENSITIVITY_PERTURBATIONS,
                              "stability_samples": n,
                              "lime_batch_calls": len(STABILITY_SEED_OFFSETS),
                              "lime_instance_explanations": len(STABILITY_SEED_OFFSETS) * n,
                              "stability_seeds": stability_seeds},
        "subgroups": {
            "test_positions": idx.tolist(), "feature": names[seg_feat],
            "cutpoints": cutpoints.tolist(), "segment_ids": seg.tolist(),
            "group_ids": [int(v) for v in dist["segment_ids"]],
            "group_counts": [int(np.sum(seg == v)) for v in dist["segment_ids"]],
            "mean_absolute_attributions": dist["importances"].tolist(),
            "metrics": {key: _measurement(value) for key, value in subgroup.items()},
        },
    })
    return result


def _counts(xs):
    values = np.asarray(xs, dtype=float)
    return {"total": len(values), "finite": int(np.isfinite(values).sum()),
            "nan": int(np.isnan(values).sum()),
            "positive_infinity": int(np.isposinf(values).sum()),
            "negative_infinity": int(np.isneginf(values).sum())}


def _measurement(value):
    status = ("finite" if np.isfinite(value) else "nan" if np.isnan(value)
              else "positive_infinity" if value > 0 else "negative_infinity")
    return {"value": float(value) if np.isfinite(value) else None, "status": status}


def _mean(xs):
    # Keep infinite failures visible; only NaN denotes unavailable observations.
    values = np.asarray(xs, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def _json_safe(value):
    """Strict JSON; metric envelopes/counts retain why a numeric value is null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _pct(xs, p):
    xs = [x for x in xs if x == x and np.isfinite(x)]
    return float(np.percentile(xs, p)) if xs else float("nan")


def _sample_test_positions(n_test, n_explain, seed):
    if isinstance(n_explain, (bool, np.bool_)) or not isinstance(n_explain, (int, np.integer)) or n_explain < 1:
        raise ValueError("n_explain must be a positive integer")
    if n_test < 1:
        raise ValueError("test partition must not be empty")
    # Dedicated RNG: background size and model iteration cannot alter this draw.
    return np.random.default_rng(seed).choice(n_test, min(n_explain, n_test), replace=False)


def _build_change_report(summary: dict, baseline: dict, comparison: dict) -> str:
    rows = []
    precision = comparison["comparison_precision_decimals"]
    for metric in METRICS:
        old_metric = baseline.get("metrics", {}).get(metric, {})
        new_metric = summary.get(metric, {})
        floor = comparison["absolute_floor_overrides"].get(
            metric, comparison["absolute_floor_default"],
        )
        for scope in comparison["comparison_scope"]:
            if scope == "pooled":
                old = old_metric.get("pooled_median")
                new = new_metric.get("pooled_median")
            else:
                old = old_metric.get("per_dataset_median", {}).get(scope)
                new = new_metric.get("per_dataset_median", {}).get(scope)
            if old is None or new is None or not np.isfinite(old) or not np.isfinite(new):
                delta = threshold = None
                status = "FLAG: missing or non-finite"
            else:
                old = round(float(old), precision)
                new = round(float(new), precision)
                delta = round(new - old, precision)
                threshold = max(float(floor), comparison["relative_tolerance"] * abs(old))
                status = "MATERIAL CHANGE" if abs(delta) > threshold else "within tolerance"
            fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
            rows.append(
                f"| {metric} | {scope} | {fmt(old)} | {fmt(new)} | "
                f"{fmt(delta)} | {fmt(threshold)} | {status} |"
            )
    return "\n".join([
        "# Real-data benchmark change report",
        "",
        "## Baseline identity",
        "",
        f"- Path: `{comparison['baseline_path']}`",
        f"- SHA-256: `{comparison['baseline_sha256']}`",
        "- The baseline has summary medians but no raw run records.",
        "",
        "## Comparison rule",
        "",
        f"- Scope: {', '.join(comparison['comparison_scope'])} medians.",
        f"- Relative tolerance: {comparison['relative_tolerance']:.0%}.",
        f"- Default absolute floor: {comparison['absolute_floor_default']:.4f}.",
        "- Flag when `absolute_delta > max(metric_absolute_floor, relative_tolerance * abs(historical_value))`.",
        "- Equality is within tolerance; missing or non-finite values are always flagged.",
        "",
        "## Metric comparison",
        "",
        "| Metric | Scope | Historical | Refreshed | Signed delta | Tolerance | Status |",
        "|---|---|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## Non-comparable fields",
        "",
        "The historical file has no raw runs, split hashes, sample records, background records, or environment metadata. Paired run-level changes and provenance fields are therefore not comparable.",
        "",
        "## Interpretation",
        "",
        "A material-change flag requires a technical explanation; it is not automatically a regression failure because the historical and refreshed protocols intentionally differ.",
        "",
    ])


def _write_artifacts(config: dict, payloads: dict, change_report: str) -> dict:
    contract = config["real_data"]["output_contract"]
    artifacts = contract["artifacts"]
    rendered = {}
    for name, payload in payloads.items():
        required = set(artifacts[name]["required_top_level_fields"])
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{name} artifact is missing fields: {sorted(missing)}")
        rendered[name] = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    report_lower = change_report.lower()
    missing_sections = [
        section for section in artifacts["change_report"]["required_sections"]
        if f"## {section}" not in report_lower
    ]
    if missing_sections:
        raise ValueError(f"change report is missing sections: {missing_sections}")
    rendered["change_report"] = change_report

    repo_root = HERE.parent
    targets = {
        name: (repo_root / artifacts[name]["path"]).resolve()
        for name in rendered
    }
    historical = (repo_root / contract["historical_baseline"]).resolve()
    if historical in targets.values():
        raise ValueError("refusing to overwrite historical benchmark results")
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        targets[name].write_text(text)
    return targets


def main(n_explain=None) -> None:
    n_explain = N_EXPLAIN if n_explain is None else n_explain
    _sample_test_positions(1, n_explain, 0)  # validate before downloads or model fitting
    frozen_config, config_sha256, baseline = _load_frozen_config()
    frozen_n_explain = frozen_config["real_data"]["explanations_per_dataset_model_seed"]
    if n_explain != frozen_n_explain:
        raise ValueError(
            f"n_explain={n_explain} does not match frozen real-data config {frozen_n_explain}"
        )
    datasets = {"adult": load_adult(), "diabetes": load_diabetes()}
    sources = _source_context(frozen_config)
    rows = []
    for dname, data in datasets.items():
        for seed in SEEDS:
            Xtr, Xte, ytr, _, names, split_context = prepare_split(
                data, seed, test_size=TEST_FRACTION,
            )
            test_positions = _sample_test_positions(len(Xte), n_explain, seed)
            for mname, mk in MODELS.items():
                model = mk(seed).fit(Xtr, ytr)
                rng = np.random.default_rng(seed)
                background_positions = rng.choice(len(Xtr), min(BG, len(Xtr)), replace=False)
                X_bg = Xtr[background_positions]
                m = _run_metrics(model, Xte[test_positions], X_bg, names, Xte, seed,
                                 test_positions=test_positions)
                m["sampling"] = {"method": "uniform_without_replacement", "seed": seed,
                                 "unit": "test_row", "requested": int(n_explain),
                                 "available": len(Xte), "actual": len(test_positions),
                                 "test_positions": test_positions.tolist()}
                m.update({"dataset": dname, "model": mname, "seed": seed, "split_context": split_context})
                m["feature_names"] = names
                m["background_train_positions"] = background_positions.tolist()
                rows.append(m)
                print(f"[{dname}/{mname}/seed{seed}] done", flush=True)

    summary = {}
    for metric in METRICS:
        direction, c_good, c_warn = CURRENT_DEFAULTS[metric]
        pooled = [r[metric] for r in rows]
        per_dataset = {
            dname: [r[metric] for r in rows if r["dataset"] == dname]
            for dname in datasets
        }
        summary[metric] = {
            "direction": direction,
            "run_counts": _counts(pooled),
            "per_dataset_run_counts": {d: _counts(v) for d, v in per_dataset.items()},
            "pooled_median": round(_pct(pooled, 50), 4),
            "pooled_p10": round(_pct(pooled, 10), 4),
            "pooled_p90": round(_pct(pooled, 90), 4),
            "per_dataset_median": {d: round(_pct(v, 50), 4) for d, v in per_dataset.items()},
            "role": "descriptive" if direction == "descriptive" else "scored",
            "current_default": None if direction == "descriptive" else [c_good, c_warn],
        }

    expected_runs = len(datasets) * len(SEEDS) * len(MODELS)
    if len(rows) != expected_runs:
        raise RuntimeError(f"incomplete benchmark: produced {len(rows)} of {expected_runs} runs")

    aggregation = {
        "run_metric": "mean_excluding_nan_preserving_infinity",
        "summary": "finite_run_percentiles",
        "summary_unit": "dataset_model_seed_run",
        "note": "Percentiles describe runs, not confidence intervals or independent samples.",
    }
    budget = {
        "requested_explanations": int(n_explain), "background_max": BG,
        "subgroup_max": DIST_N, "lime_samples": LIME_SAMPLES, "top_k": TOPK,
        "infidelity_perturbations": INFIDELITY_PERTURBATIONS,
        "infidelity_strategy": INFIDELITY_STRATEGY,
        "infidelity_normalized": INFIDELITY_NORMALIZED,
        "comprehensiveness_random": COMPREHENSIVENESS_RANDOM_DRAWS,
        "sensitivity_perturbations": SENSITIVITY_PERTURBATIONS,
        "sensitivity_radius": SENSITIVITY_RADIUS,
        "stability_seed_offsets": STABILITY_SEED_OFFSETS,
        "sensitivity_and_stability_scope": "all_explained_samples",
    }
    exported_runs = [
        {**{key: value for key, value in row.items() if key not in METRICS},
         "metrics": {key: _measurement(row[key]) for key in METRICS}}
        for row in rows
    ]
    dataset_summary = {
        name: {"raw_rows": len(data.frame),
               "raw_features": len(data.numeric) + len(data.categorical)}
        for name, data in datasets.items()
    }
    environment = _environment_snapshot(config_sha256, sources)
    run_context = {
        "git_commit": environment["git"]["commit"],
        "git_dirty": environment["git"]["dirty"],
        "config_sha256": config_sha256,
        "environment_file": "environment.json",
        "raw_runs_file": "raw_runs.json",
    }
    raw_payload = {
        "schema_version": 1,
        "run_context": run_context,
        "n_runs": len(exported_runs),
        "runs": exported_runs,
    }
    summary_payload = {
        "schema_version": 1,
        "run_context": run_context,
        "frozen_config": frozen_config,
        "aggregation": aggregation,
        "budget": budget,
        "datasets": dataset_summary,
        "n_runs": len(exported_runs),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "metrics": summary,
    }
    change_report = _build_change_report(
        summary, baseline, frozen_config["real_data"]["historical_comparison"],
    )
    written = _write_artifacts(
        frozen_config,
        {"raw_runs": raw_payload, "summary": summary_payload, "environment": environment},
        change_report,
    )

    print("\n" + "=" * 118)
    print(f"{'metric':26s} {'dir':6s} {'adult':>8s} {'diabetes':>9s} {'pooled':>8s} {'P10':>8s} {'P90':>8s} | {'default':>11s} {'verdict@median':>15s}")
    print("-" * 118)
    for metric in METRICS:
        s = summary[metric]
        direction, c_good, c_warn = CURRENT_DEFAULTS[metric]
        med = s["pooled_median"]
        # what would the current default say about the pooled median?
        if direction == "descriptive":
            verdict = "descriptive"
        elif not np.isfinite(med):
            verdict = "unavailable"
        elif direction == "higher":
            verdict = "good" if med >= c_good else ("warn" if med >= c_warn else "bad")
        else:
            verdict = "good" if med <= c_good else ("warn" if med <= c_warn else "bad")
        if direction != "descriptive" and s["run_counts"]["positive_infinity"]:
            verdict = "infinite: inspect"
        print(
            f"{metric:26s} {direction:6s} "
            f"{s['per_dataset_median']['adult']:>8.3f} {s['per_dataset_median']['diabetes']:>9.3f} "
            f"{med:>8.3f} {s['pooled_p10']:>8.3f} {s['pooled_p90']:>8.3f} | "
            f"{str([c_good, c_warn]) if direction != 'descriptive' else 'not scored':>11s} {verdict:>15s}"
        )
    print("=" * 118)
    for name in ("raw_runs", "summary", "environment", "change_report"):
        print(f"wrote {written[name]}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-explain", type=int, default=N_EXPLAIN,
                        help="test rows per run; must match frozen config (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.n_explain < 1:
        parser.error("--n-explain must be a positive integer")
    return args


if __name__ == "__main__":
    sys.exit(main(n_explain=_parse_args().n_explain))
