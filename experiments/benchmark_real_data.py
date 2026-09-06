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

Output: ``benchmark_results.json`` plus a printed summary table.

Run from the repo root:

    python3 experiments/benchmark_real_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
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

MODELS = {
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=60, max_depth=5, random_state=s, n_jobs=-1),
    "GradientBoosting": lambda s: GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=s),
    "LogisticRegression": lambda s: LogisticRegression(max_iter=1000, random_state=s),
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
    comps = [comprehensiveness_ratio(pred, X_explain[i], S[i], X_bg, top_k=TOPK, n_random=20, seed=seed + i) for i in range(n)]
    infids = [infidelity(pred, X_explain[i], L[i], X_bg, n_perturbations=200, seed=seed + i) for i in range(n)]

    def explain_single(x):
        return shap_attributions(model, x.reshape(1, -1), X_background=X_bg, method="auto")[0]

    sensitivities = [
        max_sensitivity(explain_single, X_explain[i], X_bg, n_perturbations=6, seed=seed + i)
        for i in range(n)
    ]

    # Repeat the same full batch at the displayed LIME budget; reuse its first
    # contribution matrix so stability describes the explanations above.
    stability_seeds = list(range(seed, seed + 5))
    repeated_contributions = [C]
    for repeat_seed in stability_seeds[1:]:
        attr = lime_attributions(model, X_explain, X_bg, feature_names=names,
                                 num_samples=LIME_SAMPLES, seed=repeat_seed)
        repeated_contributions.append(to_contribution_scale(attr, X_explain, X_bg))
    stabilities = [
        cross_run_stability(lambda seed: repeated_contributions[seed][i], n_runs=5, top_k=TOPK)
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
                              "sensitivity_perturbations_per_sample": 6,
                              "stability_samples": n, "lime_batch_calls": 5,
                              "lime_instance_explanations": 5 * n,
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


def main(n_explain=None) -> None:
    n_explain = N_EXPLAIN if n_explain is None else n_explain
    _sample_test_positions(1, n_explain, 0)  # validate before downloads or model fitting
    datasets = {"adult": load_adult(), "diabetes": load_diabetes()}
    rows = []
    for dname, data in datasets.items():
        for seed in SEEDS:
            Xtr, Xte, ytr, _, names, split_context = prepare_split(data, seed)
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

    payload = {
        "schema_version": 3,
        "aggregation": {"run_metric": "mean_excluding_nan_preserving_infinity",
                        "summary": "finite_run_percentiles",
                        "summary_unit": "dataset_model_seed_run",
                        "note": "Percentiles describe runs, not confidence intervals or independent samples."},
        "budget": {"requested_explanations": int(n_explain), "background_max": BG,
                   "subgroup_max": DIST_N, "lime_samples": LIME_SAMPLES, "top_k": TOPK,
                   "infidelity_perturbations": 200, "comprehensiveness_random": 20,
                   "sensitivity_perturbations": 6, "stability_seed_offsets": list(range(5)),
                   "sensitivity_and_stability_scope": "all_explained_samples"},
        "runs": [
            {**{key: value for key, value in row.items() if key not in METRICS},
             "metrics": {key: _measurement(row[key]) for key in METRICS}}
            for row in rows
        ],
        "datasets": {name: {"raw_rows": len(data.frame),
                            "raw_features": len(data.numeric) + len(data.categorical)}
                     for name, data in datasets.items()},
        "n_runs": len(rows),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "metrics": summary,
        "run_contexts": [
            {"dataset": r["dataset"], "model": r["model"], "seed": r["seed"], "shap": r["shap_context"],
             "split": r["split_context"]}
            for r in rows
        ],
    }
    (HERE / "benchmark_results.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False))

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
    print(f"wrote {HERE / 'benchmark_results.json'}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-explain", type=int, default=N_EXPLAIN,
                        help="test rows to explain per dataset/model/seed (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.n_explain < 1:
        parser.error("--n-explain must be a positive integer")
    return args


if __name__ == "__main__":
    sys.exit(main(n_explain=_parse_args().n_explain))
