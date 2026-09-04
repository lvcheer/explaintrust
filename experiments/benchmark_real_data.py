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

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

from experiments.real_datasets import load_adult, load_diabetes  # noqa: E402
SEEDS = range(8)
N_EXPLAIN = 4
BG = 200
TOPK = 3
LIME_SAMPLES = 1000
DIST_N = 2000  # subsample size for the distribution-verification check

MODELS = {
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=60, max_depth=5, random_state=s, n_jobs=-1),
    "GradientBoosting": lambda s: GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=s),
    "LogisticRegression": lambda s: LogisticRegression(max_iter=1000, random_state=s),
}

# Current hand-picked defaults in explaintrust/report.py (kept here for the
# comparison table; update if report.py changes).
CURRENT_DEFAULTS = {
    "removal_corr": ("higher", 0.5, 0.2),
    "comprehensiveness": ("higher", 1.5, 1.0),
    "infidelity": ("lower", 0.1, 0.5),
    "sensitivity": ("lower", 0.5, 2.0),
    "stability_rank": ("higher", 0.9, 0.7),
    "stability_rank_topk": ("higher", 0.9, 0.7),
    "stability_sign": ("higher", 0.9, 0.7),
    "disagreement_sign": ("lower", 0.2, 0.5),
    "disagreement_rank": ("higher", 0.7, 0.4),
    "disagreement_rank_topk": ("higher", 0.7, 0.4),
    "disagreement_topk": ("higher", 0.66, 0.33),
    "disagreement_magnitude": ("lower", 1.0, 1.5),
    "distribution_rank": ("higher", 0.7, 0.4),
    "distribution_flip": ("lower", 0.34, 0.67),
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


def _run_metrics(model, X_explain, X_bg, names, X_dist, seed):
    pred = scalar_predictor(model)
    S = shap_attributions(model, X_explain, X_background=X_bg, method="auto")
    L = lime_attributions(model, X_explain, X_bg, feature_names=names, num_samples=LIME_SAMPLES, seed=seed)
    C = to_contribution_scale(L, X_explain, X_bg)

    n = len(X_explain)
    removals = [removal_effect_correlation(pred, X_explain[i], S[i], X_bg) for i in range(n)]
    comps = [comprehensiveness_ratio(pred, X_explain[i], S[i], X_bg, top_k=TOPK, seed=seed + i) for i in range(n)]
    infids = [infidelity(pred, X_explain[i], L[i], X_bg, seed=seed + i) for i in range(n)]

    def explain_single(x):
        return shap_attributions(model, x.reshape(1, -1), X_background=X_bg, method="auto")[0]

    sens = max_sensitivity(explain_single, X_explain[0], X_bg, n_perturbations=6, seed=seed)

    def lime_seeded(seed: int):
        return lime_attributions(model, X_explain[:1], X_bg, feature_names=names, num_samples=LIME_SAMPLES, seed=seed)[0]

    stab = cross_run_stability(lime_seeded, n_runs=5, top_k=TOPK)

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
    seg = np.digitize(Xd[:, seg_feat], np.quantile(Xd[:, seg_feat], [0.33, 0.66]))
    dist = cross_segment_stability(Xd, seg, Ad, top_k=TOPK)

    def nanmean(xs):
        xs = [x for x in xs if x == x and np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "removal_corr": nanmean(removals),
        "comprehensiveness": nanmean(comps),
        "infidelity": nanmean(infids),
        "sensitivity": sens,
        "stability_rank": stab["rank_corr"],
        "stability_rank_topk": stab["topk_rank_corr"],
        "stability_sign": stab["sign_agreement"],
        "disagreement_sign": nanmean(ds["sign"]),
        "disagreement_rank": nanmean(ds["rank"]),
        "disagreement_rank_topk": nanmean(ds["ranktopk"]),
        "disagreement_topk": nanmean(ds["topk"]),
        "disagreement_magnitude": nanmean(ds["mag"]),
        "distribution_rank": dist["rank_corr"],
        "distribution_flip": dist["topk_flip_rate"],
    }


def _pct(xs, p):
    xs = [x for x in xs if x == x and np.isfinite(x)]
    return float(np.percentile(xs, p)) if xs else float("nan")


def main() -> None:
    datasets = {"adult": load_adult(), "diabetes": load_diabetes()}
    rows = []
    for dname, (X, y, names) in datasets.items():
        for mname, mk in MODELS.items():
            for seed in SEEDS:
                Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=seed)
                model = mk(seed).fit(Xtr, ytr)
                rng = np.random.default_rng(seed)
                X_bg = Xtr[rng.choice(len(Xtr), min(BG, len(Xtr)), replace=False)]
                m = _run_metrics(model, Xte[:N_EXPLAIN], X_bg, names, Xte, seed)
                m.update({"dataset": dname, "model": mname, "seed": seed})
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
            "pooled_median": round(_pct(pooled, 50), 4),
            "pooled_p10": round(_pct(pooled, 10), 4),
            "pooled_p90": round(_pct(pooled, 90), 4),
            "per_dataset_median": {d: round(_pct(v, 50), 4) for d, v in per_dataset.items()},
            "current_default": [c_good, c_warn],
        }

    payload = {
        "datasets": {"adult": datasets["adult"][0].shape, "diabetes": datasets["diabetes"][0].shape},
        "n_runs": len(rows),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "metrics": summary,
    }
    (HERE / "benchmark_results.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 118)
    print(f"{'metric':26s} {'dir':6s} {'adult':>8s} {'diabetes':>9s} {'pooled':>8s} {'P10':>8s} {'P90':>8s} | {'default':>11s} {'verdict@median':>15s}")
    print("-" * 118)
    for metric in METRICS:
        s = summary[metric]
        direction, c_good, c_warn = CURRENT_DEFAULTS[metric]
        med = s["pooled_median"]
        # what would the current default say about the pooled median?
        if direction == "higher":
            verdict = "good" if med >= c_good else ("warn" if med >= c_warn else "bad")
        else:
            verdict = "good" if med <= c_good else ("warn" if med <= c_warn else "bad")
        print(
            f"{metric:26s} {direction:6s} "
            f"{s['per_dataset_median']['adult']:>8.3f} {s['per_dataset_median']['diabetes']:>9.3f} "
            f"{med:>8.3f} {s['pooled_p10']:>8.3f} {s['pooled_p90']:>8.3f} | "
            f"{str([c_good, c_warn]):>11s} {verdict:>15s}"
        )
    print("=" * 118)
    print(f"wrote {HERE / 'benchmark_results.json'}")


if __name__ == "__main__":
    sys.exit(main())
