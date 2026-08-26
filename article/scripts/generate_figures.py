"""Generate figures and data for the explorable article.

Run from the repo root:

    python3 article/scripts/generate_figures.py

Outputs (into article/figures/):
  * conversion.json       — the "category error" flip: SHAP vs raw-LIME vs
                            contribution-scaled-LIME (centerpiece interactive)
  * conversion_flip.png   — static fallback of the before/after rank correlation
  * endpoints.png         — clean vs collinear per-feature SHAP/LIME bars

The centerpiece tells a *true* and *clean* story: most "SHAP and LIME wildly
disagree" results are partly a category error — comparing LIME's slopes against
SHAP's contributions. Put them on the same scale and the apparent disagreement
largely vanishes. The residual disagreement (the collinearity part) is shown
per-feature in the second figure.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from explaintrust import (  # noqa: E402
    make_collinear_dataset,
    shap_attributions,
    lime_attributions,
    to_contribution_scale,
)
from explaintrust.metrics import explainer_disagreement  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _make_dataset_with_rho(n: int, rho: float, seed: int):
    """Like make_collinear_dataset, but with tunable x0/x2 collinearity rho."""
    rng = np.random.default_rng(seed)
    d = 10
    X = rng.normal(0.0, 1.0, size=(n, d))
    X[:, 2] = rho * X[:, 0] + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.normal(0.0, 0.35, size=n)
    logit = (
        1.5 * X[:, 0]
        - 1.0 * X[:, 1]
        + 0.6 * X[:, 0] * X[:, 1]
        + 0.3 * X[:, 0] ** 2
        - 0.25
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    names = [f"x{i}" for i in range(d)]
    return X, y, names


def _fit_and_explain(X, y, names, n_explain, seed):
    rng = np.random.default_rng(seed)
    n_train = int(0.7 * len(X))
    model = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=seed)
    model.fit(X[:n_train], y[:n_train])
    X_bg = X[:n_train][rng.choice(n_train, size=80, replace=False)]
    X_explain = X[n_train:][:n_explain]
    shap = shap_attributions(model, X_explain, method="tree")
    raw_lime = lime_attributions(model, X_explain, X_bg, feature_names=names, num_samples=1500, seed=seed)
    contrib_lime = to_contribution_scale(raw_lime, X_explain, X_bg)
    return shap, raw_lime, contrib_lime


def _agreement(a, b, n_explain):
    rank_corrs, sign_diss = [], []
    for i in range(n_explain):
        d = explainer_disagreement(a[i], b[i], top_k=3)
        rank_corrs.append(d["rank_corr"])
        sign_diss.append(d["sign_disagreement"])
    return float(np.nanmean(rank_corrs)), float(np.mean(sign_diss))


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Centerpiece: the category-error flip (collinear dataset, rho=.85)
    # ------------------------------------------------------------------ #
    X, y, names = _make_dataset_with_rho(n=1500, rho=0.85, seed=0)
    n_explain = 20
    shap, raw_lime, contrib_lime = _fit_and_explain(X, y, names, n_explain, seed=0)

    before_corr, before_sign = _agreement(shap, raw_lime, n_explain)
    after_corr, after_sign = _agreement(shap, contrib_lime, n_explain)

    inst = 0
    payload = {
        "features": names,
        "shap": [round(float(v), 3) for v in shap[inst]],
        "raw_lime": [round(float(v), 3) for v in raw_lime[inst]],
        "contrib_lime": [round(float(v), 3) for v in contrib_lime[inst]],
        "before_rank_corr": round(before_corr, 3),
        "after_rank_corr": round(after_corr, 3),
        "before_sign_disagreement": round(before_sign, 3),
        "after_sign_disagreement": round(after_sign, 3),
        "n_instances": n_explain,
    }
    with open(os.path.join(FIG_DIR, "conversion.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote conversion.json (before={before_corr:.3f}, after={after_corr:.3f})")

    # Static fallback: before/after rank correlation.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["SHAP vs raw LIME\n(wrong scale)", "SHAP vs contribution LIME\n(same scale)"],
           [before_corr, after_corr], color=["#EF553B", "#636EFA"])
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_ylabel("rank correlation (higher = agree)")
    ax.set_ylim(-0.5, 1.0)
    ax.set_title("The apparent disagreement is mostly a category error")
    for i, v in enumerate([before_corr, after_corr]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "conversion_flip.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Collinearity story: clean vs collinear, per-feature credit split
    # ------------------------------------------------------------------ #
    profiles = {}
    for key, rho in [("clean", 0.0), ("collinear", 0.85)]:
        Xr, yr, nr = _make_dataset_with_rho(n=1200, rho=rho, seed=42)
        sr, _, cr = _fit_and_explain(Xr, yr, nr, n_explain=10, seed=42)
        profiles[key] = {
            "features": nr[:4],
            "shap": [round(float(np.mean(sr[:, j])), 3) for j in range(4)],
            "lime": [round(float(np.mean(cr[:, j])), 3) for j in range(4)],
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (key, title) in zip(axes, [("clean", "No collinearity (ρ = 0)"),
                                       ("collinear", "Collinear x0/x2 (ρ = 0.85)")]):
        prof = profiles[key]
        x = np.arange(len(prof["features"]))
        w = 0.38
        ax.bar(x - w / 2, prof["shap"], w, label="SHAP", color="#636EFA")
        ax.bar(x + w / 2, prof["lime"], w, label="LIME", color="#EF553B")
        ax.set_xticks(x)
        ax.set_xticklabels(prof["features"])
        ax.set_title(title)
        ax.axhline(0, color="#999", linewidth=0.8)
    axes[0].legend()
    axes[0].set_ylabel("mean |attribution| (contribution scale)")
    fig.suptitle("Collinearity quietly teaches LIME to credit x2 (which is NOT causal)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "endpoints.png"), dpi=150)
    plt.close(fig)
    print("wrote endpoints.png")

    print("done")


if __name__ == "__main__":
    main()
