"""Generate the canonical result bundle for the explorable article.

Run from the repository root:

    python article/scripts/generate_figures.py

The seeded analysis is written once to ``article/figures/article_results.json``.
The public JSON, numerical claims, and figures are then rendered from that
bundle by ``experiments.build_all_results`` so they cannot silently diverge.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from explaintrust import (
    lime_attributions,
    scalar_predictor,
    shap_attributions,
    to_contribution_scale,
)
from explaintrust.metrics import (
    comprehensiveness_ratio,
    cross_run_stability,
    explainer_disagreement,
    removal_effect_correlation,
)


ROOT = Path(__file__).parents[2]
RESULTS_PATH = ROOT / "article/figures/article_results.json"
PNG_SOURCE_KEY = "Article-Results-SHA256"


def _make_dataset_with_rho(n: int, rho: float, seed: int):
    """Return the article dataset with x0/x2 population correlation ``rho``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 10))
    X[:, 2] = rho * X[:, 0] + np.sqrt(max(0.0, 1 - rho**2)) * rng.normal(size=n)
    logit = (
        1.5 * X[:, 0]
        - 1.0 * X[:, 1]
        + 0.6 * X[:, 0] * X[:, 1]
        + 0.3 * X[:, 0] ** 2
        - 0.25
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < probability).astype(int)
    return X, y, [f"x{i}" for i in range(X.shape[1])]


def _fit_and_explain(
    X,
    y,
    names,
    *,
    n_explain: int,
    seed: int,
    lime_samples: int,
    train_fraction: float,
    background_rows: int,
    n_estimators: int,
    max_depth: int,
):
    rng = np.random.default_rng(seed)
    n_train = int(train_fraction * len(X))
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed
    )
    model.fit(X[:n_train], y[:n_train])
    background = X[:n_train][
        rng.choice(n_train, size=background_rows, replace=False)
    ]
    explained = X[n_train:][:n_explain]
    shap_values, shap_context = shap_attributions(
        model,
        explained,
        X_background=background,
        method="tree",
        return_context=True,
    )
    raw_lime = lime_attributions(
        model,
        explained,
        background,
        feature_names=names,
        num_samples=lime_samples,
        seed=seed,
    )
    contribution_lime = to_contribution_scale(raw_lime, explained, background)
    return {
        "model": model,
        "background": background,
        "explained": explained,
        "shap": shap_values,
        "raw_lime": raw_lime,
        "contribution_lime": contribution_lime,
        "shap_context": shap_context,
        "test_accuracy": float(model.score(X[n_train:], y[n_train:])),
    }


def _agreement(
    shap_values, lime_values, *, top_k: int
) -> tuple[list[float], list[float]]:
    diagnostics = [
        explainer_disagreement(shap_values[i], lime_values[i], top_k=top_k)
        for i in range(len(shap_values))
    ]
    return (
        [float(item["rank_corr"]) for item in diagnostics],
        [float(item["sign_disagreement"]) for item in diagnostics],
    )


def _rounded(values, digits: int = 6) -> list[float]:
    rounded = [round(float(value), digits) for value in values]
    return [0.0 if value == 0 else value for value in rounded]


def generate_article_results() -> dict:
    """Run the seeded article analysis and return its serializable result bundle."""
    config = {
        "dataset_rows": 1500,
        "population_x0_x2_correlation": 0.85,
        "seed": 0,
        "train_fraction": 0.7,
        "background_rows": 80,
        "explained_instances": 20,
        "lime_samples": 1500,
        "top_k": 3,
        "comprehensiveness_random_draws": 20,
        "stability_runs": 8,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 80,
            "max_depth": 4,
        },
    }
    X, y, names = _make_dataset_with_rho(
        config["dataset_rows"],
        config["population_x0_x2_correlation"],
        config["seed"],
    )
    analysis = _fit_and_explain(
        X,
        y,
        names,
        n_explain=config["explained_instances"],
        seed=config["seed"],
        lime_samples=config["lime_samples"],
        train_fraction=config["train_fraction"],
        background_rows=config["background_rows"],
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
    )
    predictor = scalar_predictor(analysis["model"])

    removals = []
    comprehensiveness = []
    for index in range(config["explained_instances"]):
        removals.append(
            removal_effect_correlation(
                predictor,
                analysis["explained"][index],
                analysis["shap"][index],
                analysis["background"],
            )
        )
        comprehensiveness.append(
            comprehensiveness_ratio(
                predictor,
                analysis["explained"][index],
                analysis["shap"][index],
                analysis["background"],
                top_k=config["top_k"],
                n_random=config["comprehensiveness_random_draws"],
                seed=index,
            )
        )

    before_rank, before_sign = _agreement(
        analysis["shap"], analysis["raw_lime"], top_k=config["top_k"]
    )
    after_rank, after_sign = _agreement(
        analysis["shap"],
        analysis["contribution_lime"],
        top_k=config["top_k"],
    )

    def explain_first_instance(seed: int):
        raw = lime_attributions(
            analysis["model"],
            analysis["explained"][:1],
            analysis["background"],
            feature_names=names,
            num_samples=config["lime_samples"],
            seed=seed,
        )
        return to_contribution_scale(
            raw, analysis["explained"][:1], analysis["background"]
        )[0]

    stability = cross_run_stability(
        explain_first_instance,
        n_runs=config["stability_runs"],
        top_k=config["top_k"],
    )

    profiles = {}
    endpoint_config = {
        "dataset_rows": 1200,
        "seed": 42,
        "explained_instances": 10,
        "display_features": 4,
        "population_correlations": {"clean": 0.0, "collinear": 0.85},
    }
    for key, rho in endpoint_config["population_correlations"].items():
        X_profile, y_profile, profile_names = _make_dataset_with_rho(
            endpoint_config["dataset_rows"], rho, endpoint_config["seed"]
        )
        profile = _fit_and_explain(
            X_profile,
            y_profile,
            profile_names,
            n_explain=endpoint_config["explained_instances"],
            seed=endpoint_config["seed"],
            lime_samples=config["lime_samples"],
            train_fraction=config["train_fraction"],
            background_rows=config["background_rows"],
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
        )
        displayed = endpoint_config["display_features"]
        profiles[key] = {
            "empirical_x0_x2_correlation": round(
                float(np.corrcoef(X_profile[:, 0], X_profile[:, 2])[0, 1]), 6
            ),
            "features": profile_names[:displayed],
            "shap_mean_absolute": _rounded(
                np.mean(np.abs(profile["shap"][:, :displayed]), axis=0)
            ),
            "lime_mean_absolute": _rounded(
                np.mean(
                    np.abs(profile["contribution_lime"][:, :displayed]), axis=0
                )
            ),
        }

    return {
        "schema_version": 1,
        "generated_by": "article/scripts/generate_figures.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "shap": version("shap"),
            "lime": version("lime"),
        },
        "config": config,
        "dataset": {
            "empirical_x0_x2_correlation": round(
                float(np.corrcoef(X[:, 0], X[:, 2])[0, 1]), 6
            ),
            "test_accuracy": round(analysis["test_accuracy"], 6),
        },
        "faithfulness": {
            "removal_effect_correlation": round(float(np.nanmean(removals)), 6),
            "comprehensiveness_ratio": round(
                float(np.nanmean(comprehensiveness)), 6
            ),
            "per_instance_removal_effect_correlation": _rounded(removals),
            "per_instance_comprehensiveness_ratio": _rounded(comprehensiveness),
        },
        "agreement": {
            "before_rank_corr": round(float(np.nanmean(before_rank)), 6),
            "after_rank_corr": round(float(np.nanmean(after_rank)), 6),
            "before_sign_disagreement": round(float(np.mean(before_sign)), 6),
            "after_sign_disagreement": round(float(np.mean(after_sign)), 6),
            "per_instance_before_rank_corr": _rounded(before_rank),
            "per_instance_after_rank_corr": _rounded(after_rank),
            "per_instance_before_sign_disagreement": _rounded(before_sign),
            "per_instance_after_sign_disagreement": _rounded(after_sign),
        },
        "stability": {
            "rank_corr": round(float(stability["rank_corr"]), 6),
            "topk_rank_corr": round(float(stability["topk_rank_corr"]), 6),
            "sign_agreement": round(float(stability["sign_agreement"]), 6),
            "topk_overlap": round(float(stability["topk_overlap"]), 6),
            "n_features": int(stability["n_features"]),
        },
        "conversion": {
            "features": names,
            "shap": _rounded(analysis["shap"][0]),
            "raw_lime": _rounded(analysis["raw_lime"][0]),
            "contribution_lime": _rounded(analysis["contribution_lime"][0]),
            "n_instances": config["explained_instances"],
            "shap_context": analysis["shap_context"],
        },
        "endpoints": {
            "config": endpoint_config,
            "profiles": profiles,
        },
    }


def conversion_payload(results: dict) -> dict:
    """Project the canonical bundle to the stable OJS data contract."""
    conversion = results["conversion"]
    agreement = results["agreement"]
    return {
        "features": conversion["features"],
        "shap": _rounded(conversion["shap"], 3),
        "raw_lime": _rounded(conversion["raw_lime"], 3),
        "contrib_lime": _rounded(conversion["contribution_lime"], 3),
        "before_rank_corr": round(float(agreement["before_rank_corr"]), 3),
        "after_rank_corr": round(float(agreement["after_rank_corr"]), 3),
        "before_sign_disagreement": round(
            float(agreement["before_sign_disagreement"]), 3
        ),
        "after_sign_disagreement": round(
            float(agreement["after_sign_disagreement"]), 3
        ),
        "n_instances": conversion["n_instances"],
        "shap_context": conversion["shap_context"],
    }


def results_digest(results: dict) -> str:
    encoded = json.dumps(
        results,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _save_png(fig, results: dict, *, dpi: int = 150) -> bytes:
    import matplotlib.pyplot as plt

    output = io.BytesIO()
    fig.savefig(
        output,
        format="png",
        dpi=dpi,
        metadata={"Software": "explaintrust", PNG_SOURCE_KEY: results_digest(results)},
    )
    plt.close(fig)
    return output.getvalue()


def render_conversion_figure(results: dict) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agreement = results["agreement"]
    values = [agreement["before_rank_corr"], agreement["after_rank_corr"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["SHAP vs raw LIME\n(wrong scale)", "SHAP vs contribution LIME\n(same scale)"],
        values,
        color=["#EF553B", "#636EFA"],
    )
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_ylabel("rank correlation (higher = agree)")
    ax.set_ylim(-0.5, 1.0)
    ax.set_title("Unit alignment removes part of the apparent disagreement")
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center", fontweight="bold")
    fig.tight_layout()
    return _save_png(fig, results)


def render_endpoints_figure(results: dict) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = results["endpoints"]["profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key in zip(axes, ["clean", "collinear"]):
        profile = profiles[key]
        x = np.arange(len(profile["features"]))
        width = 0.38
        ax.bar(
            x - width / 2,
            profile["shap_mean_absolute"],
            width,
            label="SHAP",
            color="#636EFA",
        )
        ax.bar(
            x + width / 2,
            profile["lime_mean_absolute"],
            width,
            label="LIME",
            color="#EF553B",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(profile["features"])
        empirical = profile["empirical_x0_x2_correlation"]
        ax.set_title(f"{key.capitalize()} profile (empirical r = {empirical:.2f})")
        ax.axhline(0, color="#999", linewidth=0.8)
    axes[0].legend()
    axes[0].set_ylabel("mean |attribution| (contribution scale)")
    fig.suptitle("Collinearity can shift model credit to the non-causal proxy x2")
    fig.tight_layout()
    return _save_png(fig, results)


def main() -> None:
    results = generate_article_results()
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments import build_all_results

    changed = build_all_results.build(check=False, root=ROOT)
    print(f"wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"updated {len(changed)} public result artifacts")


if __name__ == "__main__":
    main()
