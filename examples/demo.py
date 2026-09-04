"""Headless end-to-end demo of explaintrust.

Run from the repo root:

    python3 examples/demo.py

This trains a model on the synthetic collinear dataset, explains a few
instances with both SHAP and LIME, runs every trust metric, and prints a
human-readable trust report. It is the reference pipeline the Streamlit app
wraps in a UI.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from explaintrust import (
    make_collinear_dataset,
    shift_distribution,
    scalar_predictor,
    shap_attributions,
    lime_attributions,
    to_contribution_scale,
    build_trust_report,
    per_feature_reliability,
)
from explaintrust.metrics import (
    infidelity,
    removal_effect_correlation,
    comprehensiveness_ratio,
    max_sensitivity,
    cross_run_stability,
    explainer_disagreement,
    cross_segment_stability,
)


def main() -> None:
    rng = np.random.default_rng(0)

    X, y, names = make_collinear_dataset(n=1500, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    print(f"model test accuracy: {model.score(X_test, y_test):.3f}\n")

    X_bg = X_train[rng.choice(len(X_train), size=100, replace=False)]
    n_explain = 5
    X_explain = X_test[:n_explain]

    print("computing SHAP (TreeExplainer) ...")
    shap_attr = shap_attributions(model, X_explain, method="tree")
    print("computing LIME (tabular) ...")
    lime_attr = lime_attributions(
        model, X_explain, X_bg, feature_names=names, num_samples=2000, seed=0
    )

    # LIME weights are slopes; convert to SHAP-comparable contribution units
    # before any cross-explainer comparison.
    lime_contrib = to_contribution_scale(lime_attr, X_explain, X_bg)
    print(f"shap {shap_attr.shape}, lime {lime_attr.shape}\n")

    pred = scalar_predictor(model)

    # --- faithfulness: SHAP (contribution-scale) ---------------------------
    removals, comps = [], []
    for i in range(n_explain):
        removals.append(removal_effect_correlation(pred, X_explain[i], shap_attr[i], X_bg))
        comps.append(comprehensiveness_ratio(pred, X_explain[i], shap_attr[i], X_bg, top_k=3, seed=i))
    mean_removal = float(np.nanmean(removals))
    mean_comp = float(np.nanmean(comps))
    print(f"SHAP removal-effect corr (higher better): {mean_removal:.3f}")
    print(f"SHAP comprehensiveness ratio (higher, >1 good): {mean_comp:.3f}\n")

    # --- faithfulness: LIME (gradient-scale) -------------------------------
    lime_infids = [infidelity(pred, X_explain[i], lime_attr[i], X_bg, seed=i) for i in range(n_explain)]
    mean_lime_infid = float(np.mean(lime_infids))
    print(f"LIME infidelity (lower better): {mean_lime_infid:.5f}\n")

    # --- sensitivity (expensive: re-explains each perturbation) ------------
    def tree_explainer_single(x):
        return shap_attributions(model, x.reshape(1, -1), method="tree")[0]

    sens = max_sensitivity(tree_explainer_single, X_explain[0], X_bg, n_perturbations=15, seed=0)
    print(f"max sensitivity (instance 0, lower better): {sens:.3f}\n")

    # --- run-to-run stability (stochastic explainer: LIME) ----------------
    def lime_explainer_seeded(seed: int):
        attr = lime_attributions(
            model, X_explain[:1], X_bg, feature_names=names, num_samples=1000, seed=seed
        )
        return to_contribution_scale(attr, X_explain[:1], X_bg)[0]

    stability = cross_run_stability(lime_explainer_seeded, n_runs=8, top_k=3)
    print(
        f"LIME run-to-run: rank_corr={stability['rank_corr']:.3f}, "
        f"topk_sign_agreement={stability['sign_agreement']:.3f}, "
        f"top3_overlap={stability.get('topk_overlap', float('nan')):.3f}\n"
    )

    # --- SHAP vs LIME disagreement (on the same contribution scale) --------
    dis_sign, dis_rank_topk, dis_rank_full, dis_topk, dis_mag = [], [], [], [], []
    for i in range(n_explain):
        d = explainer_disagreement(shap_attr[i], lime_contrib[i], top_k=3)
        dis_sign.append(d["sign_disagreement"])
        dis_rank_topk.append(d["topk_rank_corr"])
        dis_rank_full.append(d["rank_corr"])
        dis_topk.append(d["topk_overlap"])
        dis_mag.append(d["magnitude_disagreement"])
    disagreement = {
        "sign_disagreement": float(np.mean(dis_sign)),
        "topk_rank_corr": float(np.nanmean(dis_rank_topk)),
        "rank_corr": float(np.nanmean(dis_rank_full)),
        "topk_overlap": float(np.mean(dis_topk)),
        "magnitude_disagreement": float(np.nanmean(dis_mag)),
    }
    print(
        f"SHAP vs LIME: sign_disagreement={disagreement['sign_disagreement']:.3f}, "
        f"rank_corr(top-3)={disagreement['topk_rank_corr']:.3f}, "
        f"rank_corr(all)={disagreement['rank_corr']:.3f}, "
        f"top3_overlap={disagreement['topk_overlap']:.3f}, "
        f"magnitude_disagreement={disagreement['magnitude_disagreement']:.3f}\n"
    )

    # --- subgroup consistency inside a drifted test set --------------------
    X_shift, y_shift, _ = make_collinear_dataset(n=400, seed=7)
    X_shift, _, _ = shift_distribution(X_shift, y_shift, shift="x1_drift", seed=1)
    attr_shift = shap_attributions(model, X_shift, method="tree")

    segments = np.digitize(X_shift[:, 1], np.quantile(X_shift[:, 1], [0.33, 0.66]))
    distribution = cross_segment_stability(X_shift, segments, attr_shift, top_k=3)
    print(
        f"cross-segment (shifted data): rank_corr={distribution['rank_corr']:.3f}, "
        f"top3_flip_rate={distribution['topk_flip_rate']:.3f}\n"
    )

    # --- build report ------------------------------------------------------
    report = build_trust_report(
        removal_corr=mean_removal,
        comprehensiveness=mean_comp,
        lime_infidelity=mean_lime_infid,
        sensitivity_value=sens,
        stability=stability,
        disagreement=disagreement,
        distribution=distribution,
        top_k=3,
    )
    print("=" * 70)
    print(f"TRUST VERDICT: {report.overall}")
    print(report.overall_reason)
    print("=" * 70)
    print(report.as_dataframe().to_string(index=False))

    print("\n--- per-feature reliability (instance 0) ---")
    feat_df = per_feature_reliability(
        shap_attr, lime_contrib, stability_std=stability["std"], feature_names=names, instance_index=0
    )
    print(feat_df.to_string(index=False))


if __name__ == "__main__":
    main()
