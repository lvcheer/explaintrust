"""explaintrust — interactive demo.

Run from the repo root:

    streamlit run app/streamlit_app.py

Two modes:

* **Synthetic demo** — trains on a dataset with a known generative process
  (collinearity + interaction) so you can inspect how proxy features affect
  explainer agreement and stability.
* **Upload CSV** — point the same trust battery at your own tabular data: pick a
  target column, classify vs regress, train a model, and get a trust report.

In both modes the app explains instances with SHAP and LIME, runs the full
trust-metric battery, and renders a report.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

# Make the repo root importable so ``app.tabular_utils`` resolves whether the
# app is launched with ``streamlit run`` (cwd anywhere) or imported directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.tabular_utils import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    infer_task,
    make_model,
    prepare_tabular,
    read_csv_bytes,
    segment_feature,
)

from explaintrust import (
    FEATURE_ROLES,
    build_trust_report,
    lime_attributions,
    make_collinear_dataset,
    per_feature_reliability,
    prediction_output_space,
    scalar_predictor,
    shap_attributions,
    shift_distribution,
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


def _run_battery(
    model,
    X_explain,
    X_bg,
    names,
    n_explain,
    lime_samples,
    n_runs,
    seed,
    X_dist,
    seg_feature,
):
    """Run SHAP + LIME and the full trust-metric battery for a fitted model."""
    top_k = min(3, len(names))
    output_space = prediction_output_space(model)
    pred = scalar_predictor(model, output_space=output_space)

    shap_attr = shap_attributions(model, X_explain, X_background=X_bg, method="auto")
    lime_attr = lime_attributions(
        model,
        X_explain,
        X_bg,
        feature_names=names,
        num_samples=lime_samples,
        seed=seed,
        output_space=output_space,
    )
    lime_contrib = to_contribution_scale(lime_attr, X_explain, X_bg)

    # Faithfulness (SHAP, contribution-scale)
    removals = [
        removal_effect_correlation(pred, X_explain[i], shap_attr[i], X_bg)
        for i in range(n_explain)
    ]
    comps = [
        comprehensiveness_ratio(
            pred, X_explain[i], shap_attr[i], X_bg, top_k=top_k, seed=seed + i
        )
        for i in range(n_explain)
    ]
    mean_removal = float(np.nanmean(removals))
    mean_comp = float(np.nanmean(comps))

    # Faithfulness (LIME, gradient-scale)
    lime_infids = [
        infidelity(pred, X_explain[i], lime_attr[i], X_bg, seed=seed + i)
        for i in range(n_explain)
    ]
    mean_lime_infid = float(np.mean(lime_infids))

    # Sensitivity (one instance, re-explains each perturbation -> budget it)
    def explain_single(x):
        return shap_attributions(model, x.reshape(1, -1), X_background=X_bg, method="auto")[0]

    sens = max_sensitivity(explain_single, X_explain[0], X_bg, n_perturbations=10, seed=seed)

    # Run-to-run stability (stochastic LIME)
    def lime_seeded(seed: int):
        attr = lime_attributions(
            model,
            X_explain[:1],
            X_bg,
            feature_names=names,
            num_samples=max(500, lime_samples // 2),
            seed=seed,
            output_space=output_space,
        )
        return to_contribution_scale(attr, X_explain[:1], X_bg)[0]

    stability = cross_run_stability(lime_seeded, n_runs=n_runs, top_k=top_k)

    # SHAP vs LIME disagreement (contribution scale)
    dis_sign, dis_rank_topk, dis_rank_full, dis_topk, dis_mag = [], [], [], [], []
    for i in range(n_explain):
        d = explainer_disagreement(shap_attr[i], lime_contrib[i], top_k=top_k)
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

    # Exploratory subgroup consistency
    attr_dist = shap_attributions(model, X_dist, X_background=X_bg, method="auto")
    col = X_dist[:, seg_feature]
    segments = np.digitize(col, np.quantile(col, [0.33, 0.66]))
    distribution = cross_segment_stability(X_dist, segments, attr_dist, top_k=top_k)

    report = build_trust_report(
        removal_corr=mean_removal,
        comprehensiveness=mean_comp,
        lime_infidelity=mean_lime_infid,
        sensitivity_value=sens,
        stability=stability,
        disagreement=disagreement,
        distribution=distribution,
        top_k=top_k,
    )
    report.context.update(
        {
            "output_space": output_space,
            "class_index": 1 if hasattr(model, "predict_proba") else None,
            "faithfulness_instances": n_explain,
            "sensitivity_instances": 1,
            "stability_instances": 1,
            "subgroup_instances": len(X_dist),
            "background_instances": len(X_bg),
            "subgroup_feature": names[seg_feature],
            "seed": seed,
        }
    )

    return {
        "names": names,
        "shap_attr": shap_attr,
        "lime_contrib": lime_contrib,
        "report": report,
        "stability": stability,
        "distribution": distribution,
    }


@st.cache_data(show_spinner=False)
def run_pipeline(
    dataset: str,
    model_name: str,
    n_estimators: int,
    max_depth: int,
    n_explain: int,
    seed: int,
    lime_samples: int,
    n_runs: int,
):
    """Train on the synthetic dataset, explain, and compute every metric."""
    rng = np.random.default_rng(seed)

    if dataset == "collinear":
        X, y, names = make_collinear_dataset(n=1500, seed=seed)
        ground_truth = FEATURE_ROLES
    else:  # "clean" — no collinearity, so explanations should agree
        X, y, names = make_collinear_dataset(n=1500, seed=seed)
        X[:, 2] = rng.normal(0.0, 1.0, size=len(X))  # break x0/x2 collinearity
        ground_truth = {k: ("noise" if k == "x2" else v) for k, v in FEATURE_ROLES.items()}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    model = make_model("classification", model_name, n_estimators, max_depth, seed)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    X_bg = X_train[rng.choice(len(X_train), size=100, replace=False)]
    X_explain = X_test[:n_explain]

    # Explore subgroup heterogeneity inside a deliberately drifted dataset.
    X_dist, y_dist, _ = make_collinear_dataset(n=400, seed=seed + 7)
    X_dist, _, _ = shift_distribution(X_dist, y_dist, shift="x1_drift", seed=seed + 1)

    out = _run_battery(
        model, X_explain, X_bg, names, n_explain, lime_samples, n_runs, seed,
        X_dist=X_dist, seg_feature=1,
    )
    out.update(
        {
            "accuracy": accuracy,
            "task": "classification",
            "ground_truth": ground_truth,
            "X_explain": X_explain,
            "n_explain": n_explain,
        }
    )
    return out


def run_uploaded_pipeline(
    csv_bytes: bytes,
    target_col: str,
    task: str,
    model_name: str,
    n_estimators: int,
    max_depth: int,
    n_explain: int,
    seed: int,
    lime_samples: int,
    n_runs: int,
):
    """Run the same trust battery on the user's own CSV."""
    df = read_csv_bytes(csv_bytes)
    X, y, names, preprocessing = prepare_tabular(df, target_col, task)

    if len(X) < 40:
        raise ValueError(f"need at least 40 usable rows, got {len(X)}")
    if X.shape[1] < 2:
        raise ValueError(f"need at least 2 numeric feature columns, got {X.shape[1]}")
    if task == "classification" and len(np.unique(y)) != 2:
        raise ValueError("v0.1 supports binary classification only")
    if task == "classification" and np.min(np.bincount(y)) < 2:
        raise ValueError("each class needs at least 2 usable rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )

    model = make_model(task, model_name, n_estimators, max_depth, seed)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    rng = np.random.default_rng(seed)
    bg_size = min(100, len(X_train))
    X_bg = X_train[rng.choice(len(X_train), size=bg_size, replace=False)]
    n_explain = min(n_explain, len(X_test))
    X_explain = X_test[:n_explain]

    out = _run_battery(
        model, X_explain, X_bg, names, n_explain, lime_samples, n_runs, seed,
        X_dist=X_test, seg_feature=segment_feature(X_test),
    )
    out.update(
        {
            "accuracy": score,
            "task": task,
            "ground_truth": None,
            "X_explain": X_explain,
            "n_explain": n_explain,
            "preprocessing": preprocessing,
        }
    )
    out["report"].context["preprocessing"] = preprocessing
    return out


def main():
    st.set_page_config(page_title="explaintrust", page_icon="🔍", layout="wide")

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    st.sidebar.title("🔍 explaintrust")
    st.sidebar.caption(
        "Post-hoc explanations are easy to produce and easy to over-trust. "
        "This app runs them through a trust battery."
    )

    data_source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload CSV"], index=0)

    n_explain = st.sidebar.slider("Instances to explain", 1, 5, 1)
    seed = st.sidebar.number_input("Seed", 0, 100, 0)
    lime_samples = st.sidebar.slider("LIME samples / instance", 500, 5000, 2000, step=500)
    n_runs = st.sidebar.slider("Stability runs", 3, 10, 5)

    csv_bytes: Optional[bytes] = None
    if data_source == "Synthetic demo":
        dataset = st.sidebar.selectbox("Dataset", ["collinear", "clean"], index=0)
        model_name = st.sidebar.selectbox("Model", CLASSIFICATION_MODELS, index=0)
        n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100, step=10)
        max_depth = st.sidebar.slider("max_depth", 2, 10, 4)
    else:
        dataset = None
        uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            csv_bytes = uploaded.getvalue()
            preview = read_csv_bytes(csv_bytes)
            target_col = st.sidebar.selectbox("Target column (y)", list(preview.columns))
            inferred = infer_task(preview[target_col])
            task = st.sidebar.selectbox(
                "Task",
                ["classification", "regression"],
                index=0 if inferred == "classification" else 1,
            )
            model_choices = (
                CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
            )
            model_name = st.sidebar.selectbox("Model", model_choices, index=0)
            n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100, step=10)
            max_depth = st.sidebar.slider("max_depth", 2, 10, 4)

    # -----------------------------------------------------------------------
    # Run the pipeline
    # -----------------------------------------------------------------------
    if data_source == "Synthetic demo":
        with st.spinner("Training model, running SHAP + LIME + trust metrics…"):
            out = run_pipeline(
                dataset, model_name, n_estimators, max_depth, n_explain, int(seed), int(lime_samples), n_runs
            )
    else:
        if csv_bytes is None:
            st.info("👆 Upload a CSV file in the sidebar to run the trust battery on your own data.")
            st.stop()
        with st.spinner("Preparing data, running SHAP + LIME + trust metrics…"):
            try:
                out = run_uploaded_pipeline(
                    csv_bytes,
                    target_col,
                    task,
                    model_name,
                    n_estimators,
                    max_depth,
                    n_explain,
                    int(seed),
                    int(lime_samples),
                    n_runs,
                )
            except Exception as exc:  # noqa: BLE001 - surface a clean message to the user
                st.error(f"Could not run the trust battery: {exc}")
                st.stop()

    # -----------------------------------------------------------------------
    # Header + ground truth
    # -----------------------------------------------------------------------
    st.title("Can you trust this explanation?")
    st.markdown(
        "**explaintrust** evaluates SHAP and LIME explanations the way a careful "
        "reviewer would — not by how pretty they look, but by whether they are "
        "*faithful, stable, mutually consistent, and consistent across selected "
        "subgroups*."
    )

    c1, c2 = st.columns(2)
    if out["task"] == "regression":
        c1.metric("Model test R²", f"{out['accuracy']:.3f}")
    else:
        c1.metric("Model test accuracy", f"{out['accuracy']:.3f}")
    c2.metric("Explained instances", f"{out['n_explain']}")

    if data_source == "Upload CSV":
        prep = out["preprocessing"]
        with st.expander("Data preparation audit", expanded=False):
            st.json(prep)
            st.caption(
                "Non-numeric feature columns and rows with missing values are excluded in v0.1. "
                "Confirm that these exclusions are scientifically appropriate before interpreting the report."
            )

    if data_source == "Synthetic demo" and dataset == "collinear":
        with st.expander("ℹ️ Ground truth (synthetic data — only visible in the demo)", expanded=False):
            roles = pd.DataFrame(
                [{"feature": k, "role": v} for k, v in out["ground_truth"].items()]
            )
            st.dataframe(roles, width="stretch", hide_index=True)
            st.caption(
                "x0 and x1 enter the label-generating mechanism; x2 is a correlated proxy "
                "that may still be used by the predictive model. Compare how SHAP and LIME "
                "allocate model-level credit, but do not interpret either as causal evidence."
            )

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    report = out["report"]
    if report.overall.startswith("NO ISSUES"):
        verdict_color = "green"
    elif report.overall.startswith(("MIXED", "INSUFFICIENT")):
        verdict_color = "orange"
    else:
        verdict_color = "red"
    st.markdown(
        f"<div style='padding:1rem;border-radius:8px;border:1px solid {verdict_color};"
        f"background:{verdict_color}1a'>"
        f"<b style='font-size:1.1rem'>Verdict: {report.overall}</b><br>"
        f"<span style='color:#555'>{report.overall_reason}</span></div>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Scorecard
    # -----------------------------------------------------------------------
    st.subheader("Trust scorecard")
    score_df = report.as_dataframe()
    st.dataframe(score_df, width="stretch", hide_index=True)
    with st.expander("Method context", expanded=False):
        st.json(report.context)

    # -----------------------------------------------------------------------
    # SHAP vs LIME for the first instance
    # -----------------------------------------------------------------------
    st.subheader("SHAP vs LIME — where do they agree?")
    n_explain_used = out["n_explain"]
    instance_idx = st.slider("Instance", 0, n_explain_used - 1, 0) if n_explain_used > 1 else 0
    names = out["names"]
    s = out["shap_attr"][instance_idx]
    l = out["lime_contrib"][instance_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="SHAP (contribution)", x=names, y=s, marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="LIME (contribution)", x=names, y=l, marker_color="#EF553B"))
    fig.update_layout(barmode="group", height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    feat_df = per_feature_reliability(
        out["shap_attr"],
        out["lime_contrib"],
        stability_std=out["stability"]["std"],
        feature_names=names,
        instance_index=instance_idx,
    )
    report.feature_reliability = feat_df
    report.context["feature_reliability_instance"] = instance_idx
    st.markdown("Features sorted by SHAP–LIME gap (flagged = the two explainers disagree on this feature):")
    st.dataframe(feat_df, width="stretch", hide_index=True)
    st.download_button(
        "Download report JSON",
        data=report.to_json(),
        file_name="explaintrust-report.json",
        mime="application/json",
    )

    # -----------------------------------------------------------------------
    # Subgroup consistency
    # -----------------------------------------------------------------------
    st.subheader("Does the story stay consistent across selected subgroups?")
    dist = out["distribution"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-segment rank stability", f"{dist['rank_corr']:.3f}")
    c2.metric(f"Top-{report.context['top_k']} flip rate", f"{dist['topk_flip_rate']:.3f}")
    c3.metric("Segments", f"{len(dist['segment_ids'])}")

    seg_df = pd.DataFrame(dist["importances"], columns=names)
    seg_df.insert(0, "segment", [f"seg {s}" for s in dist["segment_ids"]])
    st.dataframe(seg_df, width="stretch", hide_index=True)
    st.caption(
        "Global feature importance (mean |attribution|) per subpopulation. "
        "A ranking change indicates subgroup heterogeneity; it does not by itself "
        "measure source-to-target distribution shift."
    )


if __name__ == "__main__":
    main()
