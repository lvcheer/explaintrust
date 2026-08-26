"""explaintrust — interactive demo.

Run from the repo root:

    streamlit run app/streamlit_app.py

The app trains a model, explains instances with SHAP and LIME, runs the full
trust-metric battery, and renders a report. On the synthetic dataset it also
shows the (normally hidden) ground-truth feature roles, so you can see the
metrics catch real problems (collinearity, explainer disagreement) that a plain
SHAP plot would never reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from explaintrust import (
    FEATURE_ROLES,
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


st.set_page_config(page_title="explaintrust", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner=False)
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
    """Train a model, explain, and compute every trust metric (cached)."""
    rng = np.random.default_rng(seed)

    if dataset == "collinear":
        X, y, names = make_collinear_dataset(n=1500, seed=seed)
        ground_truth = FEATURE_ROLES
    else:  # "clean" — no collinearity, so explanations should agree
        X, y, names = make_collinear_dataset(n=1500, seed=seed)
        X[:, 2] = rng.normal(0.0, 1.0, size=len(X))  # break x0/x2 collinearity
        ground_truth = {k: ("noise" if k == "x2" else v) for k, v in FEATURE_ROLES.items()}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed)

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    X_bg = X_train[rng.choice(len(X_train), size=100, replace=False)]
    X_explain = X_test[:n_explain]

    shap_attr = shap_attributions(model, X_explain, method="tree")
    lime_attr = lime_attributions(
        model, X_explain, X_bg, feature_names=names, num_samples=lime_samples, seed=seed
    )
    lime_contrib = to_contribution_scale(lime_attr, X_explain, X_bg)

    pred = scalar_predictor(model)

    # Faithfulness (SHAP, contribution-scale)
    removals = [removal_effect_correlation(pred, X_explain[i], shap_attr[i], X_bg) for i in range(n_explain)]
    comps = [comprehensiveness_ratio(pred, X_explain[i], shap_attr[i], X_bg, top_k=3, seed=seed + i) for i in range(n_explain)]
    mean_removal = float(np.nanmean(removals))
    mean_comp = float(np.nanmean(comps))

    # Faithfulness (LIME, gradient-scale)
    lime_infids = [infidelity(pred, X_explain[i], lime_attr[i], X_bg, seed=seed + i) for i in range(n_explain)]
    mean_lime_infid = float(np.mean(lime_infids))

    # Sensitivity (one instance, re-explains each perturbation -> budget it)
    def tree_single(x):
        return shap_attributions(model, x.reshape(1, -1), method="tree")[0]

    sens = max_sensitivity(tree_single, X_explain[0], X_bg, n_perturbations=10, seed=seed)

    # Run-to-run stability (stochastic LIME)
    def lime_seeded(seed: int):
        return lime_attributions(
            model, X_explain[:1], X_bg, feature_names=names, num_samples=max(500, lime_samples // 2), seed=seed
        )[0]

    stability = cross_run_stability(lime_seeded, n_runs=n_runs, top_k=3)

    # SHAP vs LIME disagreement (contribution scale)
    dis_sign, dis_rank, dis_topk = [], [], []
    for i in range(n_explain):
        d = explainer_disagreement(shap_attr[i], lime_contrib[i], top_k=3)
        dis_sign.append(d["sign_disagreement"])
        dis_rank.append(d["rank_corr"])
        dis_topk.append(d["topk_overlap"])
    disagreement = {
        "sign_disagreement": float(np.mean(dis_sign)),
        "rank_corr": float(np.nanmean(dis_rank)),
        "topk_overlap": float(np.mean(dis_topk)),
    }

    # Distribution verification on a drifted set
    X_shift, y_shift, _ = make_collinear_dataset(n=400, seed=seed + 7)
    X_shift, _, _ = shift_distribution(X_shift, y_shift, shift="x1_drift", seed=seed + 1)
    attr_shift = shap_attributions(model, X_shift, method="tree")
    segments = np.digitize(X_shift[:, 1], np.quantile(X_shift[:, 1], [0.33, 0.66]))
    distribution = cross_segment_stability(X_shift, segments, attr_shift, top_k=3)

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

    return {
        "names": names,
        "accuracy": accuracy,
        "ground_truth": ground_truth,
        "X_explain": X_explain,
        "shap_attr": shap_attr,
        "lime_contrib": lime_contrib,
        "report": report,
        "stability": stability,
        "distribution": distribution,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🔍 explaintrust")
st.sidebar.caption("Post-hoc explanations are easy to produce and easy to over-trust. "
                   "This app runs them through a trust battery.")

dataset = st.sidebar.selectbox("Dataset", ["collinear", "clean"], index=0)
model_name = st.sidebar.selectbox("Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"], index=0)
n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100, step=10)
max_depth = st.sidebar.slider("max_depth", 2, 10, 4)
n_explain = st.sidebar.slider("Instances to explain", 1, 5, 1)
seed = st.sidebar.number_input("Seed", 0, 100, 0)
lime_samples = st.sidebar.slider("LIME samples / instance", 500, 5000, 2000, step=500)
n_runs = st.sidebar.slider("Stability runs", 3, 10, 5)

with st.spinner("Training model, running SHAP + LIME + trust metrics…"):
    out = run_pipeline(dataset, model_name, n_estimators, max_depth, n_explain, int(seed), int(lime_samples), n_runs)

# ---------------------------------------------------------------------------
# Header + ground truth
# ---------------------------------------------------------------------------
st.title("Can you trust this explanation?")
st.markdown(
    "**explaintrust** evaluates SHAP and LIME explanations the way a careful "
    "reviewer would — not by how pretty they look, but by whether they are "
    "*faithful, stable, mutually consistent, and robust across the data "
    "distribution*."
)

c1, c2 = st.columns(2)
c1.metric("Model test accuracy", f"{out['accuracy']:.3f}")
c2.metric("Explained instances", f"{n_explain}")

if dataset == "collinear":
    with st.expander("ℹ️ Ground truth (synthetic data — only visible in the demo)", expanded=False):
        roles = pd.DataFrame(
            [{"feature": k, "role": v} for k, v in out["ground_truth"].items()]
        )
        st.dataframe(roles, width="stretch", hide_index=True)
        st.caption(
            "x0 and x1 truly drive the label; x2 is collinear with x0 (not causal). "
            "A good trust report should flag that SHAP and LIME fight over x0 vs x2, "
            "and that the explanation is unstable — even though the model is fine."
        )

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
report = out["report"]
verdict_color = {"TRUSTWORTHY": "green", "MIXED": "orange"}.get(
    "TRUSTWORTHY" if report.overall.startswith("TRUSTWORTHY") else "MIXED", "red"
)
st.markdown(
    f"<div style='padding:1rem;border-radius:8px;border:1px solid {verdict_color};"
    f"background:{verdict_color}1a'>"
    f"<b style='font-size:1.1rem'>Verdict: {report.overall}</b><br>"
    f"<span style='color:#555'>{report.overall_reason}</span></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
st.subheader("Trust scorecard")
score_df = report.as_dataframe()
st.dataframe(score_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# SHAP vs LIME for the first instance
# ---------------------------------------------------------------------------
st.subheader("SHAP vs LIME — where do they agree?")
instance_idx = st.slider("Instance", 0, n_explain - 1, 0) if n_explain > 1 else 0
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
st.markdown("Features sorted by SHAP–LIME gap (flagged = the two explainers disagree on this feature):")
st.dataframe(feat_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Distribution verification
# ---------------------------------------------------------------------------
st.subheader("Does the story hold across the distribution?")
dist = out["distribution"]
c1, c2, c3 = st.columns(3)
c1.metric("Cross-segment rank stability", f"{dist['rank_corr']:.3f}")
c2.metric("Top-3 flip rate", f"{dist['topk_flip_rate']:.3f}")
c3.metric("Segments", f"{len(dist['segment_ids'])}")

seg_df = pd.DataFrame(dist["importances"], columns=names)
seg_df.insert(0, "segment", [f"seg {s}" for s in dist["segment_ids"]])
st.dataframe(seg_df, width="stretch", hide_index=True)
st.caption("Global feature importance (mean |attribution|) per subpopulation. "
           "If the ranking flips across segments, the explanation does not generalize.")
