"""explaintrust — interactive demo.

Run from the repo root:

    streamlit run app/streamlit_app.py

The page follows explicit data, split, training, explanation and quality-analysis
stages. Synthetic presets, UCI Adult/Diabetes and uploaded CSVs are supported.
Users select SHAP, LIME or both; unselected checks remain unavailable.
The legacy pipeline helpers below remain available to existing callers/tests.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Make the repo root importable so ``app.tabular_utils`` resolves whether the
# app is launched with ``streamlit run`` (cwd anywhere) or imported directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.i18n import st

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
    methods=("SHAP", "LIME"),
    precomputed=None,
    quality_config=None,
):
    """Run SHAP + LIME and the full trust-metric battery for a fitted model."""
    # Keep removal and set comparisons informative on two/three-feature data.
    config = {"top_k": min(3, max(1, len(names) - 1)), "sensitivity_samples": 1,
              "sensitivity_perturbations": 10, "sensitivity_radius": 0.1,
              "infidelity_perturbations": 200, "comprehensiveness_random": 20}
    config.update(quality_config or {})
    top_k = config["top_k"]
    shap_samples = (precomputed or {}).get("shap_samples", 100)
    output_space = prediction_output_space(model)
    pred = scalar_predictor(model, output_space=output_space)

    use_shap, use_lime = "SHAP" in methods, "LIME" in methods
    if not (use_shap or use_lime):
        raise ValueError("select at least one explanation method")
    if use_shap:
        if precomputed is not None:
            shap_attr, shap_context = precomputed["shap_attr"], precomputed["shap_context"]
        else:
            shap_attr, shap_context = shap_attributions(
                model, X_explain, X_background=X_bg, method="auto", nsamples=shap_samples, seed=seed, return_context=True)
    else:
        shap_attr, shap_context = np.full_like(X_explain, np.nan), None

    def explain_lime(run_seed):
        return lime_attributions(
            model,
            X_explain,
            X_bg,
            feature_names=names,
            num_samples=lime_samples,
            seed=run_seed,
            output_space=output_space,
        )

    lime_attr = (precomputed["lime_attr"] if precomputed is not None else explain_lime(seed)) if use_lime else np.full_like(X_explain, np.nan)
    lime_contrib = to_contribution_scale(lime_attr, X_explain, X_bg)

    # Faithfulness (SHAP, contribution-scale)
    removals = [
        removal_effect_correlation(pred, X_explain[i], shap_attr[i], X_bg) if use_shap else np.nan
        for i in range(n_explain)
    ]
    comps = [
        comprehensiveness_ratio(
            pred, X_explain[i], shap_attr[i], X_bg, top_k=top_k, n_random=config["comprehensiveness_random"], seed=seed + i
        ) if use_shap else np.nan
        for i in range(n_explain)
    ]
    mean_removal = float(np.nanmean(removals)) if use_shap else np.nan
    mean_comp = float(np.nanmean(comps)) if use_shap else np.nan

    # Faithfulness (LIME, gradient-scale)
    lime_infids = [
        infidelity(pred, X_explain[i], lime_attr[i], X_bg, n_perturbations=config["infidelity_perturbations"], seed=seed + i) if use_lime else np.nan
        for i in range(n_explain)
    ]
    mean_lime_infid = float(np.mean(lime_infids))

    # Sensitivity (one instance, re-explains each perturbation -> budget it)
    def explain_single(x):
        return shap_attributions(model, x.reshape(1, -1), X_background=X_bg, method="auto", nsamples=shap_samples, seed=seed)[0]

    sensitivity_count = min(n_explain, config["sensitivity_samples"]) if use_shap else 0
    sensitivity_values = [max_sensitivity(explain_single, X_explain[i], X_bg,
        n_perturbations=config["sensitivity_perturbations"], radius=config["sensitivity_radius"], seed=seed+i)
        for i in range(sensitivity_count)]
    sens = float(np.mean(sensitivity_values)) if sensitivity_values else None

    # Repeat the same batch/configuration so each row keeps its place in LIME's
    # random stream. The displayed explanation counts as the first run.
    stability_seeds = [seed + run for run in range(n_runs)]
    contribution_runs = [lime_contrib]
    for run_seed in (stability_seeds[1:] if use_lime else []):
        contribution_runs.append(
            to_contribution_scale(explain_lime(run_seed), X_explain, X_bg)
        )
    stability_by_instance = []
    for i in range(n_explain):
        result = cross_run_stability(
            lambda seed, instance=i: contribution_runs[seed][instance],
            n_runs=n_runs, top_k=top_k,
        ) if use_lime else {"rank_corr": np.nan, "topk_rank_corr": np.nan,
                            "sign_agreement": np.nan, "topk_overlap": np.nan,
                            "std": np.full(len(names), np.nan)}
        stability_by_instance.append(result)
    # Keep local std vectors separate; they must never be averaged and then
    # reused as if they belonged to an individual instance.
    stability = {
        key: float(np.mean([result[key] for result in stability_by_instance]))
        for key in ("rank_corr", "topk_rank_corr", "sign_agreement", "topk_overlap")
    }
    stability["n_features"] = len(names)

    # SHAP vs LIME disagreement (contribution scale)
    dis_sign, dis_rank_topk, dis_rank_full, dis_topk, dis_mag = [], [], [], [], []
    for i in range(n_explain):
        d = explainer_disagreement(shap_attr[i], lime_contrib[i], top_k=top_k) if use_shap and use_lime else {k: np.nan for k in ("sign_disagreement", "topk_rank_corr", "rank_corr", "topk_overlap", "magnitude_disagreement")}
        dis_sign.append(d["sign_disagreement"])
        dis_rank_topk.append(d["topk_rank_corr"])
        dis_rank_full.append(d["rank_corr"])
        dis_topk.append(d["topk_overlap"])
        dis_mag.append(d["magnitude_disagreement"])
    disagreement = {
        "sign_disagreement": float(np.mean(dis_sign)),
        "topk_rank_corr": float(np.nanmean(dis_rank_topk)) if top_k > 1 and use_shap and use_lime else float("nan"),
        "rank_corr": float(np.nanmean(dis_rank_full)) if use_shap and use_lime else np.nan,
        "topk_overlap": float(np.mean(dis_topk)),
        "magnitude_disagreement": float(np.nanmean(dis_mag)) if use_shap and use_lime else np.nan,
    }

    # Exploratory subgroup consistency
    if use_shap:
        attr_dist = shap_attributions(model, X_dist, X_background=X_bg, method="auto", nsamples=shap_samples, seed=seed)
        col = X_dist[:, seg_feature]
        segments = np.digitize(col, np.quantile(col, [0.33, 0.66]))
        distribution = cross_segment_stability(X_dist, segments, attr_dist, top_k=top_k)
    else:
        distribution = {"rank_corr": np.nan, "topk_flip_rate": np.nan,
                        "segment_ids": [], "importances": np.empty((0, len(names)))}

    report = build_trust_report(
        removal_corr=mean_removal,
        comprehensiveness=mean_comp,
        lime_infidelity=mean_lime_infid,
        sensitivity_value=sens,
        stability=stability,
        disagreement=disagreement,
        distribution=distribution,
        top_k=top_k,
        n_features=len(names),
    )
    report.context.update(
        {
            "output_space": output_space,
            "shap": shap_context,
            "selected_methods": list(methods),
            "lime_reference": {
                "background_sha256": shap_context["background"]["sha256"] if shap_context else None,
                "centering": "mean of the same background rows used by SHAP",
                "contribution_comparison": "local slope times deviation; approximate for nonlinear models",
            },
            "class_index": 1 if hasattr(model, "predict_proba") else None,
            "faithfulness_instances": n_explain,
            "sensitivity_instances": sensitivity_count,
            "quality_config": config,
            "sensitivity_values": sensitivity_values,
            "stability_instances": n_explain if use_lime else 0,
            "stability_config": {
                "explainer": "LIME",
                "attribution_scale": "contribution",
                "num_samples": lime_samples,
                "n_runs": n_runs if use_lime else 0,
                "seeds": stability_seeds if use_lime else [],
                "main_explanation_included": use_lime,
                "instance_indices": list(range(n_explain)),
                "aggregation": "arithmetic mean across explained instances",
            },
            "subgroup_instances": len(X_dist) if use_shap else 0,
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
        "stability_by_instance": stability_by_instance,
        "distribution": distribution,
    }


def _update_feature_reliability(out, instance_index):
    """Keep the selected table and exported report tied to the same instance."""
    table = per_feature_reliability(
        out["shap_attr"],
        out["lime_contrib"],
        stability_std=out["stability_by_instance"][instance_index]["std"],
        feature_names=out["names"],
        instance_index=instance_index,
    )
    out["report"].feature_reliability = table
    out["report"].context["feature_reliability_instance"] = instance_index
    return table


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
    """Analyze synthetic data with shared backgrounds and per-row stability.

    Reports separate scored checks from descriptive method/subgroup diagnostics.
    """
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



_METRIC_COPY = [
    ("SHAP removal", "SHAP 移除效果", "特征归因是否与移除该特征后的模型输出变化一致。"),
    ("SHAP comprehensiveness", "SHAP 关键特征检查", "移除重要特征是否比随机移除同样数量的特征影响更大。"),
    ("LIME local", "LIME 局部拟合", "局部解释能否近似模型的输出变化；数值越低，误差越小。"),
    ("Max sensitivity", "解释敏感度", "各已评估样本的扰动最大归因变化的均值；覆盖数量见摘要。"),
    ("Run-to-run rank", "重复运行排名稳定性", "不同随机种子下，重要特征的排名是否一致。"),
    ("Run-to-run sign", "重复运行方向稳定性", "不同随机种子下，重要特征的贡献方向是否一致。"),
    ("SHAP vs LIME sign", "方法间方向差异", "两种方法给出不同贡献方向的特征比例。"),
    ("SHAP vs LIME rank", "方法间排名一致性", "重要特征排名有多相似；相似程度不代表解释质量。"),
    ("SHAP vs LIME top-", "方法间重要特征重合", "两种方法选出的重要特征有多少重合。"),
    ("SHAP vs LIME magnitude", "方法间贡献大小差异", "重要特征的相对归因差值；需要结合方法与任务解释。"),
    ("Cross-segment", "子群排名一致性", "各子群的重要特征排序有多相似。"),
    ("Top-", "子群重要特征变化率", "与参考子群相比，重要特征集合发生变化的比例。"),
]


def _display_metrics(metrics):
    statuses = {"good": "通过", "warn": "需关注", "bad": "未通过", "info": "缺少结果",
                "not_applicable": "不适用", "descriptive": "描述性指标"}
    rows = []
    for metric in metrics:
        title, explanation = next(((title, copy) for prefix, title, copy in _METRIC_COPY
                                   if metric.name.startswith(prefix)), (metric.name, metric.explanation))
        if metric.verdict == "not_applicable":
            explanation = "当前特征数或 top-k 设置下，此项比较不适用；详见原始指标说明。"
        elif metric.verdict == "info":
            explanation = "未计算或返回了无法解释的非有限值；详见原始指标说明。"
        value = f"{metric.value:.3f}" if np.isfinite(metric.value) else (
            "+∞" if np.isposinf(metric.value) else "−∞" if np.isneginf(metric.value) else "—")
        rows.append({"检查项": title, "状态": statuses.get(metric.verdict, metric.verdict),
                     "数值": value, "如何理解": explanation})
    return pd.DataFrame(rows)

def main():
    st.set_page_config(page_title="explaintrust", page_icon="🔍", layout="wide")
    from app.workflow import render_workflow
    result = render_workflow(_run_battery)
    if result is not None:
        _render_results(result, "Upload CSV", None, result["lime_samples"], result["n_runs"])


def _render_results(out, data_source, dataset, lime_samples, n_runs):
    # Results: conclusion first, then evidence and optional detail.
    report = out["report"]
    scored = [m for m in report.metric_results if m.role == "scored"]
    flagged = [m for m in scored if m.verdict in ("bad", "warn", "info")]
    passed = sum(m.verdict == "good" for m in scored)
    applicable = sum(m.included_in_overall for m in scored)
    st.title("解释评估")
    st.caption("explaintrust · 了解解释的表现，以及哪些结果需要进一步核查")

    with st.container(border=True):
        if report.overall.startswith("NO ISSUES"):
            st.success("当前检查未发现问题")
        elif report.overall.startswith("INSUFFICIENT"):
            st.warning("评估证据不完整，请先查看缺失与失败的检查")
        elif report.overall.startswith("MIXED"):
            st.warning("部分检查需要关注")
        else:
            st.error("多项检查未通过")
        st.write("结论来自忠实性、敏感度和重复运行稳定性检查；方法间分歧与子群差异单独呈现。")
        c1, c2, c3 = st.columns(3)
        c1.metric("通过的检查", f"{passed} / {applicable}")
        c2.metric("已解释样本", str(out["n_explain"]))
        c3.metric("模型测试 R²" if out["task"] == "regression" else "模型测试准确率",
                  f"{out['accuracy']:.3f}")
        st.caption(
            f"覆盖范围：忠实性 {report.context['faithfulness_instances']} 个样本 · "
            f"稳定性 {report.context['stability_instances']} 个样本 · "
            f"敏感度 {report.context['sensitivity_instances']} 个样本。"
            "通过检查不等于证明解释正确；模型预测表现也不等于解释质量。"
        )

    checks_tab, instance_tab, differences_tab, details_tab = st.tabs(
        ["质量检查", "单个样本", "方法与子群差异", "评估详情与导出"]
    )
    with checks_tab:
        st.subheader("先看需要关注的检查")
        if flagged:
            st.dataframe(_display_metrics(flagged), width="stretch", hide_index=True)
        else:
            st.info("当前没有失败、提醒或缺失的质量检查。")
        with st.expander("查看全部质量检查", expanded=not bool(flagged)):
            st.dataframe(_display_metrics(scored), width="stretch", hide_index=True)
        st.caption("状态根据当前配置阈值生成；不适用项不计入检查总数。具体阈值与原始说明可在评估详情查看。")

    with instance_tab:
        st.subheader("这个样本的解释是什么？")
        n_explain_used = out["n_explain"]
        instance_idx = st.selectbox("选择样本", list(range(n_explain_used)),
                                   format_func=lambda i: f"样本 {i + 1}")
        names = out["names"]
        s, l = out["shap_attr"][instance_idx], out["lime_contrib"][instance_idx]
        st.caption("柱高表示对模型输出的贡献；正负表示贡献方向。两种方法分配贡献的差异需要结合模型与数据理解。")
        fig = go.Figure()
        if "SHAP" in report.context.get("selected_methods", ["SHAP", "LIME"]):
            fig.add_trace(go.Bar(name="SHAP", x=names, y=s, marker_color="#4F46E5"))
        if "LIME" in report.context.get("selected_methods", ["SHAP", "LIME"]):
            fig.add_trace(go.Bar(name="LIME", x=names, y=l, marker_color="#0D9488"))
        fig.update_layout(barmode="group", height=380, margin=dict(t=20, b=20),
                          yaxis_title="归因贡献", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        if "feature_dictionary" in out:
            st.dataframe(out["feature_dictionary"], width="stretch", hide_index=True,
                         translate_values=report.context.get("dataset") != "上传 CSV")
        both_methods = set(report.context.get("selected_methods", ["SHAP", "LIME"])) == {"SHAP", "LIME"}
        if both_methods:
            feat_df = _update_feature_reliability(out, instance_idx)
            with st.expander("查看逐特征对比"):
                st.caption("按 SHAP–LIME 绝对差值排序；差异标记不等同于解释错误。")
                st.dataframe(feat_df.rename(columns={"feature": "特征", "shap": "SHAP",
                             "lime": "LIME", "abs_gap": "绝对差值", "agree": "未触发差异标记",
                             "signal_to_noise": "信噪比"}), width="stretch", hide_index=True)
        else:
            report.feature_reliability = None
            st.info("当前只选择了一种解释方法，因此不生成方法间差异标记。")

    with differences_tab:
        st.subheader("差异描述，不参与总体判定")
        st.caption("方法可能关注不同的特征，子群也可能存在真实差异。差异较大不直接意味着不可靠，一致也不代表正确。")
        descriptive = [m for m in report.metric_results if m.role == "descriptive"]
        st.dataframe(_display_metrics(descriptive), width="stretch", hide_index=True)
        dist = out["distribution"]
        with st.expander("查看子群特征重要性"):
            st.caption(f"分组特征：{report.context['subgroup_feature']} · "
                       f"覆盖 {report.context['subgroup_instances']} 个样本 · "
                       f"{len(dist['segment_ids'])} 个子群。数值为各子群的平均绝对归因。")
            seg_df = pd.DataFrame(dist["importances"], columns=names)
            seg_df.insert(0, "子群", [f"组 {v}" for v in dist["segment_ids"]])
            st.dataframe(seg_df, width="stretch", hide_index=True)
            st.caption("这里描述选定子群之间的异质性，不是源数据到目标数据的分布漂移检验。")

    with details_tab:
        st.subheader("评估范围与计算设置")
        st.write(f"已选方法：{', '.join(report.context.get('selected_methods', ['SHAP', 'LIME']))}；"
                 f"背景样本 {report.context['background_instances']} 行。")
        if "LIME" in report.context.get("selected_methods", ["SHAP", "LIME"]):
            st.caption(f"LIME 每个样本采样 {lime_samples:,} 次，稳定性重复 {n_runs} 次。")
        with st.expander("原始结论、指标与阈值"):
            st.write(report.overall)
            st.write(report.overall_reason)
            st.dataframe(report.as_dataframe(), width="stretch", hide_index=True)
        with st.expander("完整方法与计算参数"):
            st.json(report.context)
        if data_source == "Upload CSV" and "preprocessing" in out:
            with st.expander("数据预处理记录"):
                st.json(out["preprocessing"])
                st.caption("当前版本排除非数值特征列和含缺失值的行，请结合任务核查这些排除是否合适。")
        if data_source == "Synthetic demo" and dataset == "collinear":
            with st.expander("合成数据的已知机制"):
                st.dataframe(pd.DataFrame([{"特征": k, "角色": v}
                                          for k, v in out["ground_truth"].items()]),
                             width="stretch", hide_index=True)
                st.caption("x0、x1 参与标签生成；x2 是相关代理特征。SHAP 与 LIME 描述模型贡献，不提供因果证据。")
        st.download_button("下载完整报告 JSON", data=report.to_json(),
                           file_name="explaintrust-report.json", mime="application/json")
        st.caption("报告保留原始指标和计算参数。" + (f"逐特征对比对应当前样本 {instance_idx + 1}。" if both_methods else "未选择的方法以缺失检查呈现。"))


if __name__ == "__main__":
    main()
