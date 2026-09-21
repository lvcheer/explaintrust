# explaintrust

[![CI](https://github.com/lvcheer/explaintrust/actions/workflows/ci.yml/badge.svg)](https://github.com/lvcheer/explaintrust/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/tag/lvcheer/explaintrust?label=version)](https://github.com/lvcheer/explaintrust/tags)
[![Python](https://img.shields.io/badge/python-3.9--3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

中文说明见[下方](#中文说明) · A Chinese version of this README is [below](#中文说明).

**explaintrust tests whether tabular SHAP and LIME explanations behave
faithfully and reproducibly under perturbation, repeated runs, and subgroup
changes.**

It scores faithfulness, sensitivity, and run-to-run stability. Differences
between explainers and across subgroups remain visible as descriptive evidence;
they do not silently become quality labels. A passing report means that the
configured checks found no failure, not that an explanation is true or causal.

**[Live demo · 在线演示](https://lvcheer-explaintrust-appstreamlit-app-x7k48h.streamlit.app/)** — Run the five-step workflow in English or Chinese without installing anything.

[Project page](article/index.qmd) ·
[Two-page technical brief](output/pdf/explaintrust-technical-brief.pdf) ·
[Citation](CITATION.cff)

## 60-second start

Until the public PyPI package is verified, install the reviewed repository
version directly:

```bash
git clone https://github.com/lvcheer/explaintrust.git
cd explaintrust
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/demo.py
```

The seeded demo trains a model, computes SHAP and LIME explanations, evaluates
all metric families, and prints a report. For the interactive app, install
with `python -m pip install -e ".[app]"` and run
`streamlit run app/streamlit_app.py`.

## Example trust report

An abridged headless-demo report looks like this:

```text
TRUST VERDICT: MIXED — investigate before trusting
good  SHAP removal-effect correlation
good  SHAP comprehensiveness (top-k vs random)
bad   LIME local fidelity (infidelity, normalized)
good  Max sensitivity
...   method and subgroup diagnostics shown separately
```

The overall verdict uses scored checks only. The full report also records metric
roles, unavailable or inapplicable checks, decision-policy defaults, explainer
context, and a per-feature comparison table.

## What it measures — and what it does not

| Family | Metrics | Role in the report | Boundary |
|---|---|---|---|
| Faithfulness | SHAP removal-effect correlation and top-k-vs-random comprehensiveness; normalized LIME infidelity | scored | Match contribution metrics to SHAP and gradient metrics to LIME. Comprehensiveness `> 1` is only a not-noise gate. |
| Robustness | max-sensitivity | scored | Lower is better within the recorded perturbation neighbourhood. |
| Reproducibility | run-to-run top-k rank and sign stability | scored | Applies to repeated stochastic explanations. |
| Method comparison | SHAP–LIME sign, rank, top-k, and magnitude differences | descriptive | Agreement is not evidence that either method is correct. |
| Subgroup comparison | cross-segment rank stability and top-k flip rate | descriptive | Heterogeneity may reflect real model behaviour; it is not automatically a failure. |

`DEFAULT_THRESHOLDS` contains six versioned decision-policy defaults, not
universally calibrated cut-offs. Unsupported thresholds for descriptive metrics
are rejected explicitly.

## What the refreshed benchmarks show

- In controlled synthetic regimes, exactly three scored metrics met the held-out
  criterion of at least 0.80 good-pass and 0.75 stress-flag rates: SHAP
  removal-effect correlation, LIME infidelity, and max-sensitivity. The other
  diagnostics did not earn new quality thresholds.
- Across 48 Adult/Diabetes run aggregates, the pooled top-k stability median was
  `0.875`. Restricting the same refreshed raw runs to the first explained
  instance restores `1.000`; evaluating four sampled instances exposed
  variability hidden by the earlier first-instance summary.
- Comprehensiveness is heavy-tailed: its pooled median is `20.67`, while P90 is
  `8.112e+09`. Near-zero random-removal denominators make it useful as a
  not-noise gate, not as a graded effect size or a way to rank datasets.

These are Adult/Diabetes tabular results, not universal validation. Several
protocol corrections changed together, and the historical baseline has no raw
runs, so the refresh cannot isolate individual causes or prove that explanation
quality improved. See the [synthetic study](experiments/README.md),
[real-data benchmark](experiments/benchmark_README.md), and
[change report](experiments/results/real/change_report.md) for the complete
protocol, tables, negative findings, and one explicitly uncomputed subgroup run.

---

## Why the details matter

A SHAP plot alone leaves several technical choices unresolved:

1. **Output space.** SHAP's native output depends on the model and explainer:
   sklearn random forests explain probability, while gradient boosting commonly
   explains a raw margin. The library detects that space and makes LIME and the
   perturbation metrics use the same scalar output.
2. **Contribution vs gradient.** SHAP values are *contributions*
   (`Σ φ_i ≈ f(x) − E[f]`); LIME weights are *slopes* (`f(x̃) ≈ f(x) + φ·Δx`).
   Feeding SHAP values into the standard *infidelity* formula is a category
   error — infidelity is for gradient explanations, ablation metrics are for
   SHAP. We keep the two families separate and label each metric.
3. **Cross-explainer comparison needs compatible units.** LIME's standardized
   coefficients are converted back to original feature units, then
   `to_contribution_scale` produces a baseline-relative local approximation for
   diagnostic SHAP-vs-LIME comparison. It is not an identity between the two
   explanation methods.
4. **Sign stability only over features that matter.** Averaging sign flips over
   all features lets near-zero noise weights dominate the number.

These choices keep the report's assumptions inspectable instead of hiding them
behind a single reliability score.

---

## Interactive app and library API

### Interactive app

Use the sidebar language switch for Chinese or English. Classification training
shows accuracy, balanced accuracy, precision, recall, F1, ROC AUC, a confusion
matrix and ROC curve; regression shows R², MAE and RMSE. Model-specific controls
include an MLP with configurable hidden layers and training-only scaling.
Quality analysis exposes top-k, sensitivity coverage/radius/perturbations, LIME
repeats, infidelity perturbations and subgroup settings. Larger budgets, especially
KernelSHAP for MLP, take longer. Raw data and exported machine keys retain their
original text.

The app now follows five explicit button-driven steps: data selection, splitting,
model training, prediction explanation, and explanation-quality analysis. Choose
from two synthetic presets, UCI Adult/Diabetes, or CSV upload. Feature meanings
are shown alongside values; uploaded fields can have user-provided descriptions.
SHAP and LIME can be selected individually or together. Omitted checks remain
unavailable, so a single-method run does not silently pass the full battery.
Re-executing a stage invalidates its downstream results. Install optional
XGBoost support with `pip install -e ".[app,xgboost]"`.


```bash
streamlit run app/streamlit_app.py
```

Then open the printed URL (default http://localhost:8501).

### Library quickstart

```python
from explaintrust import (
    lime_attributions,
    prediction_output_space,
    scalar_predictor,
    shap_attributions,
    to_contribution_scale,
)

# model is fitted; X_eval and X_background are numeric 2D arrays.
output_space = prediction_output_space(model)
predict = scalar_predictor(model, output_space=output_space)
shap_values = shap_attributions(
    model, X_eval, X_background=X_background, method="auto"
)
lime_slopes = lime_attributions(
    model, X_eval, X_background, output_space=output_space
)
lime_contributions = to_contribution_scale(
    lime_slopes, X_eval, X_background
)
```

See [`examples/demo.py`](https://github.com/lvcheer/explaintrust/blob/main/examples/demo.py)
for the complete metric and report
pipeline. Public imports are listed in `explaintrust.__all__`; compatibility is
maintained according to semantic versioning during the alpha phase.

When `X_background` is supplied, TreeSHAP now uses **interventional** mode and
LinearSHAP uses every supplied row, with no implicit subsampling. Select a smaller
reference explicitly before passing the same rows to SHAP and LIME if needed.
Tree calls without a background retain the training-path mode. Request
`shap_attributions(..., return_context=True)` for `(values, context)`, including
the selected-class SHAP base value, mode, and background digest. The app exports
this context. Shared reference rows do not make nonlinear LIME contributions an
exact SHAP decomposition. Reviewed synthetic calibration and real-data benchmark
outputs are stored under `experiments/results/`; their public tables are generated
from those canonical summaries, while the top-level JSON files remain historical
baselines.

### Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```

The same suite also runs in GitHub Actions on Python 3.9 and 3.12.

For development and release instructions, see
[CONTRIBUTING.md](https://github.com/lvcheer/explaintrust/blob/main/CONTRIBUTING.md),
[SECURITY.md](https://github.com/lvcheer/explaintrust/blob/main/SECURITY.md),
[CHANGELOG.md](https://github.com/lvcheer/explaintrust/blob/main/CHANGELOG.md), and
[RELEASING.md](https://github.com/lvcheer/explaintrust/blob/main/RELEASING.md).

---

## Repository layout

```
explaintrust/
  explainers.py        # SHAP + LIME -> normalized (n, d) attribution matrix
  data.py              # synthetic datasets (collinearity + input-shift helper)
  metrics/
    faithfulness.py    # infidelity, removal corr, comprehensiveness
    sensitivity.py     # max-sensitivity
    stability.py       # cross-run stability
    disagreement.py    # cross-explainer disagreement
    distribution.py    # cross-segment subgroup consistency
  report.py            # trust report: scorecard + verdict + per-feature reliability
app/streamlit_app.py   # interactive demo
examples/demo.py       # headless reference pipeline
tests/test_metrics.py  # correctness/property tests
article/               # Quarto project page and interactive guide
docs/                  # technical brief source and PDF generator
output/pdf/            # distributable two-page technical brief
```

## Reference definitions

- Infidelity & sensitivity — Yeh et al., *On the (In)fidelity and Sensitivity of
  Explanations*, NeurIPS 2019.
- Comprehensiveness inspiration — DeYoung et al., *ERASER*, ACL 2020; the
  top-k-vs-random ratio here is a project-specific tabular adaptation.
- The disagreement problem — Krishna et al., *The Disagreement Problem in
  Explainable Machine Learning*, CACM 2024.

## Status

Current release (v0.2.0). Numeric tabular data, binary classification/regression, SHAP,
and LIME only; image/text, LLM interpretability, and counterfactuals are out of
scope. Thresholds are documented defaults rather than calibrated claims and can
be overridden through `build_trust_report(..., thresholds={...})`. A passing
report means that no configured check failed; it is not a certificate of truth.

---

## 中文说明

**explaintrust 用扰动、重复运行和子群变化来检验表格数据上的 SHAP/LIME 解释是否忠实、稳定且可复现。**

忠实性、敏感度和跨运行稳定性参与评分；解释器之间及不同子群之间的差异单独作为描述性证据，
不会被悄悄转换成质量标签。“通过”只表示当前配置没有检出失败，不代表解释真实或具有因果意义。

**[打开在线演示](https://lvcheer-explaintrust-appstreamlit-app-x7k48h.streamlit.app/)**：无需安装，可直接体验支持中英文的五步分析流程。

### 60 秒运行

在公开 PyPI 安装源完成验证前，请直接安装经过审查的仓库版本：

```bash
git clone https://github.com/lvcheer/explaintrust.git
cd explaintrust
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/demo.py
```

这个固定随机种子的 Demo 会训练模型、生成 SHAP/LIME 解释、计算全部指标并打印报告。
如需本地交互界面，请运行 `python -m pip install -e ".[app]"`，然后执行
`streamlit run app/streamlit_app.py`。

### 报告示例

无界面 Demo 的精简输出如下：

```text
TRUST VERDICT: MIXED — investigate before trusting
good  SHAP removal-effect correlation
good  SHAP comprehensiveness (top-k vs random)
bad   LIME local fidelity (infidelity, normalized)
good  Max sensitivity
...   方法间与子群诊断单独展示
```

总体结论只汇总评分项。完整报告还会记录指标角色、缺失或不适用的检查、判定策略、解释器上下文，
以及逐特征比较表。

### 测量内容与边界

| 类别 | 指标 | 报告角色 | 边界 |
|---|---|---|---|
| 忠实性 | SHAP 移除效应相关、top-k 与随机移除的完备性比；归一化 LIME infidelity | 评分 | 贡献类指标用于 SHAP，梯度类指标用于 LIME；完备性比 `> 1` 只是一道“非噪声”门槛。 |
| 鲁棒性 | 最大敏感度（max-sensitivity） | 评分 | 在记录的扰动邻域内越低越好。 |
| 可复现性 | 多次运行的 top-k 排名与符号稳定性 | 评分 | 适用于带随机性的重复解释。 |
| 方法比较 | SHAP–LIME 的符号、排名、top-k 与幅度差异 | 描述 | 一致不代表任一方法正确。 |
| 子群比较 | 跨分段排名稳定性与 top-k 翻转率 | 描述 | 异质性可能来自真实的模型行为，不会自动判为失败。 |

`DEFAULT_THRESHOLDS` 中的六组边界是版本化判定策略，不是普适校准阈值。系统会明确拒绝为描述性指标设置质量阈值。

### 刷新后的 benchmark 说明了什么

- 在受控合成实验中，恰有三项评分指标达到留出集标准（正常条件通过率至少 0.80，压力条件
  标记率至少 0.75）：SHAP 移除效应相关、LIME infidelity 和最大敏感度。其他诊断量没有因此
  获得新的质量阈值。
- Adult/Diabetes 的 48 个运行汇总中，top-k 稳定性的合并中位数为 `0.875`。若把同一批刷新后
  的原始结果限制为每次运行的第一个解释样本，中位数会恢复到 `1.000`；四个抽样实例的覆盖
  暴露了旧汇总没有显示的波动。
- 完备性比分布重尾：合并中位数为 `20.67`，P90 为 `8.112e+09`。随机移除效应接近零时，
  比值会急剧放大，因此它适合作为“非噪声”门槛，不适合作为连续效应量或数据集排名依据。

这些结果只覆盖 Adult/Diabetes 表格数据，并非普适验证。多项协议修正同时发生，而历史基线没有
原始运行记录，因此无法分别估计每项修正的影响，也不能据此声称解释质量提高。完整协议、表格、
负面发现及一项明确未计算的 Adult 子群结果见[合成实验](experiments/README.md)、
[真实数据 benchmark](experiments/benchmark_README.md)和
[变更报告](experiments/results/real/change_report.md)。

### 为什么这些细节重要

一张 SHAP 图不会自动解决以下技术问题：

1. **输出空间**。SHAP 的原生输出空间取决于模型和解释器：例如 sklearn Random Forest 通常解释概率，而 Gradient Boosting 通常解释 raw margin。本库会检测该空间，并让 LIME 与扰动指标使用相同的标量输出。
2. **贡献 vs 梯度**。SHAP 值是 *贡献*（`Σ φ_i ≈ f(x) − E[f]`）；LIME 权重是 *斜率*（`f(x̃) ≈ f(x) + φ·Δx`）。把 SHAP 值喂给标准的 infidelity 公式是一个范畴错误——infidelity 适用于梯度解释，消融类指标适用于 SHAP。我们把这两个家族分开，并给每个指标标注适用对象。
3. **跨解释器比较需要兼容单位**。本库先把 LIME 的标准化坐标系数还原到原始特征单位，再用 `to_contribution_scale` 构造相对背景的局部近似。这是诊断性比较，并不意味着 LIME 与 SHAP 在理论上完全等价。
4. **符号稳定性只对“有分量的特征”计算**。对所有特征平均符号翻转，会让接近零的噪声权重主导这个数字。

这些处理把报告所依赖的假设留在明面上，而不是压缩成一个看似确定的“可靠性分数”。

### 交互应用与库 API

#### 交互式应用

页面按按钮逐步执行：**选择数据 → 数据划分 → 模型训练 → 解释预测 → 解释质量分析**。
提供两种模拟数据、Adult／Diabetes 真实预设，以及 CSV 上传。模拟变量展示生成角色，
真实字段展示含义；上传数据可填写业务说明。上传流程目前保留数值特征并排除缺失行，
支持二分类和回归。

侧边栏可统一切换中文或英文。分类训练展示准确率、平衡准确率、精确率、召回率、F1、ROC AUC、混淆矩阵和 ROC 曲线；回归展示 R²、MAE、RMSE。质量分析可配置 top-k、敏感度样本数与扰动、LIME 重复次数、局部拟合扰动和子群设置。原始数据及导出机器字段保留原文。

模型可选多层感知机 MLP（可配置隐藏层，标准化仅在训练集拟合）、随机森林、梯度提升、线性模型及可选的 XGBoost（安装 `.[app,xgboost]`）。
解释方法可选 SHAP、LIME 或两者；质量分析复用已生成的解释，未选择的方法不会偷偷运行。
重新执行上游步骤会清除下游结果。字段说明、划分协议和选中样本位置会写入报告。


```bash
streamlit run app/streamlit_app.py
```

然后打开打印出来的网址（默认 http://localhost:8501）。

#### Python 库快速示例

```python
from explaintrust import prediction_output_space, scalar_predictor, shap_attributions

output_space = prediction_output_space(model)
predict = scalar_predictor(model, output_space=output_space)
shap_values = shap_attributions(
    model, X_eval, X_background=X_background, method="auto"
)
```

完整指标与报告流程见
[`examples/demo.py`](https://github.com/lvcheer/explaintrust/blob/main/examples/demo.py)。公开 API 以
`explaintrust.__all__` 为准；alpha 阶段按语义化版本规则管理兼容性。

传入 `X_background` 时，TreeSHAP 采用 **interventional** 模式，LinearSHAP 使用全部
背景行，不再隐式抽样。如需控制成本，请先显式选取较小背景，再将相同行传给 SHAP 与 LIME。
树模型不传背景时保留原有路径模式。设置 `return_context=True` 可取得 `(values, context)`，
包括对应类别的 SHAP 基值、模式和背景指纹；应用会导出这些信息。共同背景并不使非线性
LIME 贡献成为精确的 SHAP 分解。经过审查的合成校准与真实数据基准结果保存在
`experiments/results/`；公开表格由这些规范化摘要生成，顶层 JSON 文件则保留为历史基线。

#### 测试

```bash
pip install -e ".[test]"
python -m pytest -q
```

同一套测试也会在 GitHub Actions 的 Python 3.9 与 3.12 环境中运行。

开发、安全、变更与发布流程分别见
[CONTRIBUTING.md](https://github.com/lvcheer/explaintrust/blob/main/CONTRIBUTING.md)、
[SECURITY.md](https://github.com/lvcheer/explaintrust/blob/main/SECURITY.md)、
[CHANGELOG.md](https://github.com/lvcheer/explaintrust/blob/main/CHANGELOG.md) 和
[RELEASING.md](https://github.com/lvcheer/explaintrust/blob/main/RELEASING.md)。

### 仓库结构

```
explaintrust/
  explainers.py        # SHAP + LIME → 归一化的 (n, d) 归因矩阵
  data.py              # 合成数据集（共线性 + 输入漂移辅助函数）
  metrics/
    faithfulness.py    # infidelity、removal corr、comprehensiveness
    sensitivity.py     # max-sensitivity
    stability.py       # 跨运行稳定性
    disagreement.py    # 跨解释器分歧
    distribution.py    # 跨子群一致性检查
  report.py            # 信任报告：打分卡 + 结论 + 逐特征可靠性
app/streamlit_app.py   # 交互式 Demo
examples/demo.py       # 无界面参考流程
tests/test_metrics.py  # 正确性/性质测试
article/               # Quarto 项目主页与交互式指南
docs/                  # 技术简报源文件与 PDF 生成器
output/pdf/            # 可分发的两页技术简报
```

### 参考定义

- Infidelity / sensitivity — Yeh 等，*On the (In)fidelity and Sensitivity of Explanations*，NeurIPS 2019。
- Comprehensiveness 的方法启发 — DeYoung 等，*ERASER*，ACL 2020；本项目的
  top-k 与随机移除之比是面向表格数据的自定义改造。
- 分歧问题 — Krishna 等，*The Disagreement Problem in Explainable Machine Learning*，CACM 2024。

### 现状

当前版本（v0.2.0）。目前仅支持数值型表格数据、二分类/回归、SHAP 与 LIME；图像/文本、LLM 可解释性和反事实解释暂不在范围内。报告阈值是文档化默认值，不是经过普适校准的结论，可通过 `build_trust_report(..., thresholds={...})` 覆盖。所有检查通过只表示“当前配置未检出问题”，并非真实性证书。
