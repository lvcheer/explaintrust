# explaintrust

中文说明见[下方](#中文说明) · A Chinese version of this README is [below](#中文说明).

> Post-hoc explanations are easy to produce and easy to over-trust. **explaintrust** asks the question most XAI tooling ignores: *"SHAP/LIME gave me a feature attribution — but can I trust it?"*

It evaluates explanations the way a careful reviewer would — not by how pretty they look, but by whether they are **faithful, stable, mutually consistent, and consistent across selected subgroups**.

Built as a **brand / research artifact**: the kernel is a documented, citable Python library. Published metrics retain their stated definitions; project-specific diagnostics are labeled as adaptations rather than universal tests.

---

## What it measures

| Family | Metric | Explainer it's valid for | Direction |
|---|---|---|---|
| Faithfulness | removal-effect correlation | SHAP (contribution) | higher |
| Faithfulness | comprehensiveness ratio (top-k vs random) | SHAP (contribution) | higher (> 1 = not noise) |
| Faithfulness | infidelity (normalized local linear surrogate) | LIME (gradient) | lower |
| Robustness | max-sensitivity | any | lower |
| Reproducibility | run-to-run rank / sign / top-k stability | stochastic explainers | higher |
| Consistency | SHAP vs LIME sign/rank/top-k disagreement | cross-explainer | — |
| Consistency | SHAP vs LIME magnitude disagreement (per-feature gap) | cross-explainer | lower |
| Subgroup consistency | cross-segment rank stability & top-k flip rate | any | higher / lower |

**The output is a "trust report"**: a scorecard of every metric with a verdict
(good / warn / bad) and a plain-English reason, plus an overall verdict and a
per-feature reliability table.

---

## Why the details matter (the point of this project)

A naive "run SHAP and show a plot" tool gets several things subtly wrong. This
library is opinionated about them on purpose:

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

These are exactly the things a reviewer (or a downstream user) would catch —
and the reason a PhD in explainable ML has an advantage a generic vibe-coder
does not.

---

## Install & run

After the first PyPI release, install the library with:

```bash
pip install explaintrust
```

To run the interactive app or contribute, clone the repository and install it
in editable mode:

```bash
git clone https://github.com/lvcheer/explaintrust.git
cd explaintrust
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[app]"          # ".[app]" also pulls streamlit + plotly
```

### Headless demo (fastest way to see the report)

```bash
python3 examples/demo.py
```

### Interactive app

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
article/               # Quarto explorable article ("Why your SHAP plot might be lying to you")
```

## Reference definitions

- Infidelity & sensitivity — Yeh et al., *On the (In)fidelity and Sensitivity of
  Explanations*, NeurIPS 2019.
- Comprehensiveness inspiration — DeYoung et al., *ERASER*, ACL 2020; the
  top-k-vs-random ratio here is a project-specific tabular adaptation.
- The disagreement problem — Krishna et al., *The Disagreement Problem in
  Explainable Machine Learning*, CACM 2024.

## Status

Prototype (v0.1.0). Numeric tabular data, binary classification/regression, SHAP,
and LIME only; image/text, LLM interpretability, and counterfactuals are out of
scope. Thresholds are documented defaults rather than calibrated claims and can
be overridden through `build_trust_report(..., thresholds={...})`. A passing
report means that no configured check failed; it is not a certificate of truth.

---

## 中文说明

> 事后解释（post-hoc explanation）很容易生成，也很容易被过度信任。**explaintrust** 追问的是大多数 XAI 工具忽略的问题：*“SHAP/LIME 给了我一组特征归因——但我能相信它吗？”*

本项目不看图表有多漂亮，而看它是否**忠实（faithful）、稳定（stable）、相互一致（mutually consistent），并在选定子群体之间保持一致**。

本项目内核是一个有文档、可引用的 Python 库。已有论文定义的指标保持其定义；项目自定义或改造的诊断量会明确标注，不将其包装成普适检验。外壳是一个轻量的交互式 Demo。

### 它测量什么

| 类别 | 指标 | 适用的解释器 | 方向 |
|---|---|---|---|
| 忠实性 | 移除效应相关（removal-effect correlation） | SHAP（贡献） | 越高越好 |
| 忠实性 | 完备性比（comprehensiveness ratio，top-k vs 随机） | SHAP（贡献） | 越高越好（> 1 说明非噪声） |
| 忠实性 | 不忠实度（infidelity，局部线性代理） | LIME（梯度） | 越低越好 |
| 鲁棒性 | 最大敏感度（max-sensitivity） | 任意 | 越低越好 |
| 可复现性 | 多次运行的 rank / sign / top-k 稳定性 | 随机性解释器 | 越高越好 |
| 一致性 | SHAP 与 LIME 的 sign/rank/top-k 分歧 | 跨解释器 | — |
| 子群一致性 | 跨分段的 rank 稳定性与 top-k 翻转率 | 任意 | 越高 / 越低 |

**输出是一份“信任报告”**：每个指标的打分卡 + 结论（good / warn / bad）+ 用大白话写的原因，外加一个总体结论和一张逐特征可靠性表。

### 为什么这些细节重要（本项目的核心）

一个“跑一下 SHAP 然后画张图”的工具会在几处地方微妙地出错。本库在如下地方展开讨论：

1. **输出空间**。SHAP 的原生输出空间取决于模型和解释器：例如 sklearn Random Forest 通常解释概率，而 Gradient Boosting 通常解释 raw margin。本库会检测该空间，并让 LIME 与扰动指标使用相同的标量输出。
2. **贡献 vs 梯度**。SHAP 值是 *贡献*（`Σ φ_i ≈ f(x) − E[f]`）；LIME 权重是 *斜率*（`f(x̃) ≈ f(x) + φ·Δx`）。把 SHAP 值喂给标准的 infidelity 公式是一个范畴错误——infidelity 适用于梯度解释，消融类指标适用于 SHAP。我们把这两个家族分开，并给每个指标标注适用对象。
3. **跨解释器比较需要兼容单位**。本库先把 LIME 的标准化坐标系数还原到原始特征单位，再用 `to_contribution_scale` 构造相对背景的局部近似。这是诊断性比较，并不意味着 LIME 与 SHAP 在理论上完全等价。
4. **符号稳定性只对“有分量的特征”计算**。对所有特征平均符号翻转，会让接近零的噪声权重主导这个数字。

这些正是审稿人（或下游用户）会发现的点——也是“解释性机器学习方向的博士”相对泛泛的开发者所具备的优势所在。

### 安装与运行

首次发布到 PyPI 后，可直接安装 Python 库：

```bash
pip install explaintrust
```

如需运行交互应用或参与开发，请克隆仓库并以可编辑模式安装：

```bash
git clone https://github.com/lvcheer/explaintrust.git
cd explaintrust
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[app]"          # ".[app]" 会额外拉取 streamlit + plotly
```

#### 无界面 Demo（最快看到报告）

```bash
python3 examples/demo.py
```

#### 交互式应用

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
article/               # Quarto 交互式文章（“Why your SHAP plot might be lying to you”）
```

### 参考定义

- Infidelity / sensitivity — Yeh 等，*On the (In)fidelity and Sensitivity of Explanations*，NeurIPS 2019。
- Comprehensiveness 的方法启发 — DeYoung 等，*ERASER*，ACL 2020；本项目的
  top-k 与随机移除之比是面向表格数据的自定义改造。
- 分歧问题 — Krishna 等，*The Disagreement Problem in Explainable Machine Learning*，CACM 2024。

### 现状

原型（v0.1.0）。目前仅支持数值型表格数据、二分类/回归、SHAP 与 LIME；图像/文本、LLM 可解释性和反事实解释暂不在范围内。报告阈值是文档化默认值，不是经过普适校准的结论，可通过 `build_trust_report(..., thresholds={...})` 覆盖。所有检查通过只表示“当前配置未检出问题”，并非真实性证书。
