# explaintrust

中文说明见[下方](#中文说明) · A Chinese version of this README is [below](#中文说明).

> Post-hoc explanations are easy to produce and easy to over-trust. **explaintrust** asks the question most XAI tooling ignores: *"SHAP/LIME gave me a feature attribution — but can I trust it?"*

It evaluates explanations the way a careful reviewer would — not by how pretty they look, but by whether they are **faithful, stable, mutually consistent, and robust across the data distribution**.

Built as a **brand / research artifact**: the kernel is a clean, documented, citable Python library (every metric is implemented from published definitions), and the shell is a thin interactive demo.

---

## What it measures

| Family | Metric | Explainer it's valid for | Direction |
|---|---|---|---|
| Faithfulness | removal-effect correlation | SHAP (contribution) | higher |
| Faithfulness | comprehensiveness ratio (top-k vs random) | SHAP (contribution) | higher (> 1 = not noise) |
| Faithfulness | infidelity (local linear surrogate) | LIME (gradient) | lower |
| Robustness | max-sensitivity | any | lower |
| Reproducibility | run-to-run rank / sign / top-k stability | stochastic explainers | higher |
| Consistency | SHAP vs LIME sign/rank/top-k disagreement | cross-explainer | — |
| Generalization | cross-segment rank stability & top-k flip rate | any | higher / lower |

**The output is a "trust report"**: a scorecard of every metric with a verdict
(good / warn / bad) and a plain-English reason, plus an overall verdict and a
per-feature reliability table.

---

## Why the details matter (the point of this project)

A naive "run SHAP and show a plot" tool gets several things subtly wrong. This
library is opinionated about them on purpose:

1. **Output space.** For classifiers, SHAP values live in *log-odds*, LIME
   weights in *probability*. Comparing them directly is meaningless. We pin
   everything to a single space (log-odds for classifiers, raw output for
   regressors).
2. **Contribution vs gradient.** SHAP values are *contributions*
   (`Σ φ_i ≈ f(x) − E[f]`); LIME weights are *slopes* (`f(x̃) ≈ f(x) + φ·Δx`).
   Feeding SHAP values into the standard *infidelity* formula is a category
   error — infidelity is for gradient explanations, ablation metrics are for
   SHAP. We keep the two families separate and label each metric.
3. **Cross-explainer comparison needs a common scale.** `to_contribution_scale`
   converts LIME weights to SHAP-comparable units before any SHAP-vs-LIME
   disagreement is computed.
4. **Sign stability only over features that matter.** Averaging sign flips over
   all features lets near-zero noise weights dominate the number.

These are exactly the things a reviewer (or a downstream user) would catch —
and the reason a PhD in explainable ML has an advantage a generic vibe-coder
does not.

---

## Install & run

Create a virtual environment and install the package in editable mode
(installs dependencies + makes `import explaintrust` work from anywhere):

```bash
python3 -m venv .venv && source .venv/bin/activate
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

### Tests

```bash
python3 tests/test_metrics.py
```

---

## Repository layout

```
explaintrust/
  explainers.py        # SHAP + LIME -> normalized (n, d) attribution matrix
  data.py              # synthetic datasets (collinearity + distribution shift)
  metrics/
    faithfulness.py    # infidelity, removal corr, comprehensiveness
    sensitivity.py     # max-sensitivity
    stability.py       # cross-run stability
    disagreement.py    # cross-explainer disagreement
    distribution.py    # cross-segment / distribution verification
  report.py            # trust report: scorecard + verdict + per-feature reliability
app/streamlit_app.py   # interactive demo
examples/demo.py       # headless reference pipeline
tests/test_metrics.py  # correctness/property tests
article/               # Quarto explorable article ("Why your SHAP plot might be lying to you")
```

## Reference definitions

- Infidelity & sensitivity — Yeh et al., *On the (In)fidelity and Sensitivity of
  Explanations*, NeurIPS 2019.
- Comprehensiveness / sufficiency — DeYoung et al., *ERASER*, ACL 2020.
- The disagreement problem — Krishna et al., *The Disagreement Problem in
  Explainable Machine Learning*, CACM 2024.

## Status

Prototype (v0.1.0). Tabular data + SHAP + LIME only; image/text, LLM
interpretability, and counterfactuals are future work. Thresholds in the report
are sensible defaults, not calibrated claims — they are labeled as such and are
meant to be overridden with domain knowledge.

---

## 中文说明

> 事后解释（post-hoc explanation）很容易生成，也很容易被过度信任。**explaintrust** 追问的是大多数 XAI 工具忽略的问题：*“SHAP/LIME 给了我一组特征归因——但我能相信它吗？”*

本项目不看图表有多漂亮，而看它是否**忠实（faithful）、稳定（stable）、自洽（mutually consistent）、并在数据分布上鲁棒（robust across the distribution）**。

本项目内核是一个干净、有文档、可引用的 Python 库（每个指标都按已发表的论文定义实现），外壳是一个轻量的交互式 Demo。

### 它测量什么

| 类别 | 指标 | 适用的解释器 | 方向 |
|---|---|---|---|
| 忠实性 | 移除效应相关（removal-effect correlation） | SHAP（贡献） | 越高越好 |
| 忠实性 | 完备性比（comprehensiveness ratio，top-k vs 随机） | SHAP（贡献） | 越高越好（> 1 说明非噪声） |
| 忠实性 | 不忠实度（infidelity，局部线性代理） | LIME（梯度） | 越低越好 |
| 鲁棒性 | 最大敏感度（max-sensitivity） | 任意 | 越低越好 |
| 可复现性 | 多次运行的 rank / sign / top-k 稳定性 | 随机性解释器 | 越高越好 |
| 一致性 | SHAP 与 LIME 的 sign/rank/top-k 分歧 | 跨解释器 | — |
| 泛化性 | 跨分段的 rank 稳定性与 top-k 翻转率 | 任意 | 越高 / 越低 |

**输出是一份“信任报告”**：每个指标的打分卡 + 结论（good / warn / bad）+ 用大白话写的原因，外加一个总体结论和一张逐特征可靠性表。

### 为什么这些细节重要（本项目的核心）

一个“跑一下 SHAP 然后画张图”的工具会在几处地方微妙地出错。本库在如下地方展开讨论：

1. **输出空间**。对分类器而言，SHAP 值在 *log-odds（对数几率）* 空间，LIME 权重在 *概率* 空间；直接比较二者没有意义。我们把一切统一到同一空间（分类器用 log-odds，回归器用原始输出）。
2. **贡献 vs 梯度**。SHAP 值是 *贡献*（`Σ φ_i ≈ f(x) − E[f]`）；LIME 权重是 *斜率*（`f(x̃) ≈ f(x) + φ·Δx`）。把 SHAP 值喂给标准的 infidelity 公式是一个范畴错误——infidelity 适用于梯度解释，消融类指标适用于 SHAP。我们把这两个家族分开，并给每个指标标注适用对象。
3. **跨解释器比较需要统一尺度**。`to_contribution_scale` 会在计算任何 SHAP-vs-LIME 分歧之前，把 LIME 权重换算成与 SHAP 可比的单位。
4. **符号稳定性只对“有分量的特征”计算**。对所有特征平均符号翻转，会让接近零的噪声权重主导这个数字。

这些正是审稿人（或下游用户）会发现的点——也是“解释性机器学习方向的博士”相对泛泛的开发者所具备的优势所在。

### 安装与运行

创建虚拟环境并以可编辑模式安装（会直接装好依赖，并让 `import explaintrust` 在任何目录可用）：

```bash
python3 -m venv .venv && source .venv/bin/activate
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

#### 测试

```bash
python3 tests/test_metrics.py
```

### 仓库结构

```
explaintrust/
  explainers.py        # SHAP + LIME → 归一化的 (n, d) 归因矩阵
  data.py              # 合成数据集（共线性 + 分布漂移）
  metrics/
    faithfulness.py    # infidelity、removal corr、comprehensiveness
    sensitivity.py     # max-sensitivity
    stability.py       # 跨运行稳定性
    disagreement.py    # 跨解释器分歧
    distribution.py    # 跨分段 / 分布校验
  report.py            # 信任报告：打分卡 + 结论 + 逐特征可靠性
app/streamlit_app.py   # 交互式 Demo
examples/demo.py       # 无界面参考流程
tests/test_metrics.py  # 正确性/性质测试
article/               # Quarto 交互式文章（“Why your SHAP plot might be lying to you”）
```

### 参考定义

- Infidelity / sensitivity — Yeh 等，*On the (In)fidelity and Sensitivity of Explanations*，NeurIPS 2019。
- Comprehensiveness / sufficiency — DeYoung 等，*ERASER*，ACL 2020。
- 分歧问题 — Krishna 等，*The Disagreement Problem in Explainable Machine Learning*，CACM 2024。

### 现状

原型（v0.1.0）。目前仅支持表格数据 + SHAP + LIME；图像/文本、LLM 可解释性、反事实解释属于后续工作。报告中的阈值是“合理的默认值”，并非经过校准的结论——代码中已如实标注，可用领域知识覆盖。
