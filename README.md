# explaintrust

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
