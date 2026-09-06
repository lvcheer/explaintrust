"""Read-only review probes; run from repository root with the project Python.

These record current behavior rather than asserting that defects are desirable.
"""
import contextlib
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import shap
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from explaintrust import build_trust_report, scalar_predictor, shap_attributions
from explaintrust.metrics import comprehensiveness_ratio, max_sensitivity


class Nonlinear:
    def predict(self, X):
        return (np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2
                + X[:, 3] * X[:, 4] + np.cos(X[:, 5] + X[:, 6]))


def main():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    model = LogisticRegression().fit(X, (X[:, 0] + X[:, 1] > 0).astype(int))
    class_zero = scalar_predictor(model, class_index=0)(X[:3])
    class_one = scalar_predictor(model, class_index=1)(X[:3])
    # Change unrelated global RNG state while holding the API seed fixed.
    saved_state = np.random.get_state()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            np.random.seed(100)
            a = shap_attributions(Nonlinear(), X[:1], X[:15], method="kernel", nsamples=30, seed=7)
            np.random.seed(200)
            b = shap_attributions(Nonlinear(), X[:1], X[:15], method="kernel", nsamples=30, seed=7)
    finally:
        np.random.set_state(saved_state)

    x, bg = np.array([1., 2.]), np.zeros((3, 2))
    ratio = comprehensiveness_ratio(lambda Z: Z[:, 0] + Z[:, 1], x, x, bg, top_k=2)
    sensitivity = max_sensitivity(lambda z: np.full(2, np.nan), x, bg, n_perturbations=3)
    report = build_trust_report(
        .9, 2., float("inf"), 0.,
        {"topk_rank_corr": 1., "sign_agreement": 1.},
        {"sign_disagreement": 0., "topk_rank_corr": 1., "topk_overlap": 1., "magnitude_disagreement": 0.},
    )
    X_linear = np.random.default_rng(3).normal(size=(500, 3))
    linear = LinearRegression().fit(X_linear, X_linear @ np.array([1., 2., 3.]))
    linear_shap = shap_attributions(linear, X_linear[:1], X_linear, method="linear")
    expected = linear.coef_ * (X_linear[:1] - X_linear.mean(axis=0))
    print(json.dumps({
        "python": platform.python_version(), "shap": shap.__version__,
        "binary_class_zero_equals_class_one": bool(np.allclose(class_zero, class_one)),
        "kernel_same_api_seed_global_rng_change_max_gap": float(np.max(np.abs(a - b))),
        "comprehensiveness_when_k_equals_d": ratio,
        "nan_attributions_sensitivity": sensitivity,
        "infinite_infidelity_verdict": report.metric_results[2].verdict,
        "linear_shap_vs_full_background_formula_max_gap": float(np.max(np.abs(linear_shap - expected))),
    }, indent=2))


if __name__ == "__main__":
    main()
