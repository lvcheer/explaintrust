"""Correctness tests for the explaintrust kernel.

These are property tests, not unit tests for implementation details: each one
asserts a *behavior the metric must satisfy* on a known ground truth. Run with
plain Python (no pytest required):

    python3 tests/test_metrics.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from explaintrust import (
    lime_attributions,
    shap_attributions,
    to_contribution_scale,
    scalar_predictor,
    build_trust_report,
)
from explaintrust.explainers import _shap_to_matrix
from explaintrust.metrics import (
    infidelity,
    removal_effect_correlation,
    comprehensiveness_ratio,
    max_sensitivity,
    cross_run_stability,
    explainer_disagreement,
    cross_segment_stability,
)


def _linear_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.0 * X[:, 2] + 0.0 * X[:, 3]
    return X, y


def test_removal_correlation_high_on_true_importance():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    pred = scalar_predictor(model)
    # True attribution ranking: x0 > x1 >> x2, x3
    attr = np.array([3.0, -2.0, 0.0, 0.0])
    corr = removal_effect_correlation(pred, X[0], attr, X[:100])
    assert corr > 0.7, f"removal corr should be high for true importance, got {corr}"


def test_removal_correlation_low_on_wrong_importance():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    pred = scalar_predictor(model)
    # Attribute importance to the *noise* features instead.
    attr = np.array([0.0, 0.0, 3.0, 2.0])
    corr = removal_effect_correlation(pred, X[0], attr, X[:100])
    assert corr < 0.3, f"removal corr should be low for wrong importance, got {corr}"


def test_infidelity_lower_for_true_gradient_than_wrong():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    pred = scalar_predictor(model)
    true_grad = model.coef_  # the local linear model IS the true gradient
    wrong_grad = np.array([0.0, 0.0, 3.0, 2.0])
    inf_true = infidelity(pred, X[0], true_grad, X[:100], seed=0)
    inf_wrong = infidelity(pred, X[0], wrong_grad, X[:100], seed=0)
    assert inf_true < inf_wrong, f"{inf_true} should be < {inf_wrong}"


def test_comprehensiveness_detects_predictive_feature():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    pred = scalar_predictor(model)
    attr = np.array([3.0, -2.0, 0.0, 0.0])  # points to x0, x1
    ratio = comprehensiveness_ratio(pred, X[0], attr, X[:100], top_k=2, seed=0)
    assert ratio > 1.5, f"comprehensiveness should exceed random, got {ratio}"


def test_top_k_is_capped_for_low_dimensional_inputs():
    X, y = _linear_data()
    X = X[:, :2]
    model = LinearRegression().fit(X, y)
    pred = scalar_predictor(model)
    ratio = comprehensiveness_ratio(pred, X[0], model.coef_, X[:100], top_k=3, seed=0)
    assert np.isfinite(ratio)

    attrs = np.tile(np.array([1.0, 0.5]), (20, 1))
    result = cross_segment_stability(X[:20], np.repeat([0, 1], 10), attrs, top_k=3)
    assert result["topk_flip_rate"] == 0.0


def test_lime_sign_matches_true_coefficients():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    lime = lime_attributions(model, X[:1], X[:100], feature_names=["x0", "x1", "x2", "x3"],
                             num_samples=4000, seed=0)[0]
    assert np.sign(lime[0]) == np.sign(model.coef_[0])
    assert np.sign(lime[1]) == np.sign(model.coef_[1])
    assert abs(lime[2]) < 0.3 and abs(lime[3]) < 0.3, f"noise features should be ~0, got {lime}"


def test_to_contribution_scale_matches_shap_on_linear_model():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    bg = X[:100]
    shap = shap_attributions(model, X[:3], method="kernel", X_background=bg, nsamples=500, seed=0)
    lime = lime_attributions(model, X[:3], bg, num_samples=4000, seed=0)
    lime_contrib = to_contribution_scale(lime, X[:3], bg)
    # On a linear model both should be (x - mean)*coef, so they should agree closely.
    corr = np.corrcoef(shap.ravel(), lime_contrib.ravel())[0, 1]
    assert corr > 0.9, f"converted LIME should match SHAP on linear model, corr={corr}"


def test_lime_gradient_is_in_original_feature_units():
    rng = np.random.default_rng(2)
    X = np.column_stack(
        [
            rng.normal(0, 100, 600),
            rng.normal(0, 0.01, 600),
            rng.normal(0, 5, 600),
        ]
    )
    true_gradient = np.array([0.02, -100.0, 0.5])
    model = LinearRegression().fit(X, X @ true_gradient)
    lime = lime_attributions(model, X[200:201], X[:200], num_samples=5000, seed=0)[0]
    assert np.allclose(lime, true_gradient, rtol=0.02, atol=0.02), (lime, true_gradient)
    score = infidelity(scalar_predictor(model), X[200], lime, X[:200], seed=0)
    assert score < 1e-4, f"exact linear gradient should have near-zero infidelity, got {score}"


def test_tree_shap_and_scalar_predictor_share_output_space():
    X, y = _linear_data()
    yb = (y > 0).astype(int)
    models = [
        RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0),
        GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=0),
    ]
    for model in models:
        model.fit(X, yb)
        attr = shap_attributions(model, X[:3], method="tree")
        output = scalar_predictor(model)(X[:3])
        # SHAP's expected value cancels when comparing any two instances.
        assert np.allclose(
            attr.sum(axis=1) - attr.sum(axis=1)[0],
            output - output[0],
            atol=1e-6,
        ), type(model).__name__


def test_old_shap_class_list_keeps_instance_feature_shape():
    values = [np.arange(12).reshape(3, 4), 100 + np.arange(12).reshape(3, 4)]
    out = _shap_to_matrix(values, class_index=1)
    assert out.shape == (3, 4)
    assert np.array_equal(out, values[1])


def test_deterministic_explainer_is_perfectly_stable():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    def deterministic(seed: int):
        return shap_attributions(model, X[:1], method="kernel", X_background=X[:100],
                                 nsamples=300, seed=0)[0]  # seed ignored -> identical
    res = cross_run_stability(deterministic, n_runs=5, top_k=2)
    assert res["rank_corr"] == 1.0
    assert np.allclose(res["topk_rank_corr"], 1.0)
    assert res["sign_agreement"] == 1.0


def test_identical_attributions_have_zero_disagreement():
    a = np.array([0.5, -0.3, 0.1, 0.0])
    d = explainer_disagreement(a, a, top_k=2)
    assert d["sign_disagreement"] == 0.0
    assert d["rank_corr"] == 1.0
    assert np.allclose(d["topk_rank_corr"], 1.0)
    assert d["topk_overlap"] == 1.0
    assert d["magnitude_disagreement"] == 0.0


def test_rank_agreement_uses_importance_magnitude_not_sign():
    a = np.array([-3.0, 2.0, -1.0])
    b = -a
    d = explainer_disagreement(a, b, top_k=3)
    assert d["rank_corr"] == 1.0
    assert d["topk_rank_corr"] == 1.0
    assert d["sign_disagreement"] == 1.0


def test_topk_rank_corr_robust_to_noise_features():
    # Same top-3 ranking, but the many noise features shuffle their ranks:
    # the full-d rank correlation drops, the top-k one stays perfect. This is
    # the dimension-robustness the real-data benchmark demanded.
    a = np.array([0.8, 0.6, 0.4, 0.03, 0.02, 0.01])
    b = np.array([0.8, 0.6, 0.4, 0.01, 0.03, 0.02])
    d = explainer_disagreement(a, b, top_k=3)
    assert d["topk_rank_corr"] == 1.0
    assert d["topk_rank_corr"] > d["rank_corr"]


def test_magnitude_disagreement_detects_scale_gap():
    # Same ranking and signs, but a large magnitude gap on the top feature:
    # rank/sign/overlap agreement all read "perfect", magnitude disagreement must not.
    a = np.array([0.5, -0.3, 0.1, 0.0])
    b = np.array([2.0, -0.3, 0.1, 0.0])
    d = explainer_disagreement(a, b, top_k=2)
    assert d["rank_corr"] == 1.0
    assert d["sign_disagreement"] == 0.0
    assert d["topk_overlap"] == 1.0
    assert d["magnitude_disagreement"] > 0.0


def test_cross_segment_stability_perfect_when_identical():
    X = np.random.default_rng(0).normal(0, 1, size=(60, 4))
    seg = np.repeat([0, 1, 2], 20)
    attr = np.tile(np.array([0.8, 0.6, 0.1, 0.05]), (60, 1))  # same ranking everywhere
    res = cross_segment_stability(X, seg, attr, top_k=2)
    assert res["rank_corr"] == 1.0
    assert res["topk_flip_rate"] == 0.0


def test_max_sensitivity_zero_for_constant_explainer():
    X, y = _linear_data()
    def const(_x):
        return np.array([1.0, 1.0, 1.0, 1.0])
    sens = max_sensitivity(const, X[0], X[:100], n_perturbations=10, seed=0)
    assert sens == 0.0


def test_max_sensitivity_is_change_not_lipschitz_ratio():
    rng = np.random.default_rng(0)
    background = rng.normal(size=(5000, 2))
    x = np.array([1.0, -1.0])
    sens = max_sensitivity(lambda z: z, x, background, n_perturbations=200, radius=0.1, seed=0)
    assert 0.05 < sens <= 0.11, sens


def test_infidelity_normalization_uses_zero_change_baseline():
    background = np.zeros((100, 2))
    x = np.ones(2)
    model = lambda z: np.asarray(z).sum(axis=1)
    score = infidelity(
        model,
        x,
        np.zeros(2),
        background,
        n_perturbations=100,
        strategy="mask",
        seed=0,
    )
    assert np.allclose(score, 1.0), score


def test_missing_metrics_never_produce_positive_overall_verdict():
    nan = float("nan")
    report = build_trust_report(
        nan,
        nan,
        nan,
        None,
        {"topk_rank_corr": nan, "sign_agreement": nan},
        {
            "sign_disagreement": nan,
            "topk_rank_corr": nan,
            "topk_overlap": nan,
            "magnitude_disagreement": nan,
        },
        {"rank_corr": nan, "topk_flip_rate": nan},
    )
    assert report.overall.startswith("INSUFFICIENT EVIDENCE")
    assert '"value": null' in report.to_json()


def test_skipped_sensitivity_is_reported_as_missing_evidence():
    report = build_trust_report(
        1.0,
        2.0,
        0.0,
        None,
        {"topk_rank_corr": 1.0, "sign_agreement": 1.0},
        {
            "sign_disagreement": 0.0,
            "topk_rank_corr": 1.0,
            "topk_overlap": 1.0,
            "magnitude_disagreement": 0.0,
        },
    )
    assert report.overall.startswith("INSUFFICIENT EVIDENCE")


def test_report_thresholds_are_configurable():
    common = dict(
        removal_corr=0.6,
        comprehensiveness=2.0,
        lime_infidelity=0.0,
        sensitivity_value=0.0,
        stability={"topk_rank_corr": 1.0, "sign_agreement": 1.0},
        disagreement={
            "sign_disagreement": 0.0,
            "topk_rank_corr": 1.0,
            "topk_overlap": 1.0,
            "magnitude_disagreement": 0.0,
        },
    )
    assert build_trust_report(**common).overall.startswith("NO ISSUES")
    strict = build_trust_report(**common, thresholds={"removal": (0.9, 0.8)})
    assert strict.overall.startswith("MIXED")

    try:
        build_trust_report(**common, thresholds={"removal": (0.2, 0.8)})
    except ValueError as exc:
        assert "invalid order" in str(exc)
    else:
        raise AssertionError("reversed higher-is-better thresholds must be rejected")


def test_auto_shap_method_matches_explicit():
    X, y = _linear_data()
    yb = (y > 0).astype(int)

    # Tree model -> should use TreeExplainer.
    rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X, yb)
    assert np.allclose(
        shap_attributions(rf, X[:3], method="auto"),
        shap_attributions(rf, X[:3], method="tree"),
    )

    # Linear classifier -> should use LinearExplainer (log-odds space).
    lr = LogisticRegression(max_iter=1000).fit(X, yb)
    assert np.allclose(
        shap_attributions(lr, X[:3], X_background=X[:100], method="auto"),
        shap_attributions(lr, X[:3], X_background=X[:100], method="linear"),
    )

    # Linear regressor -> LinearExplainer (raw output space).
    lm = LinearRegression().fit(X, y)
    assert np.allclose(
        shap_attributions(lm, X[:3], X_background=X[:100], method="auto"),
        shap_attributions(lm, X[:3], X_background=X[:100], method="linear"),
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
