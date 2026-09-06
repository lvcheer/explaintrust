"""Background alignment includes the actual rows, SHAP mode and base value."""

import json

import numpy as np
import pytest
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from explaintrust import scalar_predictor, shap_attributions, to_contribution_scale


def data():
    return np.random.default_rng(3).normal(size=(500, 3))


def test_linear_uses_entire_background_instead_of_implicit_subsample():
    X = data()
    model = LinearRegression().fit(X, X @ np.array([1., 2., 3.]))
    actual = shap_attributions(model, X[:3], X, method="linear")
    expected = to_contribution_scale(np.tile(model.coef_, (3, 1)), X[:3], X)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["linear", "forest", "boosting", "regression"])
@pytest.mark.parametrize("class_index", [0, 1])
def test_base_value_reconstructs_selected_output_using_full_background(kind, class_index):
    X = data()
    y = X[:, 0] - X[:, 1] + X[:, 2] ** 2
    models = {
        "linear": LogisticRegression(),
        "forest": RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0),
        "boosting": GradientBoostingClassifier(n_estimators=8, max_depth=3, random_state=0),
        "regression": RandomForestRegressor(n_estimators=8, max_depth=3, random_state=0),
    }
    model = models[kind].fit(X, y if kind == "regression" else y > 0)
    # More than 100 rows catches SHAP's default Independent-masker subsampling.
    bg = X[100:350]
    values, context = shap_attributions(
        model, X[:4], bg, method="auto", class_index=class_index, return_context=True,
    )
    predict = scalar_predictor(model, class_index=class_index)
    assert context["background"]["n_rows"] == 250
    np.testing.assert_allclose(context["background"]["mean"], bg.mean(axis=0), atol=1e-12)
    assert context["feature_perturbation"] == "interventional"
    assert context["expected_value"] == pytest.approx(predict(bg).mean(), abs=1e-10)
    np.testing.assert_allclose(values.sum(axis=1) + context["expected_value"], predict(X[:4]), atol=1e-7)
    json.dumps(context, allow_nan=False)


def test_tree_background_choice_changes_reference_and_attributions():
    X = data()
    model = RandomForestRegressor(n_estimators=8, max_depth=3, random_state=0).fit(X, X[:, 0])
    low, high = X[X[:, 0] < -0.5], X[X[:, 0] > 0.5]
    a, ca = shap_attributions(model, X[:3], low, return_context=True)
    b, cb = shap_attributions(model, X[:3], high, return_context=True)
    assert not np.allclose(a, b)
    assert ca["expected_value"] < cb["expected_value"]
    assert ca["background"]["sha256"] != cb["background"]["sha256"]


def test_no_background_preserves_legacy_tree_path_mode():
    X = data()
    model = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0).fit(X, X[:, 0])
    actual, context = shap_attributions(model, X[:3], return_context=True)
    expected = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(X[:3])
    np.testing.assert_allclose(actual, expected)
    assert context["feature_perturbation"] == "tree_path_dependent"
    assert context["background"] is None
    assert context["background_source"] == "training_path_counts"


@pytest.mark.parametrize("mode,with_background", [("interventional", False), ("tree_path_dependent", True)])
def test_tree_cannot_silently_drop_or_invent_a_background(mode, with_background):
    X = data()
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, X[:, 0])
    with pytest.raises(ValueError, match="background|X_background"):
        shap_attributions(model, X[:1], X[:10] if with_background else None, tree_perturbation=mode)


def test_kernel_context_base_value_is_already_in_selected_class_space():
    X = data()
    model = LogisticRegression().fit(X, X[:, 0] > 0)
    values, context = shap_attributions(model, X[:2], X[:12], method="kernel", class_index=0, return_context=True)
    predict = scalar_predictor(model, class_index=0)
    assert context["expected_value"] == pytest.approx(predict(X[:12]).mean())
    np.testing.assert_allclose(values.sum(axis=1) + context["expected_value"], predict(X[:2]))
    assert context["background"]["n_rows"] == 12
