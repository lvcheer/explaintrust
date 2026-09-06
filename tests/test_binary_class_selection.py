"""Class selection must refer to the same output in prediction and attribution."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from explaintrust import lime_attributions, scalar_predictor, shap_attributions
from explaintrust.explainers import _shap_to_matrix


@pytest.fixture(params=["linear", "gradient_boosting", "random_forest"])
def classifier(request):
    X = np.random.default_rng(12).normal(size=(160, 4))
    # Non-0/1 labels ensure class_index means position, not the class label.
    y = np.where(2 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3] > 0, 42, 7)
    models = {
        "linear": LogisticRegression(),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0),
        "random_forest": RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0),
    }
    return models[request.param].fit(X, y), X, request.param


def test_scalar_predictions_match_selected_probability_and_margin(classifier):
    model, X, _ = classifier
    X = X[:5]
    np.testing.assert_array_equal(model.classes_, [7, 42])
    for index in [0, 1]:
        probability = scalar_predictor(model, class_index=index, output_space="probability")(X)
        np.testing.assert_allclose(probability, model.predict_proba(X)[:, index])
        margin = scalar_predictor(model, class_index=index, output_space="log_odds")(X)
        if hasattr(model, "decision_function"):
            expected = model.decision_function(X) * (-1 if index == 0 else 1)
        else:
            p = np.clip(model.predict_proba(X)[:, index], 1e-9, 1 - 1e-9)
            expected = np.log(p / (1 - p))
        np.testing.assert_allclose(margin, expected)


@pytest.mark.parametrize("method", ["auto", "native", "kernel"])
def test_shap_class_zero_matches_output_without_double_negation(classifier, method):
    model, X, kind = classifier
    if method == "native":
        method = "linear" if kind == "linear" else "tree"
    negative = shap_attributions(model, X[:5], X[20:40], method=method, class_index=0, nsamples=100)
    positive = shap_attributions(model, X[:5], X[20:40], method=method, class_index=1, nsamples=100)
    assert np.max(np.abs(positive)) > 1e-3
    np.testing.assert_allclose(negative, -positive, atol=1e-8)
    # Check against independently selected model outputs, not only the wrapper.
    if kind == "random_forest":
        expected = model.predict_proba(X[:5])[:, 0]
    else:
        expected = -model.decision_function(X[:5])
    np.testing.assert_allclose(
        negative.sum(axis=1) - negative.sum(axis=1)[0], expected - expected[0], atol=1e-7,
    )


def test_lime_class_zero_slopes_have_selected_direction(classifier):
    model, X, _ = classifier
    negative = lime_attributions(model, X[:2], X[20:40], class_index=0, num_samples=1000, seed=3)
    positive = lime_attributions(model, X[:2], X[20:40], class_index=1, num_samples=1000, seed=3)
    assert np.max(np.abs(positive)) > 1e-3
    np.testing.assert_allclose(negative, -positive, atol=1e-8)


@pytest.mark.parametrize("entry", ["predictor", "shap", "lime"])
@pytest.mark.parametrize("index", [-1, 2, 1.5, True, "class zero"])
def test_invalid_binary_indices_are_rejected(entry, index):
    X = np.random.default_rng(4).normal(size=(40, 3))
    model = LogisticRegression().fit(X, X[:, 0] > 0)
    with pytest.raises(ValueError, match="class_index"):
        if entry == "predictor":
            scalar_predictor(model, class_index=index)(X[:1])
        elif entry == "shap":
            shap_attributions(model, X[:1], X, method="auto", class_index=index)
        else:
            lime_attributions(model, X[:1], X, class_index=index, num_samples=100)


@pytest.mark.parametrize("entry", ["predictor", "shap", "lime"])
def test_multiclass_cannot_use_binary_sign_convention(entry):
    X = np.random.default_rng(2).normal(size=(60, 3))
    model = LogisticRegression().fit(X, np.arange(60) % 3)
    with pytest.raises(ValueError, match="binary classification"):
        if entry == "predictor":
            scalar_predictor(model, class_index=0)
        elif entry == "shap":
            shap_attributions(model, X[:1], X, method="auto", class_index=0)
        else:
            lime_attributions(model, X[:1], X, class_index=0, num_samples=100)


@pytest.mark.parametrize("method", ["linear", "kernel", "tree"])
def test_regression_is_not_negated_by_class_zero(method):
    X = np.random.default_rng(3).normal(size=(30, 3))
    model = RandomForestRegressor(n_estimators=5, random_state=0) if method == "tree" else LinearRegression()
    model.fit(X, X @ np.array([2., -1., 0.5]))
    a = shap_attributions(model, X[:2], X, method=method, class_index=0)
    b = shap_attributions(model, X[:2], X, method=method, class_index=1)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(scalar_predictor(model, class_index=0)(X), model.predict(X))


def test_raw_classifier_labels_are_not_class_specific_scores():
    X = np.random.default_rng(3).normal(size=(30, 3))
    model = LogisticRegression().fit(X, X[:, 0] > 0)
    with pytest.raises(ValueError, match="raw.*class"):
        scalar_predictor(model, class_index=0, output_space="raw")


@pytest.mark.parametrize("output_format", ["old_list", "modern_array"])
@pytest.mark.parametrize("index", [0, 1])
def test_explicit_shap_class_axis_is_selected_without_sign_flip(output_format, index):
    class_one = np.arange(8.).reshape(2, 4)
    per_class = [-class_one, class_one]
    values = per_class if output_format == "old_list" else np.stack(per_class, axis=-1)
    result = _shap_to_matrix(values, class_index=index, binary_classifier=True)
    np.testing.assert_array_equal(result, per_class[index])
