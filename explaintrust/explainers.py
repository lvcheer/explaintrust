"""Unified attribution interface over SHAP and LIME.

Post-hoc explainers return attributions in different shapes and conventions
(SHAP values, LIME coefficient lists, per-class or per-instance). Everything
downstream wants the same thing: a matrix ``A`` of shape ``(n_instances, d)``
where ``A[i, j]`` is the *signed* contribution of feature ``j`` to the
prediction for instance ``i``.

**A note on output space (this matters).** SHAP's native output depends on both
the model and explainer: for example, sklearn random forests explain
probabilities while gradient boosting classifiers explain raw margins. LIME and
all perturbation metrics must use that same output space. This module detects
the native SHAP space and uses it consistently throughout one analysis.

This is exactly the kind of detail a "just run SHAP and show a plot" tool gets
wrong, and the whole point of explaintrust is not to.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def prediction_output_space(model) -> str:
    """Return the scalar output space used by SHAP for ``model``.

    TreeExplainer's ``model_output='raw'`` is not one universal scale: SHAP
    exposes whether a fitted tree model's raw output is a probability, log-odds,
    or a raw regression value. Linear and model-agnostic classifier explainers
    use the decision margin (log-odds for binary classifiers).
    """
    if not hasattr(model, "predict_proba"):
        return "raw"
    if _is_tree_model(model):
        import shap

        tree_output = getattr(shap.TreeExplainer(model).model, "tree_output", None)
        if tree_output == "probability":
            return "probability"
        if tree_output == "log_odds":
            return "log_odds"
        raise ValueError(
            f"unsupported TreeSHAP output space {tree_output!r}; "
            "use a supported sklearn-style model or an explicit predictor"
        )
    return "log_odds"


def scalar_predictor(model, class_index: int = 1, output_space: Optional[str] = None):
    """Return ``f(X) -> 1D`` in the same scalar space as the explanation.

    ``output_space`` may be ``"probability"``, ``"log_odds"``, or ``"raw"``.
    By default it is detected with :func:`prediction_output_space`.
    """
    space = output_space or prediction_output_space(model)
    if space == "probability":
        if not hasattr(model, "predict_proba"):
            raise ValueError("probability output requires model.predict_proba()")
        return lambda X: np.asarray(model.predict_proba(X))[:, class_index]
    if space == "log_odds":
        if hasattr(model, "decision_function"):
            def _decision(X):
                score = np.asarray(model.decision_function(X))
                return score if score.ndim == 1 else score[:, class_index]
            return _decision
        if not hasattr(model, "predict_proba"):
            raise ValueError("log-odds output requires decision_function() or predict_proba()")

        def _log_odds(X):
            p = np.clip(np.asarray(model.predict_proba(X))[:, class_index], 1e-9, 1 - 1e-9)
            return np.log(p / (1.0 - p))
        return _log_odds
    if space != "raw":
        raise ValueError(f"unknown output space: {space!r}")
    return lambda X: np.asarray(model.predict(X)).ravel()


def to_contribution_scale(attributions: np.ndarray, X, X_background) -> np.ndarray:
    """Convert coefficient/local-effect attributions to SHAP-comparable units.

    LIME weights are local *slopes* (∂f/∂x_j), whereas SHAP values are
    *contributions* relative to a baseline: for a linear model,
    ``SHAP_j = coef_j · (x_j − mean_j)`` while ``LIME_j = coef_j``. Comparing
    them directly confuses a difference in baseline convention with genuine
    disagreement. Multiplying LIME weights by the feature's deviation from the
    background mean creates an exact match for a linear model and a documented
    heuristic comparison for nonlinear models.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(np.asarray(X_background, dtype=float), axis=0)
    return np.asarray(attributions, dtype=float) * (X - mean)


def _shap_to_matrix(values, class_index: int = 1) -> np.ndarray:
    """Normalize shap's heterogeneous outputs to (n_instances, d)."""
    if isinstance(values, list):
        # Older SHAP returns [class_0, class_1, ...]. Check this before
        # np.asarray(), which would turn the list into (classes, n, d).
        values = values[class_index]
    v = np.asarray(values, dtype=float)
    if v.ndim == 3:
        # Modern SHAP: (n, d, n_classes).
        v = v[:, :, class_index]
    if v.ndim != 2:
        raise ValueError(f"expected SHAP values with 2 or 3 dimensions, got shape {v.shape}")
    return v


def _is_tree_model(model) -> bool:
    """Best-effort detection of the tree/ensemble models TreeExplainer supports."""
    return hasattr(model, "feature_importances_") and not hasattr(model, "coef_")


def _is_linear_model(model) -> bool:
    """Best-effort detection of the linear models LinearExplainer supports."""
    return hasattr(model, "coef_")


def _auto_shap_method(model) -> str:
    """Pick a SHAP explainer appropriate for the model type."""
    if _is_tree_model(model):
        return "tree"
    if _is_linear_model(model):
        return "linear"
    return "kernel"


def shap_attributions(
    model,
    X,
    X_background=None,
    method: str = "tree",
    class_index: int = 1,
    nsamples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Compute SHAP values as an ``(n, d)`` attribution matrix.

    Values remain in the explainer's native scalar output space. Use
    :func:`scalar_predictor` for a model callable in that same space.

    Parameters
    ----------
    method : {"tree", "linear", "kernel", "auto"}
        "tree" uses ``TreeExplainer`` (fast, exact for tree ensembles, but only
        for tree models). "linear" uses ``LinearExplainer`` (closed-form, for
        linear models; requires ``X_background``). "kernel" uses
        ``KernelExplainer`` (model-agnostic but stochastic — use the same
        ``seed`` to reproduce). "auto" picks among the three based on the model.
    """
    import shap

    X = np.asarray(X, dtype=float)

    if method == "auto":
        method = _auto_shap_method(model)

    if method == "tree":
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        return _shap_to_matrix(values, class_index=class_index)

    if method == "linear":
        if X_background is None:
            raise ValueError("LinearExplainer requires X_background (reference data).")
        explainer = shap.LinearExplainer(model, np.asarray(X_background, dtype=float))
        values = explainer.shap_values(X)
        return _shap_to_matrix(values, class_index=class_index)

    if method == "kernel":
        if X_background is None:
            raise ValueError("KernelExplainer requires X_background (reference data).")
        pred = scalar_predictor(model, class_index=class_index)
        explainer = shap.KernelExplainer(pred, np.asarray(X_background, dtype=float))
        values = explainer.shap_values(X, nsamples=nsamples, random_state=seed)
        return _shap_to_matrix(values, class_index=class_index)

    raise ValueError(f"unknown method: {method!r}")


def lime_attributions(
    model,
    X,
    X_background,
    feature_names=None,
    num_features=None,
    num_samples: int = 2000,
    seed: int = 0,
    class_index: int = 1,
    output_space: Optional[str] = None,
) -> np.ndarray:
    """Compute LIME tabular attributions as an (n, d) matrix.

    LIME is run in *regression* mode against ``scalar_predictor`` so its weights
    use the same model-output space as SHAP. Continuous features are left
    undiscretized and the fitted standardized-coordinate coefficients are
    converted back to original feature units before they are returned.
    """
    from lime import lime_tabular

    X = np.asarray(X, dtype=float)
    X_bg = np.asarray(X_background, dtype=float)
    d = X.shape[1]
    num_features = num_features or d

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(d)]

    predict_fn = scalar_predictor(model, class_index=class_index, output_space=output_space)
    explainer = lime_tabular.LimeTabularExplainer(
        X_bg,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )

    attrs = np.zeros((len(X), d))
    for i, x in enumerate(X):
        exp = explainer.explain_instance(
            x,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )
        # In regression mode lime writes the correctly-signed weights under
        # key 1 (key 0 is the sign-flipped copy); see lime_tabular.py.
        local_exp = exp.local_exp.get(1, next(iter(exp.local_exp.values())))
        # LIME fits its local model on z_j=(x_j-mean_j)/scale_j. Convert the
        # returned coefficient d f / d z_j back to the original feature unit
        # d f / d x_j before exposing it as a gradient attribution.
        for feat_idx, weight in local_exp:
            j = int(feat_idx)
            attrs[i, j] = weight / explainer.scaler.scale_[j]

    return attrs
