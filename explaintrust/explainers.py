"""Unified attribution interface over SHAP and LIME.

Post-hoc explainers return attributions in different shapes and conventions
(SHAP values, LIME coefficient lists, per-class or per-instance). Everything
downstream wants the same thing: a matrix ``A`` of shape ``(n_instances, d)``
where ``A[i, j]`` is the *signed* contribution of feature ``j`` to the
prediction for instance ``i``.

**A note on output space (this matters).** For classifiers, SHAP values are
expressed in *log-odds* of the positive class, while raw LIME weights are fit in
*probability* space. Comparing the two directly — or running faithfulness
metrics that mix spaces — produces meaningless numbers. We therefore pin every
explainer to a single consistent space:

    * classifier -> log-odds of the positive class
    * regressor  -> raw output

This is exactly the kind of detail a "just run SHAP and show a plot" tool gets
wrong, and the whole point of explaintrust is not to.
"""

from __future__ import annotations

import numpy as np


def scalar_predictor(model, class_index: int = 1):
    """Return a callable ``f(X) -> 1D array`` in the explanation's output space.

    For classifiers we return the *log-odds* of ``class_index`` (so it matches
    TreeSHAP values); for regressors the raw output.
    """
    if hasattr(model, "predict_proba"):
        def _log_odds(X):
            p = np.clip(np.asarray(model.predict_proba(X))[:, class_index], 1e-9, 1 - 1e-9)
            return np.log(p / (1.0 - p))
        return _log_odds
    return lambda X: np.asarray(model.predict(X)).ravel()


def to_contribution_scale(attributions: np.ndarray, X, X_background) -> np.ndarray:
    """Convert coefficient/local-effect attributions to SHAP-comparable units.

    LIME weights are local *slopes* (∂f/∂x_j), whereas SHAP values are
    *contributions* relative to a baseline: for a linear model,
    ``SHAP_j = coef_j · (x_j − mean_j)`` while ``LIME_j = coef_j``. Comparing
    them directly confuses a difference in baseline convention with genuine
    disagreement. Multiplying LIME weights by the feature's deviation from the
    background mean puts both explainers on the same "contribution" scale, so
    sign/rank comparisons become meaningful.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(np.asarray(X_background, dtype=float), axis=0)
    return np.asarray(attributions, dtype=float) * (X - mean)


def _shap_to_matrix(values) -> np.ndarray:
    """Normalize shap's heterogeneous outputs to (n_instances, d)."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 3:
        # (n, d, n_classes) -> take positive class (index 1)
        v = v[:, :, 1]
    elif isinstance(values, list):
        # Older shap returns a list [neg, pos] for binary classifiers.
        v = np.asarray(values[1], dtype=float)
    return v.reshape(v.shape[0], v.shape[1])


def shap_attributions(
    model,
    X,
    X_background=None,
    method: str = "tree",
    class_index: int = 1,
    nsamples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Compute SHAP values as an (n, d) attribution matrix (log-odds space).

    Parameters
    ----------
    method : {"tree", "kernel"}
        "tree" uses ``TreeExplainer`` (fast, exact for tree ensembles, but only
        for tree models). "kernel" uses ``KernelExplainer`` (model-agnostic but
        stochastic — use the same ``seed`` to reproduce).
    """
    import shap

    X = np.asarray(X, dtype=float)

    if method == "tree":
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        return _shap_to_matrix(values)

    if method == "kernel":
        if X_background is None:
            raise ValueError("KernelExplainer requires X_background (reference data).")
        pred = scalar_predictor(model, class_index=class_index)
        explainer = shap.KernelExplainer(pred, np.asarray(X_background, dtype=float))
        values = explainer.shap_values(X, nsamples=nsamples, random_state=seed)
        return _shap_to_matrix(values)

    raise ValueError(f"unknown method: {method!r}")


def lime_attributions(
    model,
    X,
    X_background,
    feature_names=None,
    num_features=None,
    num_samples: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Compute LIME tabular attributions as an (n, d) matrix.

    LIME is run in *regression* mode against ``scalar_predictor`` (log-odds for
    classifiers, raw for regressors) so its weights live in the same output
    space as SHAP values. Continuous features are left undiscretized
    (``discretize_continuous=False``) so weights map 1:1 onto original columns —
    a prerequisite for a fair SHAP-vs-LIME comparison.
    """
    from lime import lime_tabular

    X = np.asarray(X, dtype=float)
    X_bg = np.asarray(X_background, dtype=float)
    d = X.shape[1]
    num_features = num_features or d

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(d)]

    predict_fn = scalar_predictor(model)
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
        for feat_idx, weight in local_exp:
            attrs[i, int(feat_idx)] = weight

    return attrs
