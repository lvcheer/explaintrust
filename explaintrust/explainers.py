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

import hashlib
from threading import RLock
from typing import Optional, Union

import numpy as np


# KernelExplainer samples through NumPy's legacy global RNG. Serialize our
# save/seed/restore sections so concurrent calls through this API cannot race.
_KERNEL_RNG_LOCK = RLock()


def _validate_class_index(model, class_index: int) -> bool:
    """Validate binary class selection; return False for regression models."""
    classifier = hasattr(model, "classes_") or hasattr(model, "predict_proba")
    if not classifier:
        return False
    if (isinstance(class_index, (bool, np.bool_))
            or not isinstance(class_index, (int, np.integer)) or class_index not in (0, 1)):
        raise ValueError("class_index must be 0 or 1, indexing model.classes_ (not a class label)")
    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes)
        if classes.ndim != 1 or len(classes) != 2:
            raise ValueError("only binary classification is supported; model.classes_ must contain two labels")
    return True


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
    For binary classifiers, ``class_index`` selects position 0 or 1 in
    ``model.classes_``, not the label itself. A one-dimensional decision margin
    belongs to class 1 and is negated for class 0. ``raw`` is for regression;
    classifier labels from ``predict`` are not class-specific scalar scores.
    Regression ignores ``class_index``.
    """
    classifier = _validate_class_index(model, class_index)
    space = output_space or prediction_output_space(model)
    if space == "probability":
        if not hasattr(model, "predict_proba"):
            raise ValueError("probability output requires model.predict_proba()")
        return lambda X: np.asarray(model.predict_proba(X))[:, class_index]
    if space == "log_odds":
        if hasattr(model, "get_booster"):
            def _margin(X):
                score = np.asarray(model.predict(X, output_margin=True))
                return -score if class_index == 0 else score
            return _margin
        if hasattr(model, "decision_function"):
            def _decision(X):
                score = np.asarray(model.decision_function(X))
                if score.ndim == 1:
                    return -score if class_index == 0 else score
                return score[:, class_index]
            return _decision
        if not hasattr(model, "predict_proba"):
            raise ValueError("log-odds output requires decision_function() or predict_proba()")

        def _log_odds(X):
            p = np.clip(np.asarray(model.predict_proba(X))[:, class_index], 1e-9, 1 - 1e-9)
            return np.log(p / (1.0 - p))
        return _log_odds
    if space != "raw":
        raise ValueError(f"unknown output space: {space!r}")
    if classifier:
        raise ValueError("raw classifier labels are not class-specific scores; use probability or log_odds")
    return lambda X: np.asarray(model.predict(X)).ravel()


def to_contribution_scale(attributions: np.ndarray, X, X_background) -> np.ndarray:
    """Convert coefficient/local-effect attributions to SHAP-comparable units.

    LIME weights are local *slopes* (∂f/∂x_j), whereas SHAP values are
    *contributions* relative to a baseline: for a linear model,
    ``SHAP_j = coef_j · (x_j − mean_j)`` while ``LIME_j = coef_j``. Comparing
    them directly confuses a difference in baseline convention with genuine
    disagreement. Multiplying LIME weights by the feature's deviation from the
    background mean matches independent-background LinearSHAP for exact linear
    coefficients. Fitted LIME coefficients retain estimation error; for nonlinear
    models this remains a heuristic, not an exact SHAP decomposition.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(np.asarray(X_background, dtype=float), axis=0)
    return np.asarray(attributions, dtype=float) * (X - mean)


def _shap_to_matrix(values, class_index: int = 1, binary_classifier: bool = False) -> np.ndarray:
    """Normalize SHAP outputs; native binary single-output values explain class 1.

    Keep ``binary_classifier=False`` for KernelSHAP: its scalar predictor has
    already selected the requested class, so negating again would be wrong.
    """
    single_output = not isinstance(values, list) and np.asarray(values).ndim == 2
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
    if binary_classifier and single_output and class_index == 0:
        v = -v
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
    *,
    tree_perturbation: str = "auto",
    return_context: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
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
    class_index : int
        Position 0 or 1 in a binary classifier's ``model.classes_``. Native
        single-output margins/attributions are negated for class 0; per-class
        probability attributions are selected directly. Ignored for regression.
    X_background : array-like or None
        The complete reference matrix; no implicit SHAP subsampling is used.
        For large references, select rows explicitly before calling both SHAP
        and LIME. Linear and kernel methods require a background.
    tree_perturbation : {"auto", "interventional", "tree_path_dependent"}
        For trees, "auto" uses interventional SHAP when X_background is supplied
        and training-path counts otherwise. Explicit interventional mode requires
        a background; explicit path mode requires omitting it. Non-tree methods
        require this option to remain "auto".
    return_context : bool
        If True, return ``(values, context)`` including the selected-class SHAP
        base value, actual method, and background count, mean and SHA-256 digest.
        Otherwise return the attribution matrix as before.

    Notes
    -----
    For deterministic model predictions in the same dependency environment,
    KernelSHAP sampling is controlled by ``seed``. NumPy's global RNG state is
    restored after the call, including on failure. KernelSHAP calls through
    this API are serialized because SHAP uses that global RNG. Unrelated
    threads using ``np.random`` or calling SHAP directly do not share this
    lock; use separate processes or independent ``np.random.Generator``
    instances for concurrent random work outside this API.
    """
    import shap

    X = np.asarray(X, dtype=float)
    classifier = _validate_class_index(model, class_index)

    if method == "auto":
        method = _auto_shap_method(model)

    if tree_perturbation not in ("auto", "interventional", "tree_path_dependent"):
        raise ValueError(f"unknown tree_perturbation: {tree_perturbation!r}")
    if method != "tree" and tree_perturbation != "auto":
        raise ValueError("tree_perturbation is only applicable to the tree method")
    background = None
    if X_background is not None:
        background = np.asarray(X_background, dtype=float)
        if (X.ndim != 2 or background.ndim != 2 or len(background) == 0
                or background.shape[1] != X.shape[1] or not np.all(np.isfinite(background))):
            raise ValueError("X_background must be finite, non-empty and match X's feature columns")

    if method == "tree":
        mode = tree_perturbation
        if mode == "auto":
            mode = "interventional" if background is not None else "tree_path_dependent"
        if mode == "interventional" and background is None:
            raise ValueError("interventional TreeSHAP requires X_background")
        if mode == "tree_path_dependent" and background is not None:
            raise ValueError("tree_path_dependent uses training-path counts; omit X_background")
        masker = shap.maskers.Independent(background, max_samples=len(background)) if background is not None else None
        explainer = shap.TreeExplainer(model, data=masker, feature_perturbation=mode, model_output="raw")
        used_background = explainer.data
        values = explainer.shap_values(X)
        matrix = _shap_to_matrix(values, class_index=class_index, binary_classifier=classifier)

    elif method == "linear":
        if background is None:
            raise ValueError("LinearExplainer requires X_background (reference data).")
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.LinearExplainer(model, masker)
        mode = "interventional"
        used_background = explainer.masker.data
        values = explainer.shap_values(X)
        matrix = _shap_to_matrix(values, class_index=class_index, binary_classifier=classifier)

    elif method == "kernel":
        if background is None:
            raise ValueError("KernelExplainer requires X_background (reference data).")
        pred = scalar_predictor(model, class_index=class_index)
        with _KERNEL_RNG_LOCK:
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                explainer = shap.KernelExplainer(pred, background)
                values = explainer.shap_values(X, nsamples=nsamples)
            finally:
                np.random.set_state(state)
        mode = "background_replacement"
        used_background = explainer.data.data
        matrix = _shap_to_matrix(values, class_index=class_index)
    else:
        raise ValueError(f"unknown method: {method!r}")

    if background is not None and not np.array_equal(used_background, background):
        raise RuntimeError("SHAP changed the supplied background; a shared-reference comparison is not valid")
    if not return_context:
        return matrix

    base = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if len(base) == 1:
        expected_value = float(base[0])
        if classifier and class_index == 0 and method != "kernel":
            expected_value = -expected_value
    elif classifier and len(base) == 2:
        expected_value = float(base[class_index])
    else:
        raise ValueError("expected a scalar or binary-class SHAP base value")

    background_context = None
    if used_background is not None:
        reference = np.asarray(used_background, dtype="<f8", order="C")
        digest = hashlib.sha256(str(reference.shape).encode("ascii") + reference.tobytes()).hexdigest()
        background_context = {
            "n_rows": len(reference), "n_features": reference.shape[1],
            "mean": reference.mean(axis=0).tolist(), "sha256": digest,
        }
    return matrix, {
        "method": method,
        "feature_perturbation": mode,
        "masker": "Independent" if mode == "interventional" else None,
        "output_space": prediction_output_space(model),
        "class_index": int(class_index) if classifier else None,
        "expected_value": expected_value,
        "background_source": "provided_rows" if background is not None else "training_path_counts",
        "background": background_context,
        "shap_version": shap.__version__,
    }


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
    ``class_index`` selects position 0 or 1 in ``model.classes_`` for binary
    classification, following :func:`scalar_predictor`; regression ignores it.
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
