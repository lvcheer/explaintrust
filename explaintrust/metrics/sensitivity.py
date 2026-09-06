"""Sensitivity: how much does the explanation change for a tiny input change?

An explanation that flips dramatically when the input is nudged by imperceptible
amounts is not a stable description of the model — it is overfitting to the
point queried. Max-sensitivity (Yeh et al., NeurIPS 2019) is the worst-case
explanation change in a bounded neighborhood:

    SENS_MAX = max_{‖z̃ − z‖₂ ≤ r} ‖attr(x̃) − attr(x)‖₂

where ``z`` denotes inputs standardized by the background feature scales.

Lower is better. Because it requires *re-explaining* each perturbed input, it is
the most expensive metric here; budget its cost with ``n_perturbations`` and
``explainer`` (prefer the cheap/closed-form explainer when possible).
"""

from __future__ import annotations

import numpy as np


def max_sensitivity(
    explainer,
    x: np.ndarray,
    X_background: np.ndarray,
    n_perturbations: int = 30,
    radius: float = 0.1,
    seed: int = 0,
) -> float:
    """Max-sensitivity of an explanation at a single instance (lower is better).

    Parameters
    ----------
    explainer : callable
        ``explainer(x: np.ndarray) -> attribution vector`` of shape (d,). It is
        called once per perturbation, so pass a fast/closed-form explainer here.
    radius : float
        Positive, finite L2 radius in background-standardized coordinates.

    Raises
    ------
    ValueError
        If inputs or attributions have invalid shapes or non-finite values,
        the perturbation budget is not a positive integer, the radius is not
        positive and finite, or a numerical overflow prevents evaluation.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    background = np.asarray(X_background, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a non-empty, finite one-dimensional vector")
    if (background.ndim != 2 or background.shape[0] == 0
            or background.shape[1] != x.size or not np.all(np.isfinite(background))):
        raise ValueError("X_background must be finite, non-empty, and have shape (n, len(x))")
    if (isinstance(n_perturbations, (bool, np.bool_))
            or not isinstance(n_perturbations, (int, np.integer)) or n_perturbations < 1):
        raise ValueError("n_perturbations must be a positive integer")
    if (not isinstance(radius, (int, float, np.integer, np.floating))
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("radius must be positive and finite")

    feature_scale = np.std(background, axis=0)
    if not np.all(np.isfinite(feature_scale)):
        raise ValueError("background feature scales must be finite")
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)

    def checked_attribution(point):
        attr = np.asarray(explainer(point), dtype=float)
        if attr.shape != x.shape:
            raise ValueError("explainer attribution must have the same shape as x")
        if not np.all(np.isfinite(attr)):
            raise ValueError("explainer attribution must contain only finite values")
        return attr

    attr_x = checked_attribution(x)

    worst = 0.0
    for _ in range(n_perturbations):
        # Sample in standardized coordinates, then map the perturbation back to
        # the original feature units. This avoids favoring large-scale columns.
        direction = rng.normal(0.0, 1.0, size=x.shape)
        direction /= (np.linalg.norm(direction) + 1e-12)
        scale = radius * rng.random()
        x_tilde = x + scale * direction * feature_scale
        if not np.all(np.isfinite(x_tilde)):
            raise ValueError("perturbed inputs must be finite")

        attr_tilde = checked_attribution(x_tilde)
        change = np.linalg.norm(attr_tilde - attr_x)
        if not np.isfinite(change):
            raise ValueError("attribution change must be finite")
        worst = max(worst, float(change))

    return worst
