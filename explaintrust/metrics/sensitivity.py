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
        L2 radius in background-standardized feature coordinates.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    background = np.asarray(X_background, dtype=float)
    feature_scale = np.std(background, axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)

    attr_x = np.asarray(explainer(x), dtype=float)

    worst = 0.0
    for _ in range(n_perturbations):
        # Sample in standardized coordinates, then map the perturbation back to
        # the original feature units. This avoids favoring large-scale columns.
        direction = rng.normal(0.0, 1.0, size=x.shape)
        direction /= (np.linalg.norm(direction) + 1e-12)
        scale = radius * rng.random()
        x_tilde = x + scale * direction * feature_scale

        attr_tilde = np.asarray(explainer(x_tilde), dtype=float)
        change = np.linalg.norm(attr_tilde - attr_x)
        worst = max(worst, float(change))

    return worst
