"""Sensitivity: how much does the explanation change for a tiny input change?

An explanation that flips dramatically when the input is nudged by imperceptible
amounts is not a stable description of the model — it is overfitting to the
point queried. Max-sensitivity (Yeh et al., NeurIPS 2019) is the worst-case
normalized change:

    SENS_MAX = max_{‖x̃ − x‖ ≤ r} ‖attr(x̃) − attr(x)‖

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
        Relative L2 radius around ``x``, expressed as a fraction of ``‖x‖``,
        inside which perturbations are drawn.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    # ``X_background`` is accepted for API consistency with the other metrics;
    # the neighborhood below is defined relative to ``‖x‖`` rather than the
    # background scale.
    _ = np.asarray(X_background, dtype=float)

    attr_x = np.asarray(explainer(x), dtype=float)
    norm_x = np.linalg.norm(x)

    worst = 0.0
    for _ in range(n_perturbations):
        # Sample a point on/inside the sphere of radius r * ‖x‖.
        direction = rng.normal(0.0, 1.0, size=x.shape)
        direction /= (np.linalg.norm(direction) + 1e-12)
        scale = radius * (norm_x + 1e-12) * rng.random()
        x_tilde = x + scale * direction

        attr_tilde = np.asarray(explainer(x_tilde), dtype=float)
        denom = np.linalg.norm(x_tilde - x) + 1e-12
        change = np.linalg.norm(attr_tilde - attr_x) / denom
        worst = max(worst, float(change))

    return worst
