"""Synthetic datasets for demos and tests.

The datasets are deliberately built around the two phenomena that make
post-hoc explanations untrustworthy in practice:

1. **Collinearity** — a non-causal feature that is highly correlated with a
   causal one. SHAP and LIME tend to *split* (or arbitrarily assign) the
   importance between the two, so their rankings disagree and their answers
   are unstable. This is the classic "explainers disagree" trigger.
2. **Distribution shift** — we can perturb the data-generating process to
   simulate a model/explanation being asked about inputs it was not validated
   on, which is what the distribution-verification metric probes.
"""

from __future__ import annotations

import numpy as np

# Feature roles for the default dataset. Exposed so the demo can show what
# the "ground truth" is — something real users never have, which is exactly
# why we need metrics.
FEATURE_ROLES = {
    "x0": "causal",
    "x1": "causal (interacts with x0)",
    "x2": "collinear with x0 (NOT causal)",
    "x3": "noise",
    "x4": "noise",
    "x5": "noise",
    "x6": "noise",
    "x7": "noise",
    "x8": "noise",
    "x9": "noise",
}


def make_collinear_dataset(n: int = 2000, seed: int = 0):
    """Generate a binary classification dataset with a known generative process.

    Returns
    -------
    X : np.ndarray, shape (n, 10)
    y : np.ndarray, shape (n,), in {0, 1}
    feature_names : list[str]
    """
    rng = np.random.default_rng(seed)
    d = 10
    X = rng.normal(0.0, 1.0, size=(n, d))

    # x0, x1 are causal; x1 also interacts with x0 (non-additive), which makes
    # any single additive explanation necessarily incomplete.
    # x2 is a collinear copy of x0 with noise -> explainers fight over x0 vs x2.
    X[:, 2] = 0.85 * X[:, 0] + rng.normal(0.0, 0.35, size=n)

    logit = (
        1.5 * X[:, 0]
        - 1.0 * X[:, 1]
        + 0.6 * X[:, 0] * X[:, 1]
        + 0.3 * X[:, 0] ** 2
        - 0.25
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)

    feature_names = [f"x{i}" for i in range(d)]
    return X, y, feature_names


def shift_distribution(X, y, feature_names=None, shift="x1_drift", seed=0):
    """Apply a distribution shift to (X, y) to simulate drifted inputs.

    This produces a second dataset whose *data-generating process* differs from
    the original (e.g. the causal feature x1 is re-centered / rescaled). An
    explanation that was validated only on the original distribution may not
    hold here — which is precisely what ``distribution``-level metrics check.

    Parameters
    ----------
    shift : str
        One of "x1_drift" (recenter + rescale x1) or "x0_drift".
    """
    X = np.asarray(X, dtype=float).copy()
    y = np.asarray(y).copy()
    rng = np.random.default_rng(seed)

    if shift == "x1_drift":
        X[:, 1] = X[:, 1] * 2.5 + 1.2
    elif shift == "x0_drift":
        X[:, 0] = X[:, 0] * 1.8 - 0.8
    elif shift == "x2_decouple":
        # Break the collinearity: now x2 is pure noise, so an explanation that
        # leaned on x2 (a collinearity artifact) becomes visibly wrong.
        X[:, 2] = rng.normal(0.0, 1.0, size=len(X))
    else:
        raise ValueError(f"unknown shift: {shift!r}")

    return X, y, feature_names
