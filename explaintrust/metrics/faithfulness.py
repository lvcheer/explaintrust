"""Faithfulness: does the explanation actually track the model's behavior?

Faithfulness metrics come in two families, and **they are not interchangeable**:

* **Gradient/coefficient-scale explanations** (LIME weights, gradients,
  Integrated Gradients) make the claim ``f(x̃) ≈ f(x) + φ·(x̃ − x)``. The
  right check is **infidelity** (Yeh et al., NeurIPS 2019): how well does that
  local linear surrogate predict the model's actual change under perturbation.

* **Contribution-scale explanations** (SHAP values) make the claim
  ``Σ φ_i ≈ f(x) − E[f(X)]``. SHAP values are *not* gradients, so feeding them
  into the infidelity formula is a category error. The right checks are
  **ablation-based**: does removing a feature actually move the prediction as
  much as its attribution says it should?

Getting this right is the difference between "a number that looks rigorous" and
"a number that means something". We keep the two families separate and label
which explainer each metric is valid for.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _perturb(
    x: np.ndarray,
    X_background: np.ndarray,
    n_perturbations: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``n_perturbations`` perturbed copies of ``x``."""
    x = np.asarray(x, dtype=float)
    d = x.shape[0]

    if strategy == "gaussian":
        std = np.std(X_background, axis=0) + 1e-12
        noise = rng.normal(0.0, 0.5 * std, size=(n_perturbations, d))
        return x[None, :] + noise

    if strategy == "mask":
        x_tilde = np.repeat(x[None, :], n_perturbations, axis=0)
        median = np.median(X_background, axis=0)
        n_mask = rng.integers(1, d, size=n_perturbations)
        for i in range(n_perturbations):
            idx = rng.choice(d, size=int(n_mask[i]), replace=False)
            x_tilde[i, idx] = median[idx]
        return x_tilde

    raise ValueError(f"unknown perturbation strategy: {strategy!r}")


def infidelity(
    model,
    x: np.ndarray,
    attribution: np.ndarray,
    X_background: np.ndarray,
    n_perturbations: int = 200,
    strategy: str = "gaussian",
    seed: int = 0,
) -> float:
    """Infidelity of a *gradient/coefficient-scale* explanation (lower better).

    INFD = E[(φ·(x̃ − x) − (f(x̃) − f(x)))²]

    ``φ`` must be a local-linear explanation (LIME weights, gradients, etc.).
    **Do not pass SHAP values** — they are contributions, not gradients, and the
    result will not be meaningful. Use ``removal_effect_correlation`` or
    ``comprehensiveness_ratio`` for SHAP.

    ``model`` is a callable returning scalar predictions (see
    ``explaintrust.scalar_predictor``).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    attr = np.asarray(attribution, dtype=float)

    x_tilde = _perturb(x, X_background, n_perturbations, strategy, rng)
    delta_x = x_tilde - x

    explained = delta_x @ attr
    actual = np.asarray(model(x_tilde)).ravel() - float(model(x[None, :])[0])

    return float(np.mean((explained - actual) ** 2))


def removal_effect_correlation(
    model,
    x: np.ndarray,
    attribution: np.ndarray,
    X_background: np.ndarray,
    seed: int = 0,
) -> float:
    """Spearman correlation between |attribution| and |removal effect| (higher better).

    Valid for any attribution (SHAP included). For each feature we replace it
    with its background median and measure how much the prediction moves. If
    the explanation is faithful, features it deems important should be the ones
    whose ablation actually matters.
    """
    x = np.asarray(x, dtype=float)
    attr = np.asarray(attribution, dtype=float)
    median = np.median(np.asarray(X_background, dtype=float), axis=0)
    base = float(model(x[None, :])[0])

    effects = np.zeros_like(attr)
    for j in range(len(x)):
        x_ablated = x.copy()
        x_ablated[j] = median[j]
        effects[j] = abs(float(model(x_ablated[None, :])[0]) - base)

    if np.std(effects) < 1e-12 or np.std(attr) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(np.abs(attr), effects).correlation)


def comprehensiveness_ratio(
    model,
    x: np.ndarray,
    attribution: np.ndarray,
    X_background: np.ndarray,
    top_k: int = 3,
    n_random: int = 20,
    seed: int = 0,
) -> float:
    """Ratio of top-k removal effect to random-k removal effect (higher better).

    Replace the top-``top_k`` features (by |attribution|) with background
    medians and measure the prediction change; compare against the mean change
    from removing ``top_k`` *random* features. A value > 1 means the features
    the explanation ranks highest genuinely move the model more than an
    arbitrary set would — i.e. the attribution is not just noise.

    Valid for SHAP and other contribution attributions.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    attr = np.asarray(attribution, dtype=float)
    median = np.median(np.asarray(X_background, dtype=float), axis=0)
    d = len(x)
    base = float(model(x[None, :])[0])

    top_idx = np.argsort(np.abs(attr))[::-1][:top_k]
    x_top = x.copy()
    x_top[top_idx] = median[top_idx]
    drop_top = abs(float(model(x_top[None, :])[0]) - base)

    drops = []
    for _ in range(n_random):
        idx = rng.choice(d, size=top_k, replace=False)
        xr = x.copy()
        xr[idx] = median[idx]
        drops.append(abs(float(model(xr[None, :])[0]) - base))
    drop_rand = float(np.mean(drops))

    return drop_top / (drop_rand + 1e-12)
