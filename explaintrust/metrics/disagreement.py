"""Cross-explainer disagreement: when SHAP and LIME tell different stories.

Different explanation methods answer subtly different questions and make
different assumptions, so they frequently *disagree* — sometimes ranking
features in opposite orders or even flipping signs. When two reasonable
methods disagree about a prediction, neither should be trusted at face value
(Krishna et al., "The Disagreement Problem in Explainable ML", CACM 2024).

This module quantifies that disagreement point-wise.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def explainer_disagreement(attr_a: np.ndarray, attr_b: np.ndarray, top_k: int = 3) -> dict:
    """Quantify disagreement between two attribution vectors (same instance).

    Parameters
    ----------
    attr_a, attr_b : np.ndarray
        Signed attribution vectors of shape (d,) from two explainers.
    top_k : int
        For the top-k overlap.

    Returns
    -------
    dict with keys:
        "sign_disagreement" — fraction of features with opposite (non-zero)
                              signs (lower better)
        "rank_corr"         — Spearman correlation of the two rankings
                              (higher better)
        "topk_overlap"      — Jaccard overlap of top-k feature sets
                              (higher better)
        "per_feature_gap"   — |attr_a − attr_b| normalized by the mean |attr|
                              magnitude (vector)
    """
    a = np.asarray(attr_a, dtype=float)
    b = np.asarray(attr_b, dtype=float)

    # Sign disagreement only counts features where BOTH explainers have a
    # non-trivial opinion; a feature both ignore is not a disagreement.
    mask = (np.abs(a) > 1e-9) & (np.abs(b) > 1e-9)
    if mask.sum() > 0:
        sign_dis = float(np.mean(np.sign(a[mask]) != np.sign(b[mask])))
    else:
        sign_dis = 0.0

    c = stats.spearmanr(a, b).correlation
    rank_corr = float(c) if c is not None else float("nan")

    top_a = set(np.argsort(np.abs(a))[::-1][:top_k])
    top_b = set(np.argsort(np.abs(b))[::-1][:top_k])
    topk_overlap = len(top_a & top_b) / top_k

    magnitude = np.mean(np.abs(a)) + np.mean(np.abs(b)) + 1e-12
    per_feature_gap = np.abs(a - b) / (0.5 * magnitude)

    return {
        "sign_disagreement": sign_dis,
        "rank_corr": rank_corr,
        "topk_overlap": topk_overlap,
        "per_feature_gap": per_feature_gap,
    }
