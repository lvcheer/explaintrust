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
        "sign_disagreement"       — fraction of features with opposite
                                    (non-zero) signs (lower better)
        "rank_corr"               — Spearman correlation of the two rankings
                                    over *all* features (higher better). Degrades
                                    with feature count; prefer ``topk_rank_corr``.
        "topk_rank_corr"          — Spearman correlation over the top-k features
                                    only, robust to many noise features
                                    (higher better)
        "topk_overlap"            — fraction of shared top-k features
                                    (intersection / k; higher better)
        "magnitude_disagreement"  — mean normalized |attr_a − attr_b| gap over
                                    the top-k features (lower better)
        "per_feature_gap"         — |attr_a − attr_b| normalized by that
                                    feature's own scale (vector)
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

    c = stats.spearmanr(np.abs(a), np.abs(b)).correlation
    rank_corr = float(c) if c is not None else float("nan")

    k = min(top_k, len(a), len(b))
    if k < 1:
        raise ValueError("top_k must be at least 1")
    top_a = set(np.argsort(np.abs(a))[::-1][:k])
    top_b = set(np.argsort(np.abs(b))[::-1][:k])
    topk_overlap = len(top_a & top_b) / k

    # Top-k rank correlation: rank agreement on the features that matter,
    # robust to the number of noise features (full-d rank corr degrades with d).
    combined = np.abs(a) + np.abs(b)
    top_idx = np.argsort(combined)[::-1][:k]
    ck = stats.spearmanr(np.abs(a[top_idx]), np.abs(b[top_idx])).correlation
    topk_rank_corr = float(ck) if ck is not None else float("nan")

    # Per-feature *relative* gap: how much the two explainers disagree on each
    # feature's magnitude, normalized by that feature's own combined scale
    # (range ~[0, 2]; 0 = agree, 2 = one side is exactly zero while the other
    # is not). This catches magnitude-splitting that rank/sign/set agreement
    # miss — e.g. SHAP and LIME both ranking x0, x1, x2 as important but
    # disagreeing on how much of the credit x2 deserves.
    relative_gap = np.abs(a - b) / (0.5 * (np.abs(a) + np.abs(b)) + 1e-12)

    magnitude_disagreement = float(np.mean(relative_gap[top_idx]))

    return {
        "sign_disagreement": sign_dis,
        "rank_corr": rank_corr,
        "topk_rank_corr": topk_rank_corr,
        "topk_overlap": topk_overlap,
        "magnitude_disagreement": magnitude_disagreement,
        "per_feature_gap": relative_gap,
    }
