"""Distribution-level verification: do explanations hold across the population?

A single explanation is a point estimate. The question that most tooling never
asks is whether the *story* the explanation tells generalizes: does the ranking
of "important features" stay stable across subpopulations, and does it survive
distribution shift?

This is the "explanation distribution verification" angle (解释分发校验): if the
global feature-importance ranking flips between segments of the data, then the
explanation is an artifact of one slice, not a property of the model.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _top_k_flip_rate(ranks_per_segment: np.ndarray, k: int) -> float:
    """Fraction of segments whose top-k set differs from the first segment's."""
    base = set(np.argsort(np.abs(ranks_per_segment[0]))[::-1][:k])
    flips = 0
    for seg in ranks_per_segment[1:]:
        if set(np.argsort(np.abs(seg))[::-1][:k]) != base:
            flips += 1
    return flips / max(1, len(ranks_per_segment) - 1)


def cross_segment_stability(
    X: np.ndarray,
    segments: np.ndarray,
    attributions: np.ndarray,
    top_k: int = 3,
) -> dict:
    """Compare global feature importance across segments of the data.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    segments : np.ndarray, shape (n,)
        Segment label for each row (e.g. cluster id, quantile bin, or a
        categorical column).
    attributions : np.ndarray, shape (n, d)
        Signed attribution for each row (from any explainer).

    Returns
    -------
    dict with keys:
        "rank_corr"     — mean pairwise Spearman correlation of per-segment
                          global importance vectors (higher better)
        "topk_flip_rate" — fraction of segments whose top-k feature set differs
                          from the reference segment's (lower better)
        "importances"   — (n_segments, d) matrix of mean |attribution| per
                          segment, and "segment_ids" list.
    """
    attributions = np.asarray(attributions, dtype=float)
    segments = np.asarray(segments)
    seg_ids = np.unique(segments)

    importances = np.zeros((len(seg_ids), attributions.shape[1]))
    for i, s in enumerate(seg_ids):
        importances[i] = np.mean(np.abs(attributions[segments == s]), axis=0)

    corrs = []
    for i in range(len(seg_ids)):
        for j in range(i + 1, len(seg_ids)):
            c = stats.spearmanr(importances[i], importances[j]).correlation
            if c is not None:
                corrs.append(c)
    rank_corr = float(np.mean(corrs)) if corrs else float("nan")

    flip_rate = _top_k_flip_rate(importances, top_k) if len(seg_ids) > 1 else 0.0

    return {
        "rank_corr": rank_corr,
        "topk_flip_rate": flip_rate,
        "importances": importances,
        "segment_ids": list(seg_ids),
    }
