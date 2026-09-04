"""Stability: is the explanation reproducible, or an artifact of randomness?

Stochastic explainers (KernelSHAP with sampling, LIME with sampling) give a
*different* answer every time you run them with a different seed. If the answer
changes materially across runs, the explanation is not something you can act on.

For deterministic explainers (TreeSHAP) these metrics are trivially perfect,
which is itself useful signal to surface in a report.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(np.abs(a))[::-1][:k])
    top_b = set(np.argsort(np.abs(b))[::-1][:k])
    return len(top_a & top_b) / k


def cross_run_stability(
    explainer,
    n_runs: int = 10,
    top_k: int | None = None,
) -> dict:
    """Measure how much an explanation varies across ``n_runs`` random seeds.

    Parameters
    ----------
    explainer : callable
        ``explainer(seed: int) -> attribution vector`` of shape (d,). The same
        instance is re-explained with a different ``seed`` each run.
    top_k : int, optional
        If given, also report the mean Jaccard overlap of the top-k features
        (by |attribution|) across all pairs of runs.

    Returns
    -------
    dict with keys:
        "rank_corr"       — mean pairwise Spearman correlation over *all* features
                            (higher better). NOTE: this degrades with feature
                            count, because near-zero noise features shuffle ranks;
                            prefer ``topk_rank_corr`` for a dimension-robust check.
        "topk_rank_corr"  — mean pairwise Spearman correlation over the top-k
                            features only (by mean |attribution|), which is robust
                            to a large number of noise features (higher better)
        "sign_agreement"  — fraction of the top-k features whose sign is identical
                            in every run (higher better)
        "topk_overlap"    — mean pairwise Jaccard overlap of top-k sets
                            (only if top_k given)
        "std"             — per-feature std of attribution across runs (vector)
    """
    runs = np.stack([np.asarray(explainer(seed=s), dtype=float) for s in range(n_runs)])
    # runs: (n_runs, d)

    d = runs.shape[1]

    # Top-k features (by mean |attribution|) — the features that actually matter.
    k = top_k if top_k is not None else max(1, d // 3)
    k = min(k, d)
    mean_abs = np.mean(np.abs(runs), axis=0)
    top_features = np.argsort(mean_abs)[::-1][:k]

    corrs, corrs_k = [], []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            c = stats.spearmanr(runs[i], runs[j]).correlation
            if c is not None:
                corrs.append(c)
            ck = stats.spearmanr(runs[i][top_features], runs[j][top_features]).correlation
            if ck is not None:
                corrs_k.append(ck)
    rank_corr = float(np.mean(corrs)) if corrs else float("nan")
    topk_rank_corr = float(np.mean(corrs_k)) if corrs_k else float("nan")

    # Sign agreement is only meaningful over features that *matter*; near-zero
    # noise weights flapping sign would otherwise dominate the average.
    sign_stable = (
        np.all(runs[:, top_features] > 0, axis=0)
        | np.all(runs[:, top_features] < 0, axis=0)
    )
    sign_agree = float(np.mean(sign_stable)) if k > 0 else float("nan")
    per_feature_std = np.std(runs, axis=0)

    result = {
        "rank_corr": rank_corr,
        "topk_rank_corr": topk_rank_corr,
        "sign_agreement": sign_agree,
        "std": per_feature_std,
    }

    if top_k is not None:
        overlaps = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                overlaps.append(_top_k_overlap(runs[i], runs[j], top_k))
        result["topk_overlap"] = float(np.mean(overlaps)) if overlaps else float("nan")

    return result
