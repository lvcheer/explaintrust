"""Assemble metric results into a human-readable "trust report".

The report deliberately separates **measurement** (the metrics) from
**interpretation** (the verdicts). Thresholds here are sensible defaults, not
scientific claims — they are labeled as such, and every verdict carries a
plain-English reason so the user can override the thresholds with their own
domain knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    """One scored metric in the report."""

    name: str
    value: float
    direction: str  # "lower" or "higher"
    verdict: str  # "good" | "warn" | "bad" | "info"
    explanation: str


@dataclass
class TrustReport:
    """A full explanation-trust assessment."""

    metric_results: list[MetricResult] = field(default_factory=list)
    feature_reliability: pd.DataFrame | None = None
    overall: str = ""
    overall_reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def as_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "metric": m.name,
                "value": round(m.value, 4) if isinstance(m.value, (int, float)) else m.value,
                "direction": m.direction,
                "verdict": m.verdict,
                "interpretation": m.explanation,
            }
            for m in self.metric_results
        ]
        return pd.DataFrame(rows)


def _verdict(name: str, value: float, direction: str, good: float, warn: float) -> str:
    """Map a value to good/warn/bad given thresholds.

    ``good`` is the threshold at-or-beyond which we are happy; ``warn`` is the
    threshold beyond which we flag as bad. For "lower" metrics both are
    upper-bounds (<=good -> good; <=warn -> warn; else bad); for "higher"
    metrics they are lower-bounds.
    """
    if np.isnan(value):
        return "info"
    if direction == "lower":
        if value <= good:
            return "good"
        if value <= warn:
            return "warn"
        return "bad"
    # higher
    if value >= good:
        return "good"
    if value >= warn:
        return "warn"
    return "bad"


def build_trust_report(
    removal_corr: float,
    comprehensiveness: float,
    lime_infidelity: float,
    sensitivity_value: float | None,
    stability: dict,
    disagreement: dict,
    distribution: dict | None = None,
    top_k: int = 3,
) -> TrustReport:
    """Turn raw metric values into a ``TrustReport``.

    Parameters
    ----------
    removal_corr : float
        Mean removal-effect correlation for SHAP (higher better).
    comprehensiveness : float
        Mean comprehensiveness ratio for SHAP: top-k removal effect vs random
        (higher better, > 1 means top features really matter).
    lime_infidelity : float
        Mean infidelity for LIME's local linear surrogate (lower better).
    sensitivity_value : float or None
        Max-sensitivity, or None if not computed (expensive).
    stability : dict
        Output of ``metrics.cross_run_stability``.
    disagreement : dict
        Output of ``metrics.explainer_disagreement`` (aggregated over instances).
    distribution : dict or None
        Output of ``metrics.cross_segment_stability``.
    """
    results: list[MetricResult] = []

    # --- faithfulness (SHAP) ------------------------------------------------
    results.append(
        MetricResult(
            name="SHAP removal-effect correlation",
            value=removal_corr,
            direction="higher",
            verdict=_verdict("removal", removal_corr, "higher", 0.5, 0.2),
            explanation="Correlation between a feature's SHAP importance and how "
                        "much removing it actually moves the prediction. High = the "
                        "important features really matter.",
        )
    )
    results.append(
        MetricResult(
            name="SHAP comprehensiveness (top-k vs random)",
            value=comprehensiveness,
            direction="higher",
            verdict=_verdict("comprehensiveness", comprehensiveness, "higher", 1.5, 1.0),
            explanation="How much removing the top-k features moves the prediction, "
                        "relative to removing k random features. > 1 = the ranking is "
                        "not noise; ~1 = indistinguishable from random.",
        )
    )

    # --- faithfulness (LIME local surrogate) --------------------------------
    results.append(
        MetricResult(
            name="LIME local fidelity (infidelity)",
            value=lime_infidelity,
            direction="lower",
            verdict=_verdict("lime_infidelity", lime_infidelity, "lower", 0.1, 0.5),
            explanation="Gap between LIME's local linear surrogate and the model's "
                        "actual output change under perturbation. Low = the linear "
                        "approximation faithfully tracks the model locally.",
        )
    )

    # --- sensitivity --------------------------------------------------------
    if sensitivity_value is not None:
        results.append(
            MetricResult(
                name="Max sensitivity",
                value=sensitivity_value,
                direction="lower",
                verdict=_verdict("sensitivity", sensitivity_value, "lower", 0.5, 2.0),
                explanation="Worst-case change in the explanation for a tiny input "
                            "nudge. Low = the explanation is stable around this point.",
            )
        )

    # --- stability ----------------------------------------------------------
    rank_corr = stability.get("rank_corr", float("nan"))
    sign_agree = stability.get("sign_agreement", float("nan"))
    results.append(
        MetricResult(
            name="Run-to-run rank stability",
            value=rank_corr,
            direction="higher",
            verdict=_verdict("stability_rank", rank_corr, "higher", 0.9, 0.7),
            explanation="Consistency of the feature ranking across random seeds of "
                        "the explainer. High = reproducible.",
        )
    )
    results.append(
        MetricResult(
            name="Run-to-run sign stability",
            value=sign_agree,
            direction="higher",
            verdict=_verdict("stability_sign", sign_agree, "higher", 0.9, 0.7),
            explanation="Fraction of the top features whose sign is identical in "
                        "every run (noise features with ~0 weight are excluded).",
        )
    )

    # --- cross-explainer disagreement --------------------------------------
    sign_dis = disagreement.get("sign_disagreement", float("nan"))
    rank_agree = disagreement.get("rank_corr", float("nan"))
    top_overlap = disagreement.get("topk_overlap", float("nan"))
    results.append(
        MetricResult(
            name="SHAP vs LIME sign disagreement",
            value=sign_dis,
            direction="lower",
            verdict=_verdict("disagreement_sign", sign_dis, "lower", 0.2, 0.5),
            explanation="Fraction of features whose sign the two explainers "
                        "disagree on. Low = the two stories agree on direction.",
        )
    )
    results.append(
        MetricResult(
            name="SHAP vs LIME rank agreement",
            value=rank_agree,
            direction="higher",
            verdict=_verdict("disagreement_rank", rank_agree, "higher", 0.7, 0.4),
            explanation="Correlation between SHAP and LIME feature rankings.",
        )
    )
    results.append(
        MetricResult(
            name=f"SHAP vs LIME top-{top_k} overlap",
            value=top_overlap,
            direction="higher",
            verdict=_verdict("disagreement_topk", top_overlap, "higher", 0.66, 0.33),
            explanation="Overlap of the most important features named by each.",
        )
    )

    # --- distribution verification -----------------------------------------
    if distribution is not None:
        dist_corr = distribution.get("rank_corr", float("nan"))
        flip = distribution.get("topk_flip_rate", float("nan"))
        results.append(
            MetricResult(
                name="Cross-segment rank stability",
                value=dist_corr,
                direction="higher",
                verdict=_verdict("dist_rank", dist_corr, "higher", 0.7, 0.4),
                explanation="Consistency of the global feature-importance ranking "
                            "across subpopulations. High = the story generalizes.",
            )
        )
        results.append(
            MetricResult(
                name=f"Top-{top_k} flip rate across segments",
                value=flip,
                direction="lower",
                verdict=_verdict("dist_flip", flip, "lower", 0.34, 0.67),
                explanation="Fraction of subpopulations whose top features differ "
                            "from the reference. Low = stable across slices.",
            )
        )

    # --- overall verdict ----------------------------------------------------
    bad = [r for r in results if r.verdict == "bad"]
    warn = [r for r in results if r.verdict == "warn"]
    if len(bad) >= 2:
        overall = "UNRELIABLE — explanations disagree or fail faithfulness checks"
        reason = f"{len(bad)} metrics are in the red ({', '.join(r.name for r in bad)}). " \
                 "Do not make decisions from these explanations without deeper analysis."
    elif bad or len(warn) >= 2:
        overall = "MIXED — investigate before trusting"
        reason = (
            (f"Red: {', '.join(r.name for r in bad)}. " if bad else "")
            + (f"Warn: {', '.join(r.name for r in warn)}." if warn else "")
            + " The explanation has weak spots; verify the flagged features."
        )
    else:
        overall = "TRUSTWORTHY — explanations are faithful, stable, and mutually consistent"
        reason = "All metrics pass. The explanation can be used with reasonable confidence."

    return TrustReport(
        metric_results=results,
        overall=overall,
        overall_reason=reason,
    )


def per_feature_reliability(
    shap_attr: np.ndarray,
    lime_attr: np.ndarray,
    stability_std: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    instance_index: int = 0,
) -> pd.DataFrame:
    """A per-feature reliability table for one instance.

    Combines cross-explainer agreement and (optionally) run-to-run spread to
    flag features whose attribution should not be taken at face value.
    """
    shap = np.asarray(shap_attr, dtype=float)
    lime = np.asarray(lime_attr, dtype=float)
    d = shap.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(d)]

    s = shap[instance_index]
    l = lime[instance_index]
    gap = np.abs(s - l)
    scale = np.abs(s) + np.abs(l) + 1e-12
    if stability_std is not None:
        std = np.asarray(stability_std, dtype=float)
        snr = scale / (2 * std + 1e-12)
    else:
        snr = np.full(d, np.nan)

    # Flag: both explainers care about it, but disagree on magnitude/direction.
    flagged = (gap / (scale + 1e-12) > 0.5) & (scale > 1e-9)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap": s,
            "lime": l,
            "abs_gap": gap,
            "agree": ~flagged,
            "signal_to_noise": snr,
        }
    )
    return df.sort_values("abs_gap", ascending=False).reset_index(drop=True)
