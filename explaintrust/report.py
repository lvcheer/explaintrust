"""Assemble metric results into a human-readable "trust report".

The report deliberately separates **measurement** (the metrics) from
**interpretation** (the verdicts). Thresholds here are sensible defaults, not
scientific claims — they are labeled as such, and every verdict carries a
plain-English reason so the user can override the thresholds with their own
domain knowledge.

A first-pass calibration study lives in ``experiments/`` (see
``experiments/calibrate_thresholds.py`` and its README). On held-out synthetic
seeds, removal-effect correlation, infidelity, and max-sensitivity separate
their engineered regimes; several other diagnostics do not. Absolute threshold
values are model- and regime-specific, so the numbers below remain documented
defaults rather than auto-fitted constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "removal": (0.5, 0.2),
    "comprehensiveness": (1.0, 1.0),
    "lime_infidelity": (0.5, 1.0),
    "sensitivity": (0.5, 2.0),
    "stability_rank": (0.9, 0.7),
    "stability_sign": (0.9, 0.7),
    "disagreement_sign": (0.2, 0.5),
    "disagreement_rank": (0.7, 0.4),
    "disagreement_topk": (0.66, 0.33),
    "disagreement_magnitude": (1.0, 1.5),
    "dist_rank": (0.7, 0.4),
    "dist_flip": (0.34, 0.67),
}

_THRESHOLD_DIRECTIONS = {
    "removal": "higher",
    "comprehensiveness": "higher",
    "lime_infidelity": "lower",
    "sensitivity": "lower",
    "stability_rank": "higher",
    "stability_sign": "higher",
    "disagreement_sign": "lower",
    "disagreement_rank": "higher",
    "disagreement_topk": "higher",
    "disagreement_magnitude": "lower",
    "dist_rank": "higher",
    "dist_flip": "lower",
}


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
    feature_reliability: Optional[pd.DataFrame] = None
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation including method context."""
        feature_rows = None
        if self.feature_reliability is not None:
            feature_rows = json.loads(self.feature_reliability.to_json(orient="records"))
        return {
            "overall": self.overall,
            "overall_reason": self.overall_reason,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value if np.isfinite(m.value) else None,
                    "direction": m.direction,
                    "verdict": m.verdict,
                    "explanation": m.explanation,
                }
                for m in self.metric_results
            ],
            "feature_reliability": feature_rows,
            "context": self.context,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report for archiving and reproducibility."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def _verdict(name: str, value: float, direction: str, good: float, warn: float) -> str:
    """Map a value to good/warn/bad given thresholds.

    ``good`` is the threshold at-or-beyond which we are happy; ``warn`` is the
    threshold beyond which we flag as bad. For "lower" metrics both are
    upper-bounds (<=good -> good; <=warn -> warn; else bad); for "higher"
    metrics they are lower-bounds.
    """
    if not np.isfinite(value):
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
    sensitivity_value: Optional[float],
    stability: dict,
    disagreement: dict,
    distribution: Optional[dict] = None,
    top_k: int = 3,
    thresholds: Optional[dict[str, tuple[float, float]]] = None,
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
    thresholds : dict or None
        Optional overrides for entries in ``DEFAULT_THRESHOLDS``. Each value is
        a ``(good, warn)`` pair in the metric's declared direction.
    """
    overrides = thresholds or {}
    unknown = set(overrides) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
    active_thresholds = {**DEFAULT_THRESHOLDS, **overrides}
    for key, pair in active_thresholds.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"threshold {key!r} must be a (good, warn) pair")
        good, warn = pair
        try:
            finite = np.isfinite(good) and np.isfinite(warn)
        except TypeError as exc:
            raise ValueError(f"threshold {key!r} must contain finite numbers") from exc
        if not finite:
            raise ValueError(f"threshold {key!r} must contain finite numbers")
        direction = _THRESHOLD_DIRECTIONS[key]
        if (direction == "higher" and good < warn) or (
            direction == "lower" and good > warn
        ):
            raise ValueError(
                f"threshold {key!r} has invalid order for a {direction}-is-better metric"
            )

    def score(key: str, value: float, direction: str) -> str:
        good, warn = active_thresholds[key]
        return _verdict(key, value, direction, good, warn)

    results: list[MetricResult] = []

    # --- faithfulness (SHAP) ------------------------------------------------
    results.append(
        MetricResult(
            name="SHAP removal-effect correlation",
            value=removal_corr,
            direction="higher",
            verdict=score("removal", removal_corr, "higher"),
            explanation="Correlation between a feature's SHAP importance and how "
                        "much removing it actually moves the prediction. High = the "
                        "important features really matter.",
        )
    )
    # Comprehensiveness is a qualitative "not noise" gate, not a graded score:
    # its absolute size saturates and is not comparable across datasets, but
    # "> 1" (top-k removal beats random removal) is a robust yes/no signal.
    if np.isnan(comprehensiveness):
        comp_verdict = "info"
    else:
        comp_verdict = (
            "good"
            if comprehensiveness > active_thresholds["comprehensiveness"][0]
            else "bad"
        )
    results.append(
        MetricResult(
            name="SHAP comprehensiveness (top-k vs random)",
            value=comprehensiveness,
            direction="higher",
            verdict=comp_verdict,
            explanation="A 'not noise' gate: removing the top-k features moves the "
                        "prediction more than removing k random ones. > 1 = the ranking "
                        "is informative; ≤ 1 = indistinguishable from random. (The ratio's "
                        "absolute size is not comparable across datasets.)",
        )
    )

    # --- faithfulness (LIME local surrogate) --------------------------------
    results.append(
        MetricResult(
            name="LIME local fidelity (infidelity, normalized)",
            value=lime_infidelity,
            direction="lower",
            verdict=score("lime_infidelity", lime_infidelity, "lower"),
            explanation="Normalized gap between LIME's local linear surrogate and the "
                            "model's actual output change (relative to always predicting zero change). "
                        "< 0.5 = the surrogate explains most of the change; ~1 = no better "
                        "than predicting zero change; > 1 = worse than nothing.",
        )
    )

    # --- sensitivity --------------------------------------------------------
    if sensitivity_value is not None:
        results.append(
            MetricResult(
                name="Max sensitivity",
                value=sensitivity_value,
                direction="lower",
                verdict=score("sensitivity", sensitivity_value, "lower"),
                explanation="Worst-case change in the explanation for a tiny input "
                            "nudge. Low = the explanation is stable around this point.",
            )
        )
    else:
        results.append(
            MetricResult(
                name="Max sensitivity",
                value=float("nan"),
                direction="lower",
                verdict="info",
                explanation="Not computed. Run a local perturbation check before drawing "
                            "an overall conclusion.",
            )
        )

    # --- stability ----------------------------------------------------------
    # Rank stability is measured over the top-k features only: a full-d Spearman
    # correlation degrades with the feature count (noise features shuffle ranks)
    # and would flag every high-dimensional explanation as unstable.
    rank_corr = stability.get("topk_rank_corr", stability.get("rank_corr", float("nan")))
    sign_agree = stability.get("sign_agreement", float("nan"))
    results.append(
        MetricResult(
            name=f"Run-to-run rank stability (top-{top_k})",
            value=rank_corr,
            direction="higher",
            verdict=score("stability_rank", rank_corr, "higher"),
            explanation="Consistency of the top-k feature ranking across random "
                        "seeds of the explainer. High = reproducible.",
        )
    )
    results.append(
        MetricResult(
            name="Run-to-run sign stability",
            value=sign_agree,
            direction="higher",
            verdict=score("stability_sign", sign_agree, "higher"),
            explanation="Fraction of the top features whose sign is identical in "
                        "every run (noise features with ~0 weight are excluded).",
        )
    )

    # --- cross-explainer disagreement --------------------------------------
    sign_dis = disagreement.get("sign_disagreement", float("nan"))
    rank_agree = disagreement.get("topk_rank_corr", disagreement.get("rank_corr", float("nan")))
    top_overlap = disagreement.get("topk_overlap", float("nan"))
    magnitude_dis = disagreement.get("magnitude_disagreement", float("nan"))
    results.append(
        MetricResult(
            name="SHAP vs LIME sign disagreement",
            value=sign_dis,
            direction="lower",
            verdict=score("disagreement_sign", sign_dis, "lower"),
            explanation="Fraction of features whose sign the two explainers "
                        "disagree on. Low = the two stories agree on direction.",
        )
    )
    results.append(
        MetricResult(
            name=f"SHAP vs LIME rank agreement (top-{top_k})",
            value=rank_agree,
            direction="higher",
            verdict=score("disagreement_rank", rank_agree, "higher"),
            explanation="Correlation between SHAP and LIME rankings over the top-k "
                        "features (robust to the number of noise features).",
        )
    )
    results.append(
        MetricResult(
            name=f"SHAP vs LIME top-{top_k} overlap",
            value=top_overlap,
            direction="higher",
            verdict=score("disagreement_topk", top_overlap, "higher"),
            explanation="Overlap of the most important features named by each.",
        )
    )
    results.append(
        MetricResult(
            name=f"SHAP vs LIME magnitude disagreement (top-{top_k})",
            value=magnitude_dis,
            direction="lower",
            verdict=score("disagreement_magnitude", magnitude_dis, "lower"),
            explanation="Mean per-feature relative |SHAP − LIME| gap over the most "
                        "important features (0 = agree, 2 = opposite). Catches how much "
                        "the two explainers disagree on the *size* of each important "
                        "feature's effect — something rank/sign/overlap agreement miss.",
        )
    )

    # --- subgroup consistency ----------------------------------------------
    if distribution is not None:
        dist_corr = distribution.get("rank_corr", float("nan"))
        flip = distribution.get("topk_flip_rate", float("nan"))
        results.append(
            MetricResult(
                name="Cross-segment rank stability",
                value=dist_corr,
                direction="higher",
                verdict=score("dist_rank", dist_corr, "higher"),
                explanation="Consistency of the global feature-importance ranking "
                            "across selected subpopulations. High = little detected "
                            "subgroup heterogeneity.",
            )
        )
        results.append(
            MetricResult(
                name=f"Top-{top_k} flip rate across segments",
                value=flip,
                direction="lower",
                verdict=score("dist_flip", flip, "lower"),
                explanation="Fraction of subpopulations whose top features differ "
                            "from the reference. Low = stable across slices.",
            )
        )

    # --- overall verdict ----------------------------------------------------
    bad = [r for r in results if r.verdict == "bad"]
    warn = [r for r in results if r.verdict == "warn"]
    info = [r for r in results if r.verdict == "info"]
    if info:
        overall = "INSUFFICIENT EVIDENCE — some checks could not be computed"
        reason = (
            f"Unavailable: {', '.join(r.name for r in info)}. "
            "Resolve the missing checks before drawing an overall conclusion."
        )
    elif len(bad) >= 2:
        overall = "UNRELIABLE — explanations disagree or fail faithfulness checks"
        reason = f"{len(bad)} metrics are in the red ({', '.join(r.name for r in bad)}). " \
                 "Do not make decisions from these explanations without deeper analysis."
    elif bad or warn:
        overall = "MIXED — investigate before trusting"
        reason = (
            (f"Red: {', '.join(r.name for r in bad)}. " if bad else "")
            + (f"Warn: {', '.join(r.name for r in warn)}." if warn else "")
            + " The explanation has weak spots; verify the flagged features."
        )
    else:
        overall = "NO ISSUES DETECTED — configured checks passed"
        reason = (
            "No failure was detected by the configured checks. This is supporting "
            "evidence, not a certificate that the explanation is correct."
        )

    return TrustReport(
        metric_results=results,
        overall=overall,
        overall_reason=reason,
        context={"thresholds": active_thresholds, "top_k": top_k},
    )


def per_feature_reliability(
    shap_attr: np.ndarray,
    lime_attr: np.ndarray,
    stability_std: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
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
