"""Assemble metric results into a human-readable "trust report".

The report deliberately separates **measurement** (the metrics) from
**interpretation** (the verdicts). Thresholds here are sensible defaults, not
scientific claims — they are labeled as such, and every verdict carries a
plain-English reason so the user can override the thresholds with their own
domain knowledge.

Method disagreement and subgroup heterogeneity are descriptive diagnostics,
not quality scores. They are displayed but never determine the overall verdict;
DEFAULT_THRESHOLDS contains only the six scored checks.

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
}

_DESCRIPTIVE_KEYS = {
    "disagreement_sign", "disagreement_rank", "disagreement_topk",
    "disagreement_magnitude", "dist_rank", "dist_flip",
}

_THRESHOLD_DIRECTIONS = {
    "removal": "higher",
    "comprehensiveness": "higher",
    "lime_infidelity": "lower",
    "sensitivity": "lower",
    "stability_rank": "higher",
    "stability_sign": "higher",
}


@dataclass
class MetricResult:
    """One scored check or descriptive diagnostic in the report."""

    name: str
    value: float
    direction: str  # "lower", "higher", or "descriptive" (no quality ordering)
    verdict: str  # "good" | "warn" | "bad" | "info" | "not_applicable" | "descriptive"
    explanation: str
    role: str = "scored"  # "scored" or "descriptive"

    @property
    def included_in_overall(self) -> bool:
        return self.role == "scored" and self.verdict != "not_applicable"


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
                "role": m.role,
                "included_in_overall": m.included_in_overall,
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
                    "role": m.role,
                    "included_in_overall": m.included_in_overall,
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
    n_features: Optional[int] = None,
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
        Positive infinity denotes a failed check, as returned by ``infidelity``
        when a nonzero surrogate error has a zero-change normalization baseline.
        Other non-finite metric values are treated as unavailable.
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
        Only faithfulness, sensitivity and run-to-run stability are scored.
        Overrides for descriptive method/subgroup diagnostics are rejected.
    n_features : int or None
        Feature count used to identify comparisons that are not applicable.
        Inferred from metric dictionaries when available. Pass it explicitly
        if aggregation drops this metadata. Without a known count, NaN remains
        unavailable evidence rather than being assumed not applicable.

    Notes
    -----
    Comparisons selecting every feature and rank correlations with fewer than
    two features are marked ``not_applicable`` and excluded from scoring.
    Their values are NaN (null in JSON); they are listed in the overall reason.
    Method disagreement and subgroup heterogeneity are descriptive: their
    magnitudes, agreement, and missing values do not determine the overall
    verdict. They remain visible with an explicit role and interpretation.
    """
    if (isinstance(top_k, (bool, np.bool_))
            or not isinstance(top_k, (int, np.integer)) or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    dimensions = [n_features] if n_features is not None else []
    for metric_data in (stability, disagreement, distribution or {}):
        if "n_features" in metric_data:
            dimensions.append(metric_data["n_features"])
    for count in dimensions:
        if (isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer)) or count < 1):
            raise ValueError("n_features must be a positive integer")
    if len(set(dimensions)) > 1:
        raise ValueError("n_features conflicts with the metric dictionaries")
    n_features = int(dimensions[0]) if dimensions else None
    top_k = int(min(top_k, n_features) if n_features is not None else top_k)

    overrides = thresholds or {}
    descriptive_overrides = set(overrides) & _DESCRIPTIVE_KEYS
    if descriptive_overrides:
        raise ValueError(
            f"thresholds cannot score descriptive diagnostics: {sorted(descriptive_overrides)}"
        )
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
        if key == "lime_infidelity" and np.isposinf(value):
            return "bad"
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
    if not np.isfinite(comprehensiveness):
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
            direction="descriptive",
            verdict="descriptive" if np.isfinite(sign_dis) else "info",
            role="descriptive",
            explanation="Fraction of features whose sign the two explainers "
                        "disagree on. Larger values mean more directional differences, "
                        "not proof that either explanation is wrong.",
        )
    )
    results.append(
        MetricResult(
            name=f"SHAP vs LIME rank agreement (top-{top_k})",
            value=rank_agree,
            direction="descriptive",
            verdict="descriptive" if np.isfinite(rank_agree) else "info",
            role="descriptive",
            explanation="Correlation between SHAP and LIME rankings over the top-k "
                        "features. Higher values mean more similar rankings; agreement "
                        "does not establish correctness and differences require interpretation.",
        )
    )
    results.append(
        MetricResult(
            name=f"SHAP vs LIME top-{top_k} overlap",
            value=top_overlap,
            direction="descriptive",
            verdict="descriptive" if np.isfinite(top_overlap) else "info",
            role="descriptive",
            explanation="Overlap of the most important features named by each method. "
                        "Higher values mean more shared features, not greater explanation quality.",
        )
    )
    results.append(
        MetricResult(
            name=f"SHAP vs LIME magnitude disagreement (top-{top_k})",
            value=magnitude_dis,
            direction="descriptive",
            verdict="descriptive" if np.isfinite(magnitude_dis) else "info",
            role="descriptive",
            explanation="Mean per-feature relative |SHAP − LIME| gap over the most "
                        "important features (0 = agreement, about 2 = maximal relative gap). "
                        "This describes attribution differences; it is not a validated trust boundary.",
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
                direction="descriptive",
                verdict="descriptive" if np.isfinite(dist_corr) else "info",
                role="descriptive",
                explanation="Consistency of the global feature-importance ranking "
                            "across selected subpopulations. Lower values indicate "
                            "heterogeneity, which may reflect real differences in model behavior.",
            )
        )
        results.append(
            MetricResult(
                name=f"Top-{top_k} flip rate across segments",
                value=flip,
                direction="descriptive",
                verdict="descriptive" if np.isfinite(flip) else "info",
                role="descriptive",
                explanation="Fraction of subpopulations whose top features differ "
                            "from the reference. Differences require task-specific interpretation; "
                            "they do not by themselves imply an unreliable explanation.",
            )
        )

    inapplicable = {}
    if top_k == 1:
        rank_reason = "A rank correlation requires at least two selected features."
        inapplicable[f"Run-to-run rank stability (top-{top_k})"] = rank_reason
        inapplicable[f"SHAP vs LIME rank agreement (top-{top_k})"] = rank_reason
    if n_features is not None and top_k == n_features:
        inapplicable["SHAP comprehensiveness (top-k vs random)"] = (
            "Top-k and random-k both remove every feature; there is no comparison."
        )
        inapplicable[f"SHAP vs LIME top-{top_k} overlap"] = (
            "Both sets contain every feature, so overlap is guaranteed."
        )
        inapplicable[f"Top-{top_k} flip rate across segments"] = (
            "Every segment selects all features, so a set change is impossible."
        )
    if n_features == 1:
        inapplicable["SHAP removal-effect correlation"] = "A correlation requires at least two features."
        inapplicable["Cross-segment rank stability"] = "A rank correlation requires at least two features."
    for result in results:
        if result.name in inapplicable:
            result.value = float("nan")
            result.verdict = "not_applicable"
            result.explanation = "Not applicable: " + inapplicable[result.name]

    # Preserve the reason a non-finite value cannot be interpreted, including
    # in JSON where NaN and infinity must be serialized as null.
    for result in results:
        if np.isfinite(result.value):
            continue
        if result.verdict == "bad" and np.isposinf(result.value):
            result.explanation = (
                "Failed: normalized infidelity is infinite (+inf); the surrogate "
                "error cannot be bounded relative to the model-output change. "
                "This is a failed check, not missing evidence."
            )
        elif result.verdict == "info":
            result.explanation = (
                "Unavailable: this check was not computed or returned a non-finite "
                "value without a defined interpretation. It provides no passing evidence. "
                + result.explanation
            )

    # --- overall verdict ----------------------------------------------------
    scored = [r for r in results if r.included_in_overall]
    bad = [r for r in scored if r.verdict == "bad"]
    warn = [r for r in scored if r.verdict == "warn"]
    info = [r for r in scored if r.verdict == "info"]
    if info:
        overall = (
            "INSUFFICIENT EVIDENCE — failed checks also detected"
            if bad else "INSUFFICIENT EVIDENCE — some checks could not be computed"
        )
        reason = (
            (f"Failed: {', '.join(r.name for r in bad)}. " if bad else "")
            + (f"Warn: {', '.join(r.name for r in warn)}. " if warn else "")
            + f"Unavailable: {', '.join(r.name for r in info)}. "
            "Resolve the missing checks before drawing an overall conclusion."
        )
    elif len(bad) >= 2:
        overall = "CHECKS FAILED — multiple evaluation thresholds exceeded"
        reason = f"{len(bad)} metrics are in the red ({', '.join(r.name for r in bad)}). " \
                 "Investigate these failures under the recorded evaluation setup; thresholds are configurable diagnostics."
    elif bad or warn:
        overall = "MIXED — investigate before trusting"
        reason = (
            (f"Red: {', '.join(r.name for r in bad)}. " if bad else "")
            + (f"Warn: {', '.join(r.name for r in warn)}." if warn else "")
            + " The explanation has weak spots; verify the flagged features."
        )
    else:
        overall = "NO ISSUES DETECTED — scored checks passed"
        reason = (
            "No failure was detected by the applicable scored checks. This is supporting "
            "evidence, not a certificate that the explanation is correct."
        )

    skipped = [r.name for r in results if r.verdict == "not_applicable"]
    if skipped:
        reason += f" Not applicable (excluded from scoring): {', '.join(skipped)}."
    reason += (
        " Descriptive method/subgroup diagnostics are displayed separately and "
        "excluded from this conclusion; interpret their values in context."
    )
    missing_descriptive = [r.name for r in results if r.role == "descriptive" and r.verdict == "info"]
    if missing_descriptive:
        reason += f" Descriptive diagnostics unavailable: {', '.join(missing_descriptive)}."

    return TrustReport(
        metric_results=results,
        overall=overall,
        overall_reason=reason,
        context={
            "thresholds": active_thresholds, "top_k": top_k, "n_features": n_features,
            "decision_policy": {
                "id": "diagnostic-separation-v1",
                "scored_families": ["faithfulness", "sensitivity", "run-to-run stability"],
                "descriptive_families": ["method disagreement", "subgroup heterogeneity"],
                "missing_descriptive_affects_overall": False,
            },
        },
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
