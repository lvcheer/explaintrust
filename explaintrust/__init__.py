"""explaintrust — evaluate the trustworthiness of post-hoc explanations.

This package answers a question that most XAI tooling ignores:

    "Okay, SHAP/LIME gave me a feature attribution — but can I *trust* it?"

It implements evaluation metrics (faithfulness, sensitivity, stability,
cross-explainer disagreement, and subgroup consistency) and turns
them into a human-readable "trust report".

The design principle: **the shell is thin, the kernel is defensible.** Metrics
either follow published definitions or are labeled as project-specific
diagnostics, with their assumptions and limitations documented.

Reference definitions:
  * Infidelity / sensitivity — Yeh et al., "On the (In)fidelity and
    Sensitivity of Explanations", NeurIPS 2019.
  * Explanations disagreeing — Krishna et al., "The Disagreement Problem in
    Explainable Machine Learning", CACM 2024.
"""

from .explainers import (
    scalar_predictor,
    prediction_output_space,
    shap_attributions,
    lime_attributions,
    to_contribution_scale,
)
from .report import DEFAULT_THRESHOLDS, TrustReport, build_trust_report, per_feature_reliability
from .data import make_collinear_dataset, shift_distribution, FEATURE_ROLES

__version__ = "0.1.0"

__all__ = [
    "scalar_predictor",
    "prediction_output_space",
    "shap_attributions",
    "lime_attributions",
    "to_contribution_scale",
    "TrustReport",
    "DEFAULT_THRESHOLDS",
    "build_trust_report",
    "per_feature_reliability",
    "make_collinear_dataset",
    "shift_distribution",
    "FEATURE_ROLES",
    "__version__",
]
