"""explaintrust — evaluate the trustworthiness of post-hoc explanations.

This package answers a question that most XAI tooling ignores:

    "Okay, SHAP/LIME gave me a feature attribution — but can I *trust* it?"

It implements evaluation metrics (faithfulness, sensitivity, stability,
cross-explainer disagreement, and distribution-level verification) and turns
them into a human-readable "trust report".

The design principle: **the shell is thin, the kernel is defensible.** Every
metric is implemented from published definitions and documented with its
assumptions and limitations, so the numbers you get can stand up to scrutiny.

Reference definitions:
  * Infidelity / sensitivity — Yeh et al., "On the (In)fidelity and
    Sensitivity of Explanations", NeurIPS 2019.
  * Explanations disagreeing — Krishna et al., "The Disagreement Problem in
    Explainable Machine Learning", CACM 2024.
"""

from .explainers import (
    scalar_predictor,
    shap_attributions,
    lime_attributions,
    to_contribution_scale,
)
from .report import TrustReport, build_trust_report, per_feature_reliability
from .data import make_collinear_dataset, shift_distribution, FEATURE_ROLES

__version__ = "0.1.0"

__all__ = [
    "scalar_predictor",
    "shap_attributions",
    "lime_attributions",
    "to_contribution_scale",
    "TrustReport",
    "build_trust_report",
    "per_feature_reliability",
    "make_collinear_dataset",
    "shift_distribution",
    "FEATURE_ROLES",
    "__version__",
]
