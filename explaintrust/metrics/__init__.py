"""Evaluation metrics for post-hoc explanations.

Each submodule implements a family of metrics from the literature. The
convention throughout:

* lower is better for error-like metrics (infidelity, sensitivity,
  disagreement-rate, flip-rate);
* higher is better for agreement/agreement-like metrics (correlations, overlaps);
* every function takes/returns plain numpy arrays and is seed-controlled so
  results are reproducible.
"""

from .faithfulness import (
    infidelity,
    removal_effect_correlation,
    comprehensiveness_ratio,
)
from .sensitivity import max_sensitivity
from .stability import cross_run_stability
from .disagreement import explainer_disagreement
from .distribution import cross_segment_stability

__all__ = [
    "infidelity",
    "removal_effect_correlation",
    "comprehensiveness_ratio",
    "max_sensitivity",
    "cross_run_stability",
    "explainer_disagreement",
    "cross_segment_stability",
]
