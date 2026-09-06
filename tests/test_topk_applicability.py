"""Top-k diagnostics must distinguish inapplicability from failure."""

import json

import numpy as np
import pytest

from explaintrust import build_trust_report
from explaintrust.metrics import (
    comprehensiveness_ratio, cross_run_stability, cross_segment_stability,
    explainer_disagreement,
)


@pytest.mark.parametrize("d,k", [(1, 1), (2, 2), (2, 3), (3, 3)])
def test_full_feature_selection_has_no_comparison_evidence(d, k):
    a = np.arange(1., d + 1)

    def unexpected(_):
        pytest.fail("inapplicable removal comparison must not call the model")

    assert np.isnan(comprehensiveness_ratio(unexpected, a, a, np.zeros((4, d)), top_k=k))
    disagreement = explainer_disagreement(a, a, top_k=k)
    stability = cross_run_stability(lambda seed: a, n_runs=3, top_k=k)
    distribution = cross_segment_stability(
        np.zeros((4, d)), [0, 0, 1, 1], np.tile(a, (4, 1)), top_k=k,
    )
    assert np.isnan(disagreement["topk_overlap"])
    assert np.isnan(stability["topk_overlap"])
    assert np.isnan(distribution["topk_flip_rate"])
    assert disagreement["n_features"] == stability["n_features"] == distribution["n_features"] == d


@pytest.mark.parametrize("d", [2, 3, 10])
def test_proper_subset_can_distinguish_informative_and_reversed_rankings(d):
    x = np.ones(d)
    good = np.zeros(d)
    good[0] = 1.
    bad = good[::-1]
    model = lambda X: X[:, 0]
    background = np.zeros((4, d))
    correct = comprehensiveness_ratio(model, x, good, background, top_k=1, n_random=100)
    reversed_score = comprehensiveness_ratio(model, x, bad, background, top_k=1, n_random=100)
    assert correct > 1.
    assert reversed_score == 0.
    assert explainer_disagreement(good, bad, top_k=1)["topk_overlap"] == 0.


def report_args():
    return dict(
        removal_corr=1., comprehensiveness=2., lime_infidelity=0., sensitivity_value=0.,
        stability={"topk_rank_corr": 1., "sign_agreement": 1.},
        disagreement={"sign_disagreement": 0., "topk_rank_corr": 1.,
                      "topk_overlap": 1., "magnitude_disagreement": 0.},
        distribution={"rank_corr": 1., "topk_flip_rate": 0.},
    )


def test_top_one_rank_is_not_applicable_while_overlap_remains_descriptive():
    args = report_args()
    args["stability"]["topk_rank_corr"] = np.nan
    args["disagreement"]["topk_rank_corr"] = np.nan
    report = build_trust_report(**args, top_k=1, n_features=2)
    skipped = [m for m in report.metric_results if m.verdict == "not_applicable"]
    assert len(skipped) == 2
    assert all("rank" in m.name.lower() and np.isnan(m.value) for m in skipped)
    assert report.overall.startswith("NO ISSUES")
    assert "Not applicable" in report.overall_reason
    assert next(m for m in report.metric_results if "overlap" in m.name).verdict == "descriptive"
    payload = json.loads(report.to_json())
    assert all(m["value"] is None for m in payload["metrics"] if m["verdict"] == "not_applicable")


@pytest.mark.parametrize("d,k", [(1, 3), (2, 2), (2, 3), (3, 3)])
def test_report_does_not_score_full_feature_selection(d, k):
    args = report_args()
    # Even old, finite but tautological scores must not count as evidence.
    args["comprehensiveness"] = 1.
    report = build_trust_report(**args, top_k=k, n_features=d)
    for result in report.metric_results:
        if "comprehensiveness" in result.name or "overlap" in result.name or "flip rate" in result.name:
            assert result.verdict == "not_applicable"
            assert np.isnan(result.value)
    assert report.overall.startswith("NO ISSUES")
    assert report.context["top_k"] == min(k, d)
    assert report.context["n_features"] == d


def test_metric_metadata_carries_dimensionality_into_report():
    a = np.array([1., 2.])
    args = report_args()
    args["stability"] = cross_run_stability(lambda seed: a, n_runs=3, top_k=2)
    args["disagreement"] = explainer_disagreement(a, a, top_k=2)
    args["comprehensiveness"] = np.nan
    report = build_trust_report(**args, top_k=2)
    assert report.context["n_features"] == 2
    assert report.overall.startswith("NO ISSUES")


def test_inapplicable_checks_do_not_hide_missing_or_failed_checks():
    args = report_args()
    args.update(removal_corr=0., sensitivity_value=None)
    report = build_trust_report(**args, top_k=1, n_features=2)
    assert report.overall.startswith("INSUFFICIENT EVIDENCE")
    assert "SHAP removal-effect correlation" in report.overall_reason
    assert "Max sensitivity" in report.overall_reason
    assert "Not applicable" in report.overall_reason


def test_conflicting_dimensionality_is_rejected():
    args = report_args()
    args["disagreement"]["n_features"] = 3
    with pytest.raises(ValueError, match="n_features"):
        build_trust_report(**args, n_features=2)
