"""Invalid sensitivity computations must never become passing evidence."""

import json

import numpy as np
import pytest

from explaintrust import build_trust_report
from explaintrust.metrics import infidelity, max_sensitivity


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("stage", ["original", "perturbed"])
def test_sensitivity_rejects_nonfinite_attributions(value, stage):
    calls = 0

    def explain(x):
        nonlocal calls
        calls += 1
        return np.full_like(x, value if stage == "original" or calls > 1 else 1.)

    with pytest.raises(ValueError, match="attribution.*finite"):
        max_sensitivity(explain, np.ones(2), np.zeros((4, 2)), n_perturbations=2)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_sensitivity_requires_positive_integer_budget(budget):
    with pytest.raises(ValueError, match="n_perturbations"):
        max_sensitivity(lambda x: x, np.ones(2), np.zeros((4, 2)), n_perturbations=budget)


@pytest.mark.parametrize("radius", [0., -0.1, np.nan, np.inf])
def test_sensitivity_requires_finite_positive_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        max_sensitivity(lambda x: x, np.ones(2), np.zeros((4, 2)), radius=radius)


@pytest.mark.parametrize("x,background", [
    (np.array([]), np.zeros((4, 0))),
    (np.ones((1, 2)), np.zeros((4, 2))),
    (np.array([np.nan, 1.]), np.zeros((4, 2))),
    (np.ones(2), np.zeros((0, 2))),
    (np.ones(2), np.zeros((4, 3))),
    (np.ones(2), np.array([[0., np.inf]])),
])
def test_sensitivity_rejects_invalid_inputs_before_explaining(x, background):
    def unexpected(_):
        pytest.fail("invalid inputs should be rejected before calling the explainer")

    with pytest.raises(ValueError):
        max_sensitivity(unexpected, x, background)


@pytest.mark.parametrize("stage", ["original", "perturbed"])
def test_sensitivity_rejects_broadcastable_wrong_attribution_shape(stage):
    calls = 0

    def explain(x):
        nonlocal calls
        calls += 1
        return np.ones(1) if stage == "original" or calls > 1 else np.ones_like(x)

    with pytest.raises(ValueError, match="attribution.*shape"):
        max_sensitivity(explain, np.ones(2), np.zeros((4, 2)), n_perturbations=2)


def good_report(**overrides):
    args = dict(
        removal_corr=1., comprehensiveness=2., lime_infidelity=0., sensitivity_value=0.,
        stability={"topk_rank_corr": 1., "sign_agreement": 1.},
        disagreement={"sign_disagreement": 0., "topk_rank_corr": 1.,
                      "topk_overlap": 1., "magnitude_disagreement": 0.},
    )
    return build_trust_report(**{**args, **overrides})


def test_infinite_infidelity_is_failure_not_missing_evidence():
    # A nonzero local surrogate contradicts a constant prediction function.
    value = infidelity(lambda X: np.zeros(len(X)), np.ones(2), np.ones(2),
                       np.array([[0., 0.], [1., 1.]]))
    assert np.isposinf(value)
    report = good_report(lime_infidelity=value)
    result = next(m for m in report.metric_results if "infidelity" in m.name)
    assert result.verdict == "bad"
    assert "infinite" in result.explanation.lower()
    assert report.overall.startswith("MIXED")
    serialized = next(m for m in json.loads(report.to_json())["metrics"] if "infidelity" in m["name"])
    assert serialized["value"] is None
    assert serialized["verdict"] == "bad"
    assert "infinite" in serialized["explanation"].lower()


def test_missing_evidence_does_not_hide_known_failures():
    report = good_report(removal_corr=0., lime_infidelity=np.inf, sensitivity_value=None)
    assert report.overall.startswith("INSUFFICIENT EVIDENCE")
    assert "failed" in report.overall.lower()
    assert "SHAP removal-effect correlation" in report.overall_reason
    assert "LIME local fidelity" in report.overall_reason
    assert "Max sensitivity" in report.overall_reason


@pytest.mark.parametrize("metric,value", [
    ("sensitivity_value", np.nan), ("sensitivity_value", np.inf),
    ("comprehensiveness", np.inf), ("comprehensiveness", -np.inf),
    ("lime_infidelity", np.nan), ("lime_infidelity", -np.inf),
])
def test_undefined_nonfinite_results_are_explicitly_unavailable(metric, value):
    report = good_report(**{metric: value})
    unavailable = [m for m in report.metric_results if m.verdict == "info"]
    assert len(unavailable) == 1
    assert "unavailable" in unavailable[0].explanation.lower()
    assert report.overall.startswith("INSUFFICIENT EVIDENCE")
    json.loads(report.to_json())
