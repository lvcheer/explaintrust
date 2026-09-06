"""Keep repeated LIME explanations tied to their instance and configuration."""

import json

import numpy as np
import pytest

from app import streamlit_app as app
from explaintrust import lime_attributions, to_contribution_scale
from sklearn.ensemble import RandomForestRegressor


def slopes(X, seed):
    base = np.array([4., 3., 2., 1.])
    change = np.array([1., -1., 0.5, -0.5])
    return base + (seed - 17) * X[:, :1] * change


@pytest.fixture
def battery(monkeypatch):
    calls = []
    X = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    background = np.random.default_rng(3).normal(size=(30, 4))

    class Model:
        def predict(self, data):
            return data @ np.array([4., 3., 2., 1.])

    def lime(model, data, X_background, **kwargs):
        calls.append((data.copy(), X_background.copy(), kwargs.copy()))
        return slopes(data, kwargs["seed"])

    monkeypatch.setattr(app, "lime_attributions", lime)
    def shap_values(model, data, **kwargs):
        values = (data - background.mean(axis=0)) * np.array([4., 3., 2., 1.])
        if kwargs.get("return_context"):
            return values, {"background": {"sha256": "test-reference"}}
        return values

    monkeypatch.setattr(app, "shap_attributions", shap_values)
    out = app._run_battery(
        Model(), X, background, ["a", "b", "c", "d"], len(X),
        lime_samples=1500, n_runs=3, seed=17, X_dist=background, seg_feature=0,
    )
    return out, calls, X, background


def test_stability_uses_main_lime_budget_batch_and_seed_schedule(battery):
    out, calls, X, background = battery
    assert len(calls) == 3  # Main explanation is one of the three runs.
    assert [kwargs["seed"] for _, _, kwargs in calls] == [17, 18, 19]
    for data, bg, kwargs in calls:
        np.testing.assert_array_equal(data, X)
        np.testing.assert_array_equal(bg, background)
        assert kwargs["num_samples"] == 1500
        assert kwargs["feature_names"] == ["a", "b", "c", "d"]
        assert kwargs["output_space"] == "raw"
    context = out["report"].context
    assert context["stability_instances"] == 2
    assert context["stability_config"]["num_samples"] == 1500
    assert context["stability_config"]["seeds"] == [17, 18, 19]
    assert context["stability_config"]["instance_indices"] == [0, 1]
    assert context["stability_config"]["n_runs"] == 3


def test_each_instance_has_its_own_contribution_scale_spread(battery):
    out, _, X, background = battery
    repeats = np.stack([slopes(X, seed) * (X - background.mean(axis=0)) for seed in [17, 18, 19]])
    expected = repeats.std(axis=0)
    assert not np.allclose(expected[0], expected[1])
    for i, result in enumerate(out["stability_by_instance"]):
        np.testing.assert_allclose(result["std"], expected[i])
    # The report's aggregate describes both instances, not just row zero.
    for key in ("rank_corr", "topk_rank_corr", "sign_agreement", "topk_overlap"):
        assert out["stability"][key] == pytest.approx(
            np.mean([row[key] for row in out["stability_by_instance"]])
        )
    assert "std" not in out["stability"]  # Never present a population average as local spread.


def test_selecting_instance_updates_table_and_export_together(battery):
    out, _, X, background = battery
    # Exercise selection in both directions to detect stale report state.
    for i in [0, 1, 0]:
        table = app._update_feature_reliability(out, i).set_index("feature")
        repeats = np.stack([slopes(X, seed) * (X - background.mean(axis=0)) for seed in [17, 18, 19]])
        scale = np.abs(out["shap_attr"][i]) + np.abs(out["lime_contrib"][i]) + 1e-12
        expected = scale / (2 * repeats[:, i].std(axis=0) + 1e-12)
        np.testing.assert_allclose(table.loc[out["names"], "signal_to_noise"], expected)
        payload = json.loads(out["report"].to_json())
        assert payload["context"]["feature_reliability_instance"] == i
        exported = {row["feature"]: row for row in payload["feature_reliability"]}
        for j, name in enumerate(out["names"]):
            assert exported[name]["signal_to_noise"] == pytest.approx(expected[j])


def test_real_lime_repeated_batches_match_each_instances_saved_spread():
    X = np.random.default_rng(11).normal(size=(60, 4))
    model = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X, 2 * X[:, 0] + X[:, 1] ** 2)
    names = ["a", "b", "c", "d"]
    background, selected = X[10:40], X[:3]
    out = app._run_battery(
        model, selected, background, names, 3, lime_samples=500, n_runs=3,
        seed=4, X_dist=X, seg_feature=0,
    )
    expected_runs = np.stack([
        to_contribution_scale(
            lime_attributions(model, selected, background, feature_names=names,
                              num_samples=500, seed=seed, output_space="raw"),
            selected, background,
        ) for seed in [4, 5, 6]
    ])
    np.testing.assert_allclose(out["lime_contrib"], expected_runs[0])
    for i in range(3):
        np.testing.assert_allclose(out["stability_by_instance"][i]["std"], expected_runs[:, i].std(axis=0))
    assert out["report"].context["stability_instances"] == 3
