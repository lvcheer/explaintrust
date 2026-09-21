"""Descriptive differences are not evidence that an explanation failed."""

import json

import numpy as np
import pytest

from explaintrust import build_trust_report


def make_report(**overrides):
    args = dict(
        removal_corr=1., comprehensiveness=2., lime_infidelity=0., sensitivity_value=0.,
        stability={"topk_rank_corr": 1., "sign_agreement": 1.},
        disagreement={"sign_disagreement": 1., "topk_rank_corr": -1.,
                      "topk_overlap": 0., "magnitude_disagreement": 2.},
        distribution={"rank_corr": -1., "topk_flip_rate": 1.},
        n_features=10,
    )
    return build_trust_report(**{**args, **overrides})


def test_extreme_disagreement_and_heterogeneity_cannot_fail_overall():
    report = make_report()
    assert report.overall.startswith("NO ISSUES")
    descriptive = [m for m in report.metric_results if m.role == "descriptive"]
    assert len(descriptive) == 6
    assert all(m.verdict == "descriptive" for m in descriptive)
    assert [m.value for m in descriptive] == [1., -1., 0., 2., -1., 1.]
    assert "Descriptive" in report.overall_reason
    assert "excluded" in report.overall_reason


def test_method_differences_cannot_turn_one_failed_check_into_multiple_failures():
    report = make_report(lime_infidelity=2.)
    assert report.overall.startswith("MIXED")
    assert [m.name for m in report.metric_results if m.verdict == "bad"] == [
        "LIME local fidelity (infidelity, normalized)"]


@pytest.mark.parametrize("failures", [
    {"removal_corr": 0., "lime_infidelity": 2.},
    {"stability": {"topk_rank_corr": 0., "sign_agreement": 0.}},
    {"sensitivity_value": 3., "lime_infidelity": 2.},
])
def test_scored_failures_remain_visible(failures):
    report = make_report(**failures)
    assert report.overall.startswith("CHECKS FAILED")
    assert len([m for m in report.metric_results if m.verdict == "bad"]) == 2


def test_missing_descriptive_values_do_not_block_scored_conclusion():
    report = make_report(disagreement={}, distribution={})
    assert report.overall.startswith("NO ISSUES")
    descriptive = [m for m in report.metric_results if m.role == "descriptive"]
    assert len(descriptive) == 6
    assert all(m.verdict == "info" and np.isnan(m.value) for m in descriptive)
    assert "Descriptive diagnostics unavailable" in report.overall_reason


def test_missing_scored_check_still_blocks_overall_and_keeps_failures():
    report = make_report(sensitivity_value=None, lime_infidelity=2.)
    assert report.overall.startswith("INSUFFICIENT EVIDENCE")
    assert "LIME local fidelity" in report.overall_reason
    assert "Max sensitivity" in report.overall_reason


def test_descriptive_agreement_does_not_become_a_passing_check():
    report = make_report(disagreement={"sign_disagreement": 0., "topk_rank_corr": 1.,
                                      "topk_overlap": 1., "magnitude_disagreement": 0.},
                         distribution={"rank_corr": 1., "topk_flip_rate": 0.})
    assert all(m.verdict == "descriptive" for m in report.metric_results if m.role == "descriptive")


def test_exports_preserve_policy_roles_values_and_scoring_scope():
    report = make_report()
    frame = report.as_dataframe()
    assert len(frame) == 12
    assert frame["included_in_overall"].sum() == 6
    payload = json.loads(report.to_json())
    assert payload["context"]["decision_policy"]["id"] == "diagnostic-separation-v1"
    assert len(payload["context"]["thresholds"]) == 6
    for row in payload["metrics"]:
        assert row["included_in_overall"] == (row["role"] == "scored")
        if row["role"] == "descriptive":
            assert row["value"] is not None


def test_top_one_inapplicability_is_preserved_for_both_roles():
    report = make_report(top_k=1, n_features=2)
    skipped = [m for m in report.metric_results if m.verdict == "not_applicable"]
    assert len(skipped) == 2
    assert {m.role for m in skipped} == {"scored", "descriptive"}
    rows = report.as_dataframe()
    assert not rows.loc[rows["verdict"] == "not_applicable", "included_in_overall"].any()


@pytest.mark.parametrize("key", ["disagreement_sign", "disagreement_rank", "disagreement_topk",
                                 "disagreement_magnitude", "dist_rank", "dist_flip"])
def test_threshold_override_cannot_silently_reenable_descriptive_scoring(key):
    with pytest.raises(ValueError, match="descriptive"):
        make_report(thresholds={key: (0., 0.)})


@pytest.mark.parametrize("forced_infidelity", [None, np.inf, np.nan])
def test_benchmark_export_no_longer_grades_descriptive_metrics(
        monkeypatch, benchmark_artifact_sandbox, forced_infidelity):
    from experiments import benchmark_real_data as benchmark
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.default_rng(3).normal(size=(90, 4))
    import pandas as pd
    from experiments.real_datasets import RawDataset
    names = ["a", "b", "c", "d"]
    dataset = RawDataset(pd.DataFrame(X, columns=names), (X[:, 0] > 0).astype(int),
                         names, [], test_mask=np.arange(len(X)) >= 60)
    monkeypatch.setattr(benchmark, "load_adult", lambda: dataset)
    grouped = RawDataset(dataset.frame, dataset.y, names, [], groups=np.repeat(np.arange(30), 3))
    monkeypatch.setattr(benchmark, "load_diabetes", lambda: grouped)
    monkeypatch.setattr(benchmark, "MODELS", {
        "RF": lambda seed: RandomForestClassifier(n_estimators=5, max_depth=2, random_state=seed),
    })
    monkeypatch.setattr(benchmark, "SEEDS", range(7, 8))
    monkeypatch.setattr(benchmark, "N_EXPLAIN", 2)
    monkeypatch.setattr(benchmark, "LIME_SAMPLES", 500)
    if forced_infidelity is not None:
        monkeypatch.setattr(benchmark, "infidelity", lambda *args, **kwargs: forced_infidelity)
    sandbox = benchmark_artifact_sandbox(benchmark, n_explain=2)
    benchmark.main()
    raw = json.loads((sandbox["results"] / "raw_runs.json").read_text())
    summary_payload = json.loads((sandbox["results"] / "summary.json").read_text())
    assert raw["schema_version"] == summary_payload["schema_version"] == 1
    assert len(raw["runs"]) == raw["n_runs"] == summary_payload["n_runs"] == 2
    for run in raw["runs"]:
        assert len(run["samples"]) == 2
        positions = benchmark._sample_test_positions(run["split_context"]["test_rows"], 2, 7).tolist()
        assert run["sampling"]["test_positions"] == positions
        assert [sample["test_position"] for sample in run["samples"]] == positions
        assert run["samples"][1]["metrics"]["sensitivity"]["status"] != "not_computed"
        assert len(run["stability_runs"]) == 2
        assert run["evaluation_counts"]["stability_seeds"] == [7, 8, 9, 10, 11]
        assert run["evaluation_counts"]["lime_instance_explanations"] == 10
        for i, repeats in enumerate(run["stability_runs"]):
            assert repeats["test_position"] == run["samples"][i]["test_position"]
            assert repeats["sample_position"] == i and len(repeats["runs"]) == 5
            assert repeats["runs"][0]["contributions"] == run["samples"][i]["lime_contributions"]
            matrices = [np.array(r["contributions"]) for r in repeats["runs"]]
            stability = benchmark.cross_run_stability(lambda seed: matrices[seed], n_runs=5, top_k=3)
            assert run["samples"][i]["metrics"]["stability_rank"] == benchmark._measurement(stability["rank_corr"])
        for key in ("sensitivity", "stability_rank", "stability_rank_topk", "stability_sign"):
            assert run["metric_counts"][key]["total"] == 2
        assert len(run["feature_names"]) == len(run["samples"][0]["shap"])
        assert sum(run["subgroups"]["group_counts"]) == len(run["subgroups"]["test_positions"])
        for metric in benchmark.METRICS:
            if metric.startswith("distribution_"):
                observations = [run["subgroups"]["metrics"][metric]]
            else:
                observations = [sample["metrics"][metric] for sample in run["samples"]
                                if sample["metrics"][metric]["status"] != "not_computed"]
            decoded = [item["value"] if item["status"] == "finite" else
                       {"nan": np.nan, "positive_infinity": np.inf,
                        "negative_infinity": -np.inf}[item["status"]] for item in observations]
            assert run["metric_counts"][metric] == benchmark._counts(decoded)
            assert run["metrics"][metric] == benchmark._measurement(benchmark._mean(decoded))
    for metric in benchmark.METRICS:
        values = [run["metrics"][metric]["value"] for run in raw["runs"]
                  if run["metrics"][metric]["status"] == "finite"]
        summary = summary_payload["metrics"][metric]
        assert summary["run_counts"]["finite"] == len(values)
        assert summary["run_counts"]["total"] == 2
        assert summary["pooled_median"] == (round(float(np.median(values)), 4) if values else None)
    for run in raw["runs"]:
        assert run["split_context"]["preprocessing_fit"] == "train_only"
        if run["dataset"] == "diabetes":
            assert run["split_context"]["patient_overlap"] == 0
    for key, result in summary_payload["metrics"].items():
        if key.startswith(("disagreement_", "distribution_")):
            assert result["role"] == "descriptive"
            assert result["current_default"] is None
    assert sandbox["historical"].read_text() == sandbox["historical_text"]
    environment = json.loads((sandbox["results"] / "environment.json").read_text())
    assert environment["config"]["sha256"] == summary_payload["run_context"]["config_sha256"]
    report = (sandbox["results"] / "change_report.md").read_text()
    for heading in ("Baseline identity", "Comparison rule", "Metric comparison",
                    "Non-comparable fields", "Interpretation"):
        assert f"## {heading}" in report
