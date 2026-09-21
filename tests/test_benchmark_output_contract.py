import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import benchmark_real_data as benchmark


def frozen_config():
    return json.loads((Path(benchmark.__file__).parent / "config.json").read_text())


def test_config_drift_fails_before_dataset_loading(tmp_path):
    config = frozen_config()
    config["real_data"]["background_max_rows"] += 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="background_max_rows"):
        benchmark._load_frozen_config(path)


def test_failure_handling_drift_is_rejected(tmp_path):
    config = frozen_config()
    config["real_data"]["failure_handling"]["subgroup_shap_additivity"][
        "check_additivity"
    ] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="check_additivity"):
        benchmark._load_frozen_config(path)


def test_output_contract_cannot_target_historical_baseline(tmp_path):
    config = frozen_config()
    historical = config["real_data"]["output_contract"]["historical_baseline"]
    config["real_data"]["output_contract"]["artifacts"]["raw_runs"]["path"] = historical
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="historical baseline|experiments/results/real"):
        benchmark._load_frozen_config(path)


def test_change_equal_to_tolerance_is_not_flagged():
    config = frozen_config()["real_data"]["historical_comparison"]
    baseline = {"metrics": {
        metric: {"pooled_median": 0.5, "per_dataset_median": {"adult": 0.5, "diabetes": 0.5}}
        for metric in benchmark.METRICS
    }}
    summary = copy.deepcopy(baseline["metrics"])
    summary["removal_corr"]["pooled_median"] = 0.55
    report = benchmark._build_change_report(summary, baseline, config)
    line = next(line for line in report.splitlines()
                if line.startswith("| removal_corr | pooled |"))
    assert "| 0.0500 | 0.0500 | within tolerance |" in line


def test_frozen_sample_count_mismatch_fails_before_loading(monkeypatch):
    config = frozen_config()
    baseline = json.loads(
        (Path(benchmark.__file__).parent / "benchmark_results.json").read_text()
    )
    monkeypatch.setattr(
        benchmark, "_load_frozen_config", lambda: (config, "config-sha", baseline),
    )
    monkeypatch.setattr(
        benchmark, "load_adult",
        lambda: pytest.fail("sample-count drift must fail before loading data"),
    )
    with pytest.raises(ValueError, match="does not match frozen"):
        benchmark.main(n_explain=5)


def test_source_archive_hash_mismatch_is_rejected(monkeypatch, tmp_path):
    config = frozen_config()
    experiment_dir = tmp_path / "experiments"
    data_dir = experiment_dir / "data"
    data_dir.mkdir(parents=True)
    for archive in benchmark.DATA_ARCHIVES.values():
        (data_dir / archive).write_bytes(b"not the frozen archive")
    monkeypatch.setattr(benchmark, "HERE", experiment_dir)
    actual = hashlib.sha256(b"not the frozen archive").hexdigest()
    with pytest.raises(ValueError, match=f"archive hash mismatch: {actual}"):
        benchmark._source_context(config)


def test_subgroup_shap_additivity_failure_is_not_computed(monkeypatch):
    X = np.arange(120, dtype=float).reshape(40, 3)

    def fail_additivity(*args, **kwargs):
        raise benchmark.ShapExplainerError("Additivity check failed in TreeExplainer")

    monkeypatch.setattr(benchmark, "shap_attributions", fail_additivity)
    values, evidence = benchmark._subgroup_metrics(
        object(), X, X[:10], ["a", "b", "c"], seed=3,
    )

    assert all(np.isnan(value) for value in values.values())
    assert len(evidence["test_positions"]) == len(X)
    assert evidence["mean_absolute_attributions"] is None
    assert benchmark._counts(list(evidence["metrics"].values())) == {
        "total": 2,
        "finite": 0,
        "nan": 0,
        "positive_infinity": 0,
        "negative_infinity": 0,
        "not_computed": 2,
    }
    for metric in benchmark.SUBGROUP_METRICS:
        measurement = evidence["metrics"][metric]
        assert measurement["value"] is None
        assert measurement["status"] == "not_computed"
        assert measurement["error"]["sampled_rows"] == len(X)
        assert measurement["error"]["test_positions_field"] == "subgroups.test_positions"
    json.dumps(benchmark._json_safe(evidence), allow_nan=False)


def test_non_shap_subgroup_failure_is_not_suppressed(monkeypatch):
    X = np.arange(120, dtype=float).reshape(40, 3)

    def fail_unexpectedly(*args, **kwargs):
        raise ValueError("unexpected failure")

    monkeypatch.setattr(benchmark, "shap_attributions", fail_unexpectedly)
    with pytest.raises(ValueError, match="unexpected failure"):
        benchmark._subgroup_metrics(object(), X, X[:10], ["a", "b", "c"], seed=3)
