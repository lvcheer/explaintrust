import copy
import hashlib
import json
from pathlib import Path

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
