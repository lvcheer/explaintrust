import json

import numpy as np
import pytest

from experiments import calibrate_thresholds as calibration


def test_result_bundle_groups_raw_observations_by_seed_and_condition(monkeypatch):
    monkeypatch.setattr(calibration, "SEEDS", range(2))
    summaries, runs = calibration._result_bundle(
        "example",
        "ExampleModel",
        [("metric", "higher", [0.1, 0.2, 0.3, 0.4], [0.5, np.nan, 0.7, np.inf])],
        2,
        nominal_condition="unaltered",
        stress_condition="corrupted",
        observation_unit="test_instance",
    )

    assert len(summaries) == 1
    assert len(runs) == 4
    assert [(run["seed"], run["condition_role"]) for run in runs] == [
        (0, "nominal_good"),
        (0, "stress"),
        (1, "nominal_good"),
        (1, "stress"),
    ]
    assert [
        item["status"] for item in runs[1]["metrics"]["metric"]["observations"]
    ] == ["finite", "nan"]
    assert [
        item["observation_index"]
        for item in runs[1]["metrics"]["metric"]["observations"]
    ] == [0, 1]
    assert [
        item["status"] for item in runs[3]["metrics"]["metric"]["observations"]
    ] == ["finite", "positive_infinity"]
    assert runs[0]["observation_count"] == 2


def test_frozen_config_matches_runner_and_detects_drift(tmp_path):
    config, digest = calibration._load_frozen_config()
    assert len(digest) == 64
    assert config["synthetic"]["top_k"] == calibration.TOPK

    config["synthetic"]["top_k"] += 1
    changed = tmp_path / "config.json"
    changed.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="synthetic.top_k"):
        calibration._load_frozen_config(changed)


def test_write_json_is_strict_and_ends_with_newline(tmp_path):
    output = tmp_path / "nested" / "result.json"
    calibration._write_json(output, {"value": {"value": None, "status": "nan"}})
    assert output.read_text().endswith("\n")
    assert json.loads(output.read_text()) == {
        "value": {"value": None, "status": "nan"}
    }

    with pytest.raises(ValueError, match="Out of range float values"):
        calibration._write_json(output, {"value": float("nan")})
