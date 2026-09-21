import copy
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def benchmark_artifact_sandbox(monkeypatch, tmp_path):
    """Redirect benchmark publication while leaving metric computation testable."""
    def configure(benchmark, *, n_explain):
        source_dir = Path(benchmark.__file__).parent
        config = json.loads((source_dir / "config.json").read_text())
        baseline = json.loads((source_dir / "benchmark_results.json").read_text())
        experiment_dir = tmp_path / "experiments"
        experiment_dir.mkdir()
        historical = experiment_dir / "benchmark_results.json"
        historical_text = json.dumps(baseline, indent=2) + "\n"
        historical.write_text(historical_text)

        real = config["real_data"]
        real["seeds"] = list(benchmark.SEEDS)
        real["models"] = {name: {} for name in benchmark.MODELS}
        real["explanations_per_dataset_model_seed"] = n_explain
        real["background_max_rows"] = benchmark.BG
        real["explainers"]["lime_samples"] = benchmark.LIME_SAMPLES
        real["historical_comparison"]["baseline_sha256"] = hashlib.sha256(
            historical.read_bytes()
        ).hexdigest()
        config_text = json.dumps(config, sort_keys=True)
        config_sha256 = hashlib.sha256(config_text.encode()).hexdigest()

        monkeypatch.setattr(benchmark, "HERE", experiment_dir)
        monkeypatch.setattr(
            benchmark, "_load_frozen_config",
            lambda: (config, config_sha256, baseline),
        )
        monkeypatch.setattr(
            benchmark, "_source_context", lambda config: {"adult": {}, "diabetes": {}},
        )
        monkeypatch.setattr(
            benchmark, "_environment_snapshot",
            lambda config_hash, sources: {
                "schema_version": 1,
                "generated_at_utc": "2026-09-21T00:00:00+00:00",
                "command": ["python", "-m", "experiments.benchmark_real_data"],
                "git": {"commit": "test-commit", "branch": "test", "dirty": False},
                "python": {"version": "test", "implementation": "test", "executable": "python"},
                "platform": "test",
                "config": {"path": "experiments/config.json", "sha256": config_hash},
                "runner": {"path": "experiments/benchmark_real_data.py", "sha256": "test"},
                "datasets": sources,
                "packages": {},
            },
        )
        return {
            "root": tmp_path,
            "results": tmp_path / "experiments" / "results" / "real",
            "historical": historical,
            "historical_text": historical_text,
            "config": copy.deepcopy(config),
        }

    return configure
