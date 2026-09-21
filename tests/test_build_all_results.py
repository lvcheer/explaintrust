import json
import shutil
from pathlib import Path

from experiments import build_all_results as builder


ROOT = Path(__file__).parents[1]


def test_rendered_tables_use_refreshed_summaries():
    mapping = json.loads((ROOT / builder.MAP_PATH).read_text())
    targets = {target["id"]: target for target in mapping["generated_targets"]}
    synthetic = json.loads(
        (ROOT / "experiments/results/synthetic/summary.json").read_text()
    )
    real = json.loads((ROOT / "experiments/results/real/summary.json").read_text())

    synthetic_table = builder._render_synthetic(synthetic, mapping)
    assert "0.6829" in synthetic_table
    assert "0.85" in synthetic_table

    real_table = builder._render_real(real, targets["real_results_table"], mapping)
    assert "0.4497" in real_table
    assert "8.112e+09" in real_table
    assert "47/48 (1 n/c)" in real_table
    assert "descriptive" in real_table


def test_check_mode_reports_drift_without_writing(tmp_path):
    paths = [
        "experiments/public_result_map.json",
        "experiments/results/synthetic/summary.json",
        "experiments/results/real/summary.json",
        "experiments/README.md",
        "experiments/benchmark_README.md",
    ]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    assert builder.build(check=True, root=tmp_path) == []

    synthetic_target = tmp_path / "experiments/README.md"
    real_target = tmp_path / "experiments/benchmark_README.md"
    synthetic_target.write_text(
        synthetic_target.read_text().replace("0.6829", "0.9999", 1)
    )
    real_target.write_text(real_target.read_text().replace("0.4497", "0.9999", 1))
    synthetic_tampered = synthetic_target.read_text()
    real_tampered = real_target.read_text()

    expected_drift = [
        "experiments/README.md", "experiments/benchmark_README.md",
    ]
    assert builder.build(check=True, root=tmp_path) == expected_drift
    assert synthetic_target.read_text() == synthetic_tampered
    assert real_target.read_text() == real_tampered

    assert builder.build(check=False, root=tmp_path) == expected_drift
    assert builder.build(check=True, root=tmp_path) == []

    tampered = synthetic_target.read_text().replace("0.6829", "0.9999", 1)
    synthetic_target.write_text(tampered)
    assert builder.build(check=True, root=tmp_path) == ["experiments/README.md"]
    assert synthetic_target.read_text() == tampered


def test_checked_in_tables_are_current():
    assert builder.build(check=True, root=ROOT) == []


def test_check_cli_returns_nonzero_for_drift(monkeypatch, capsys):
    monkeypatch.setattr(
        builder, "build", lambda *, check: ["experiments/README.md"]
    )

    assert builder.main(["--check"]) == 1
    assert "stale generated result tables" in capsys.readouterr().out
