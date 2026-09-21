import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from article.scripts import generate_figures
from experiments import build_all_results as builder


ROOT = Path(__file__).parents[1]


def _copy_result_contract(tmp_path):
    paths = [
        "experiments/public_result_map.json",
        "experiments/results/synthetic/summary.json",
        "experiments/results/real/summary.json",
        "experiments/README.md",
        "experiments/benchmark_README.md",
        "article/figures/article_results.json",
        "article/figures/conversion.json",
        "article/figures/conversion_flip.png",
        "article/figures/endpoints.png",
        "article/index.qmd",
        "article/scripts/generate_figures.py",
    ]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)


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
    _copy_result_contract(tmp_path)

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


def test_article_artifacts_derive_from_one_bundle(tmp_path):
    _copy_result_contract(tmp_path)
    results = json.loads(
        (tmp_path / "article/figures/article_results.json").read_text()
    )
    assert results["schema_version"] == 1
    assert results["dataset"]["empirical_x0_x2_correlation"] == pytest.approx(
        0.85, abs=0.03
    )
    assert results["endpoints"]["profiles"]["collinear"][
        "shap_mean_absolute"
    ] == pytest.approx([0.110619, 0.074707, 0.041706, 0.004456])
    assert all(
        value >= 0
        for profile in results["endpoints"]["profiles"].values()
        for method in ("shap_mean_absolute", "lime_mean_absolute")
        for value in profile[method]
    )

    conversion = json.loads(
        (tmp_path / "article/figures/conversion.json").read_text()
    )
    assert conversion == generate_figures.conversion_payload(results)
    expected_digest = generate_figures.results_digest(results)
    for name, dimensions in (
        ("conversion_flip.png", (900, 600)),
        ("endpoints.png", (1650, 600)),
    ):
        metadata, actual_dimensions = builder._png_metadata(
            tmp_path / "article/figures" / name
        )
        assert metadata[generate_figures.PNG_SOURCE_KEY] == expected_digest
        assert actual_dimensions == dimensions


def test_article_drift_is_detected_and_repaired(tmp_path):
    _copy_result_contract(tmp_path)
    qmd = tmp_path / "article/index.qmd"
    conversion = tmp_path / "article/figures/conversion.json"
    figure = tmp_path / "article/figures/conversion_flip.png"
    qmd.write_text(qmd.read_text().replace(">0.69<", ">9.99<", 1))
    conversion.write_text(conversion.read_text().replace("0.676", "9.999", 1))
    figure.write_bytes(
        figure.read_bytes().replace(
            generate_figures.results_digest(
                json.loads(
                    (tmp_path / "article/figures/article_results.json").read_text()
                )
            ).encode(),
            b"0" * 64,
            1,
        )
    )
    snapshots = {path: path.read_bytes() for path in (qmd, conversion, figure)}

    expected = [
        "article/index.qmd",
        "article/figures/conversion.json",
        "article/figures/conversion_flip.png",
    ]
    assert builder.build(check=True, root=tmp_path) == expected
    assert {path: path.read_bytes() for path in snapshots} == snapshots
    assert builder.build(check=False, root=tmp_path) == expected
    assert builder.build(check=True, root=tmp_path) == []


def test_article_dataset_parameter_is_a_correlation():
    X, _, _ = generate_figures._make_dataset_with_rho(50_000, 0.85, seed=7)
    assert float(np.corrcoef(X[:, 0], X[:, 2])[0, 1]) == pytest.approx(
        0.85, abs=0.01
    )


def test_check_detects_stale_article_result_bundle(tmp_path):
    _copy_result_contract(tmp_path)
    producer = tmp_path / "article/scripts/generate_figures.py"
    producer.write_text(producer.read_text() + "\n# changed\n")

    assert builder.build(check=True, root=tmp_path) == [
        "article/figures/article_results.json"
    ]
    with pytest.raises(ValueError, match="declared producer"):
        builder.build(check=False, root=tmp_path)


def test_checked_in_artifacts_are_current():
    assert builder.build(check=True, root=ROOT) == []


def test_check_cli_returns_nonzero_for_drift(monkeypatch, capsys):
    monkeypatch.setattr(
        builder, "build", lambda *, check: ["experiments/README.md"]
    )

    assert builder.main(["--check"]) == 1
    assert "stale generated result artifacts" in capsys.readouterr().out
