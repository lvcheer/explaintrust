"""Preprocessing and end-to-end tests for the public CSV workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.streamlit_app import run_uploaded_pipeline
from app.tabular_utils import prepare_tabular


def test_tabular_preprocessing_is_auditable():
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0],
            "category": ["a", "b", "a", "b"],
            "target": ["no", "yes", "no", "yes"],
        }
    )
    X, y, names, metadata = prepare_tabular(frame, "target", "classification")
    assert X.shape == (3, 1)
    assert names == ["numeric"]
    assert set(y) == {0, 1}
    assert metadata["dropped_non_numeric_features"] == ["category"]
    assert metadata["dropped_rows_missing"] == 1
    assert metadata["class_mapping"] == {"no": 0, "yes": 1}


@pytest.mark.parametrize("n_features", [2, 3])
def test_uploaded_regression_pipeline_runs_end_to_end(n_features):
    rng = np.random.default_rng(11)
    x0 = rng.normal(size=90)
    x1 = rng.normal(size=90)
    frame = pd.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "ignored_text": ["group"] * 90,
            "target": 2.0 * x0 - 0.5 * x1,
        }
    )
    if n_features == 3:
        frame["x2"] = rng.normal(size=90)

    result = run_uploaded_pipeline(
        frame.to_csv(index=False).encode(),
        target_col="target",
        task="regression",
        model_name="Linear Regression",
        n_estimators=10,
        max_depth=2,
        n_explain=1,
        seed=0,
        lime_samples=500,
        n_runs=3,
    )

    assert result["shap_attr"].shape == (1, n_features)
    assert result["lime_contrib"].shape == (1, n_features)
    assert result["preprocessing"]["dropped_non_numeric_features"] == ["ignored_text"]
    assert result["report"].context["output_space"] == "raw"
    assert result["report"].context["preprocessing"]["usable_rows"] == 90
    assert '"metrics"' in result["report"].to_json()
    report = result["report"]
    assert report.context["top_k"] == n_features - 1
    shap_context = report.context["shap"]
    assert shap_context["background"]["n_rows"] == report.context["background_instances"]
    assert shap_context["background"]["sha256"] == report.context["lime_reference"]["background_sha256"]
    assert shap_context["masker"] == "Independent"
    comp = next(m for m in report.metric_results if "comprehensiveness" in m.name)
    assert comp.value > 1. and comp.verdict == "good"
    skipped = [m for m in report.metric_results if m.verdict == "not_applicable"]
    assert len(skipped) == (2 if n_features == 2 else 0)
