"""Contract tests for the supported package surface and release metadata."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import explaintrust


EXPECTED_PUBLIC_API = {
    "DEFAULT_THRESHOLDS",
    "FEATURE_ROLES",
    "TrustReport",
    "__version__",
    "build_trust_report",
    "lime_attributions",
    "make_collinear_dataset",
    "per_feature_reliability",
    "prediction_output_space",
    "scalar_predictor",
    "shap_attributions",
    "shift_distribution",
    "to_contribution_scale",
}


def test_public_api_surface_is_explicit():
    assert set(explaintrust.__all__) == EXPECTED_PUBLIC_API
    for name in EXPECTED_PUBLIC_API:
        assert hasattr(explaintrust, name), name


def test_installed_and_source_versions_match():
    assert version("explaintrust") == explaintrust.__version__


def test_release_metadata_versions_match():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    citation = (root / "CITATION.cff").read_text()
    project_version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    citation_version = re.search(r"^version: ([^\s]+)$", citation, re.MULTILINE)
    assert project_version is not None
    assert citation_version is not None
    assert project_version.group(1) == citation_version.group(1) == explaintrust.__version__
