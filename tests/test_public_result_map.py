import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
MAP_PATH = ROOT / "experiments/public_result_map.json"


def test_public_result_map_references_are_closed_and_unique():
    mapping = json.loads(MAP_PATH.read_text())
    assert mapping["schema_version"] == 1
    assert mapping["entry_point"] == "python -m experiments.build_all_results"
    assert mapping["modes"] == ["write", "check"]
    assert mapping["implementation_status"] == "partial"

    sources = mapping["sources"]
    source_ids = [source["id"] for source in sources]
    assert len(source_ids) == len(set(source_ids))
    known_sources = set(source_ids)
    for source in sources:
        path = Path(source["path"])
        assert not path.is_absolute() and ".." not in path.parts

    targets = mapping["generated_targets"]
    target_ids = [target["id"] for target in targets]
    assert len(target_ids) == len(set(target_ids))
    for target in targets:
        assert set(target["source_ids"]) <= known_sources
        assert (ROOT / target["target"]).exists()

    manual_ids = [claim["id"] for claim in mapping["checked_manual_claims"]]
    assert len(manual_ids) == len(set(manual_ids))
    for claim in mapping["checked_manual_claims"]:
        assert set(claim["source_ids"]) <= known_sources
        assert all((ROOT / target).exists() for target in claim["targets"])


def test_only_planned_sources_may_be_absent():
    mapping = json.loads(MAP_PATH.read_text())
    for source in mapping["sources"]:
        path = ROOT / source["path"]
        exists = path.exists()
        assert exists or source["status"] == "planned"
        if exists and "schema_version" in source:
            assert json.loads(path.read_text())["schema_version"] == source["schema_version"]


def test_generated_markdown_markers_are_paired_and_unique():
    mapping = json.loads(MAP_PATH.read_text())
    markers = []
    for target in mapping["generated_targets"]:
        if target["mode"] != "markdown_block":
            continue
        assert target["start_marker"].startswith("<!-- BEGIN AUTO:")
        assert target["end_marker"].startswith("<!-- END AUTO:")
        markers.extend([target["start_marker"], target["end_marker"]])
    assert len(markers) == len(set(markers))

    article_target = next(
        target for target in mapping["generated_targets"]
        if target["mode"] == "templated_claims"
    )
    article = (ROOT / article_target["target"]).read_text()
    for claim_id in article_target["claim_ids"]:
        assert article.count(f"<!-- BEGIN AUTO:{claim_id} -->") == 1
        assert article.count(f"<!-- END AUTO:{claim_id} -->") == 1


def test_historical_baselines_are_excluded_from_generation():
    mapping = json.loads(MAP_PATH.read_text())
    exclusions = set(mapping["explicit_exclusions"])
    assert "experiments/calibration.json" in exclusions
    assert "experiments/benchmark_results.json" in exclusions

    generated_sources = {
        source_id
        for target in mapping["generated_targets"]
        for source_id in target["source_ids"]
    }
    source_paths = {source["id"]: source["path"] for source in mapping["sources"]}
    assert not {source_paths[source_id] for source_id in generated_sources} & exclusions


def test_workflow_integrations_are_unique_and_target_existing_files():
    mapping = json.loads(MAP_PATH.read_text())
    integrations = mapping["workflow_integrations"]
    ids = [integration["id"] for integration in integrations]
    assert len(ids) == len(set(ids))
    assert {integration["id"] for integration in integrations} == {
        "ci_consistency_check", "release_refresh", "contributor_guidance",
    }
    for integration in integrations:
        assert (ROOT / integration["target"]).is_file()
        assert "experiments.build_all_results" in integration["planned_command"]
