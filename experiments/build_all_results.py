"""Generate or check public result artifacts from canonical summaries.

The implemented targets in ``experiments/public_result_map.json`` currently
cover both benchmark tables and the article's numerical claims, OJS data, and
static figures. Checked manual claims remain planned follow-up work.

Run from the repository root:

    python -m experiments.build_all_results
    python -m experiments.build_all_results --check
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Optional

from explaintrust.report import DEFAULT_THRESHOLDS, _THRESHOLD_DIRECTIONS


ROOT = Path(__file__).parents[1]
MAP_PATH = "experiments/public_result_map.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _stale_produced_sources(mapping: dict, root: Path) -> list[str]:
    stale = []
    for source in mapping["sources"]:
        producer = source.get("producer")
        if not producer:
            continue
        result = _load_json(root / source["path"])
        expected = result.get("generator_sha256")
        producer_path = root / producer
        actual = (
            hashlib.sha256(producer_path.read_bytes()).hexdigest()
            if producer_path.exists()
            else None
        )
        if expected != actual:
            stale.append(source["path"])
    return stale


def _format_number(value) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) >= 1e6:
        return f"{value:.3e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def row(values):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        return "| " + " | ".join(cells) + " |"

    return "\n".join([row(headers), row(["---"] * len(headers)), *(row(item) for item in rows)])


def _render_synthetic(summary: dict, mapping: dict) -> str:
    directions = mapping["formatting"]["directions"]
    rows = []
    for metric in summary["metrics"]:
        rows.append([
            metric["name"],
            directions[metric["direction"]],
            _format_number(metric["good_median"]),
            _format_number(metric["bad_median"]),
            _format_number(metric["good_threshold"]),
            _format_number(metric["warn_threshold"]),
            f"{metric['good_pass_rate']:.2f}",
            f"{metric['bad_flag_rate']:.2f}",
        ])
    return _markdown_table(
        ["Metric", "dir", "med(good)", "med(stress)", "good", "warn", "pass", "flag"],
        rows,
    )


def _validate_real_policy(summary: dict, target: dict) -> None:
    policy_keys = target["policy_keys"]
    for metric, values in summary["metrics"].items():
        if values["role"] == "descriptive":
            if values["current_default"] is not None:
                raise ValueError(f"descriptive metric {metric} must not have a default")
            continue
        policy_key = policy_keys[metric]
        if list(DEFAULT_THRESHOLDS[policy_key]) != values["current_default"]:
            raise ValueError(f"real summary default for {metric} differs from report policy")
        if _THRESHOLD_DIRECTIONS[policy_key] != values["direction"]:
            raise ValueError(f"real summary direction for {metric} differs from report policy")


def _verdict(values: dict) -> str:
    if values["role"] == "descriptive":
        return "descriptive"
    median = values["pooled_median"]
    if median is None or not math.isfinite(float(median)):
        return "unavailable"
    good, warn = values["current_default"]
    if values["direction"] == "higher":
        verdict = "good" if median >= good else "warn" if median >= warn else "bad"
    else:
        verdict = "good" if median <= good else "warn" if median <= warn else "bad"
    if values["run_counts"]["positive_infinity"]:
        return "infinite: inspect"
    return verdict


def _render_real(summary: dict, target: dict, mapping: dict) -> str:
    _validate_real_policy(summary, target)
    directions = mapping["formatting"]["directions"]
    rows = []
    for metric, values in summary["metrics"].items():
        counts = values["run_counts"]
        finite = f"{counts['finite']}/{counts['total']}"
        if counts["not_computed"]:
            finite += f" ({counts['not_computed']} n/c)"
        default = "not scored"
        if values["current_default"] is not None:
            default = " / ".join(_format_number(value) for value in values["current_default"])
        rows.append([
            target["row_labels"][metric],
            directions[values["direction"]],
            _format_number(values["per_dataset_median"]["adult"]),
            _format_number(values["per_dataset_median"]["diabetes"]),
            _format_number(values["pooled_median"]),
            _format_number(values["pooled_p10"]),
            _format_number(values["pooled_p90"]),
            finite,
            default,
            _verdict(values),
        ])
    table = _markdown_table(
        [
            "Metric", "dir", "Adult", "Diabetes", "pooled", "P10", "P90",
            "finite runs", "default (good/warn)", "verdict@median",
        ],
        rows,
    )
    return table + "\n\n`n/c` means an explicitly not-computed run; descriptive metrics are not scored."


RENDERERS = {
    "synthetic_results_table": _render_synthetic,
    "real_results_table": _render_real,
}


def _replace_block(text: str, target: dict, rendered: str) -> str:
    start = target["start_marker"]
    end = target["end_marker"]
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"{target['target']} must contain exactly one marker pair")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + start + "\n" + rendered.rstrip() + "\n" + end + after


def _nested_value(values: dict, dotted_path: str):
    current = values
    for part in dotted_path.split("."):
        current = current[part]
    return current


def _render_article_claims(text: str, results: dict, target: dict) -> str:
    for claim_id in target["claim_ids"]:
        start = f"<!-- BEGIN AUTO:{claim_id} -->"
        end = f"<!-- END AUTO:{claim_id} -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"{target['target']} must contain one marker pair for {claim_id}")
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        value = f"{float(_nested_value(results, claim_id)):.2f}"
        text = before + start + value + end + after
    return text


def _png_metadata(path: Path) -> tuple[dict[str, str], Optional[tuple[int, int]]]:
    """Read uncompressed PNG text fields and dimensions without image libraries."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return {}, None
    metadata = {}
    dimensions = None
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(data):
            return {}, None
        payload = data[start:end]
        expected_crc = struct.unpack(">I", data[end : end + 4])[0]
        actual_crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            return {}, None
        if kind == b"IHDR" and len(payload) >= 8:
            dimensions = struct.unpack(">II", payload[:8])
        elif kind == b"tEXt" and b"\0" in payload:
            key, value = payload.split(b"\0", 1)
            metadata[key.decode("latin-1")] = value.decode("latin-1")
        offset = end + 4
        if kind == b"IEND":
            break
    return metadata, dimensions


def _article_figure_is_current(path: Path, results: dict, target: dict) -> bool:
    from article.scripts.generate_figures import PNG_SOURCE_KEY, results_digest

    if not path.exists():
        return False
    metadata, dimensions = _png_metadata(path)
    return (
        metadata.get(PNG_SOURCE_KEY) == results_digest(results)
        and dimensions == tuple(target["dimensions"])
    )


def _render_article_figure(results: dict, renderer: str) -> bytes:
    from article.scripts import generate_figures

    if renderer == "article_conversion_figure":
        return generate_figures.render_conversion_figure(results)
    if renderer == "article_endpoints_figure":
        return generate_figures.render_endpoints_figure(results)
    raise ValueError(f"unknown article figure renderer: {renderer}")


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if name in names:
                return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.literal_eval(node.value)
    raise ValueError(f"{path} does not define a literal {name}")


def _synthetic_findings_current(summary: dict, text: str) -> bool:
    metrics = {metric["name"]: metric for metric in summary["metrics"]}
    magnitude = metrics["SHAP vs LIME magnitude disagreement (top-3)"]
    expected = (
        f"median is {magnitude['good_median']:.2f} in the nominally good regime "
        f"and {magnitude['bad_median']:.2f} in the stress regime"
    )
    separating = {
        metric["name"]
        for metric in summary["metrics"]
        if metric["good_pass_rate"] >= 0.8 and metric["bad_flag_rate"] >= 0.75
    }
    return expected in text and separating == {
        "SHAP removal-effect correlation",
        "LIME local fidelity (infidelity)",
        "Max sensitivity",
    }


def _root_readme_findings_current(
    synthetic: dict, real: dict, change_report: str, text: str
) -> bool:
    separating = {
        metric["name"]
        for metric in synthetic["metrics"]
        if metric["good_pass_rate"] >= 0.8 and metric["bad_flag_rate"] >= 0.75
    }
    metrics = real["metrics"]
    stability = metrics["stability_rank_topk"]
    comprehensiveness = metrics["comprehensiveness"]
    total = stability["run_counts"]["total"]
    p90 = _format_number(comprehensiveness["pooled_p90"])
    normalized = " ".join(text.split())
    normalized_change_report = " ".join(change_report.split())
    expected = [
        "SHAP removal-effect correlation, LIME infidelity, and max-sensitivity",
        "SHAP 移除效应相关、LIME infidelity 和最大敏感度",
        f"Across {total} Adult/Diabetes run aggregates",
        f"Adult/Diabetes 的 {total} 个运行汇总中",
        f"pooled top-k stability median was `{stability['pooled_median']:.3f}`",
        f"top-k 稳定性的合并中位数为 `{stability['pooled_median']:.3f}`",
        (
            f"pooled median is `{comprehensiveness['pooled_median']:.2f}`, "
            f"while P90 is `{p90}`"
        ),
        (
            f"合并中位数为 `{comprehensiveness['pooled_median']:.2f}`，"
            f"P90 为 `{p90}`"
        ),
    ]
    return (
        all(value in normalized for value in expected)
        and separating == {
            "SHAP removal-effect correlation",
            "LIME local fidelity (infidelity)",
            "Max sensitivity",
        }
        and "gives a median of 1.0000" in normalized_change_report
        and "restores `1.000`" in normalized
        and "中位数会恢复到 `1.000`" in normalized
    )


def _real_findings_current(summary: dict, change_report: str, text: str) -> bool:
    metrics = summary["metrics"]
    expected = [
        f"pooled top-k stability is now {metrics['stability_rank_topk']['pooled_median']:.2f}",
        (
            f"median is {metrics['comprehensiveness']['pooled_median']:.2f} and "
            f"P90 is {_format_number(metrics['comprehensiveness']['pooled_p90'])}"
        ),
        (
            f"with pooled median {_format_number(metrics['sensitivity']['pooled_median'])} "
            f"and P90 {metrics['sensitivity']['pooled_p90']:.4f}"
        ),
        (
            f"({metrics['removal_corr']['per_dataset_median']['adult']:.2f} vs "
            f"{metrics['removal_corr']['per_dataset_median']['diabetes']:.2f})"
        ),
        (
            f"({metrics['comprehensiveness']['per_dataset_median']['adult']:.2f} vs "
            f"{metrics['comprehensiveness']['per_dataset_median']['diabetes']:.2f})"
        ),
        f"**{metrics['stability_rank_topk']['pooled_median']:.2f}** (stability)",
        f"**{metrics['disagreement_rank_topk']['pooled_median']:.2f}** (agreement)",
        (
            f"Adult {metrics['infidelity']['per_dataset_median']['adult']:.2f} vs "
            f"Diabetes {metrics['infidelity']['per_dataset_median']['diabetes']:.2f}"
        ),
    ]
    not_computed = metrics["distribution_rank"]["run_counts"]["not_computed"]
    return (
        all(value in text for value in expected)
        and not_computed == 1
        and "Twelve of the 42 dataset/scope comparisons" in change_report
        and "an automatic regression failure" in change_report
    )


def _decision_policy_current(root: Path, mapping: dict) -> bool:
    report_path = root / "explaintrust/report.py"
    runner_path = root / "experiments/benchmark_real_data.py"
    defaults = _literal_assignment(report_path, "DEFAULT_THRESHOLDS")
    directions = _literal_assignment(report_path, "_THRESHOLD_DIRECTIONS")
    runner = _literal_assignment(runner_path, "CURRENT_DEFAULTS")
    real_target = next(
        target for target in mapping["generated_targets"]
        if target["id"] == "real_results_table"
    )
    policy_keys = real_target["policy_keys"]
    scored = set()
    for metric, (direction, good, warn) in runner.items():
        if direction == "descriptive":
            if good is not None or warn is not None or metric in policy_keys:
                return False
            continue
        policy_key = policy_keys.get(metric)
        if policy_key is None:
            return False
        scored.add(metric)
        if directions.get(policy_key) != direction:
            return False
        if defaults.get(policy_key) != (good, warn):
            return False
    return scored == set(policy_keys)


def _manual_claims_stale(mapping: dict, root: Path) -> list[str]:
    stale = []

    def mark(path: str):
        if path not in stale:
            stale.append(path)

    for claim in mapping["checked_manual_claims"]:
        if claim.get("implementation_status") != "implemented":
            continue
        texts = {}
        for target in claim["targets"]:
            texts[target] = (root / target).read_text()
            required = claim.get("required_text", {}).get(target, [])
            forbidden = claim.get("forbidden_text", {}).get(target, [])
            if not all(value in texts[target] for value in required):
                mark(target)
            if any(value in texts[target] for value in forbidden):
                mark(target)

        kind = claim.get("check_kind")
        if kind == "root_readme_findings":
            synthetic = _load_json(root / "experiments/results/synthetic/summary.json")
            real = _load_json(root / "experiments/results/real/summary.json")
            change_report = (root / "experiments/results/real/change_report.md").read_text()
            if not _root_readme_findings_current(
                synthetic, real, change_report, texts["README.md"]
            ):
                mark("README.md")
        elif kind == "synthetic_findings":
            summary = _load_json(root / "experiments/results/synthetic/summary.json")
            if not _synthetic_findings_current(summary, texts["experiments/README.md"]):
                mark("experiments/README.md")
        elif kind == "real_findings":
            summary = _load_json(root / "experiments/results/real/summary.json")
            change_report = (root / "experiments/results/real/change_report.md").read_text()
            if not _real_findings_current(
                summary, change_report, texts["experiments/benchmark_README.md"]
            ):
                mark("experiments/benchmark_README.md")
        elif kind == "decision_policy_mirror":
            if not _decision_policy_current(root, mapping):
                mark("experiments/benchmark_real_data.py")
    return stale


def build(*, check: bool, root: Path = ROOT) -> list[str]:
    mapping = _load_json(root / MAP_PATH)
    sources = {source["id"]: source for source in mapping["sources"]}
    changed = _stale_produced_sources(mapping, root)
    if changed and not check:
        raise ValueError(
            "generated source bundle is stale; run its declared producer first: "
            + ", ".join(changed)
        )
    for target in mapping["generated_targets"]:
        if target.get("implementation_status") != "implemented":
            continue
        source_id = target["source_ids"][0]
        source = _load_json(root / sources[source_id]["path"])
        renderer_name = target["renderer"]
        path = root / target["target"]
        expected = None

        if renderer_name in RENDERERS:
            renderer = RENDERERS[renderer_name]
            if renderer_name == "synthetic_results_table":
                rendered = renderer(source, mapping)
            else:
                rendered = renderer(source, target, mapping)
            current = path.read_text()
            expected = _replace_block(current, target, rendered)
            is_current = expected == current
        elif renderer_name == "article_numeric_claims":
            current = path.read_text()
            expected = _render_article_claims(current, source, target)
            is_current = expected == current
        elif renderer_name == "article_conversion_data":
            from article.scripts.generate_figures import conversion_payload

            expected = json.dumps(
                conversion_payload(source),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
            is_current = path.read_text() == expected
        elif renderer_name in {
            "article_conversion_figure", "article_endpoints_figure",
        }:
            is_current = _article_figure_is_current(path, source, target)
        else:
            raise ValueError(f"unknown implemented renderer: {renderer_name}")

        if is_current:
            continue
        changed.append(target["target"])
        if check:
            continue
        if renderer_name in {
            "article_conversion_figure", "article_endpoints_figure",
        }:
            path.write_bytes(_render_article_figure(source, renderer_name))
        else:
            path.write_text(expected)

    manual_stale = _manual_claims_stale(mapping, root)
    for path in manual_stale:
        if path not in changed:
            changed.append(path)
    if manual_stale and not check:
        raise ValueError(
            "manual result claims require review; they were not modified automatically: "
            + ", ".join(manual_stale)
        )
    return changed


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="report stale generated artifacts without modifying files",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    changed = build(check=args.check)
    if args.check and changed:
        print("stale public result artifacts or manual contracts:")
        for path in changed:
            print(f"- {path}")
        return 1
    mapping = _load_json(ROOT / MAP_PATH)
    total = sum(
        target.get("implementation_status") == "implemented"
        for target in mapping["generated_targets"]
    )
    manual_total = sum(
        claim.get("implementation_status") == "implemented"
        for claim in mapping["checked_manual_claims"]
    )
    if args.check:
        print(
            f"checked {total} generated result artifacts and "
            f"{manual_total} manual result contracts"
        )
    else:
        print(
            f"updated {len(changed)} of {total} generated result artifacts; "
            f"verified {manual_total} manual result contracts"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
