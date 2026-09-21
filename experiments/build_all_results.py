"""Generate or check public result tables from canonical summaries.

Current scope: the synthetic and real-data Markdown tables declared as
``implemented`` in ``experiments/public_result_map.json``. Article artifacts
and checked manual claims remain planned follow-up work.

Run from the repository root:

    python -m experiments.build_all_results
    python -m experiments.build_all_results --check
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from explaintrust.report import DEFAULT_THRESHOLDS, _THRESHOLD_DIRECTIONS


ROOT = Path(__file__).parents[1]
MAP_PATH = "experiments/public_result_map.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


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


def build(*, check: bool, root: Path = ROOT) -> list[str]:
    mapping = _load_json(root / MAP_PATH)
    sources = {source["id"]: source for source in mapping["sources"]}
    changed = []
    for target in mapping["generated_targets"]:
        if target.get("implementation_status") != "implemented":
            continue
        source_id = target["source_ids"][0]
        summary = _load_json(root / sources[source_id]["path"])
        renderer = RENDERERS[target["renderer"]]
        if target["renderer"] == "synthetic_results_table":
            rendered = renderer(summary, mapping)
        else:
            rendered = renderer(summary, target, mapping)
        path = root / target["target"]
        current = path.read_text()
        expected = _replace_block(current, target, rendered)
        if expected == current:
            continue
        changed.append(target["target"])
        if not check:
            path.write_text(expected)
    return changed


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="report stale generated tables without modifying files",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    changed = build(check=args.check)
    if args.check and changed:
        print("stale generated result tables:")
        for path in changed:
            print(f"- {path}")
        return 1
    if args.check:
        print(f"checked {len(RENDERERS)} generated result tables")
    else:
        print(f"updated {len(changed)} of {len(RENDERERS)} generated result tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
