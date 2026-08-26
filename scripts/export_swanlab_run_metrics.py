#!/usr/bin/env python3
"""Export a SwanLab run's complete scalar history and compact Chinese report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shopping_grpo.evaluation.swanlab_history import (
    build_markdown_report,
    extract_metric_points,
    is_important_key,
    json_safe,
    metric_key,
    serialize_points,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-path",
        default="mode/shopping-multiturn-agentic/4cmh0p3k",
        help="SwanLab run path: username/project/run_id",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/swanlab-bpo-step200"),
    )
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument(
        "--all-custom",
        action="store_true",
        help="Download every custom scalar rather than the important subset.",
    )
    return parser.parse_args()


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    try:
        import swanlab
    except ImportError as exc:
        raise SystemExit(
            "缺少 SwanLab SDK；请使用项目 GRPO Python（固定 swanlab==0.9.1）运行"
        ) from exc

    api_key = os.environ.get("SWANLAB_API_KEY")
    api = swanlab.Api(api_key=api_key) if api_key else swanlab.Api()
    run = api.run(path=args.run_path)
    run_metadata = json_safe(run.json())

    custom_items = list(run.series(metric_type="SCALAR", metric_class="CUSTOM"))
    system_items = list(run.series(metric_type="SCALAR", metric_class="SYSTEM"))
    custom_catalog = sorted(metric_key(item) for item in custom_items)
    system_catalog = sorted(metric_key(item) for item in system_items)
    custom_keys = [
        key for key in custom_catalog if args.all_custom or is_important_key(key)
    ]
    system_keys = [key for key in system_catalog if is_important_key(key, system=True)]

    raw_custom = [
        json_safe(run.metrics(keys=chunk, all=True))
        for chunk in _chunks(custom_keys, args.chunk_size)
    ]
    raw_system = [
        json_safe(run.metrics(keys=chunk, all=True))
        for chunk in _chunks(system_keys, args.chunk_size)
    ]
    custom_series = {
        key: [point for payload in raw_custom for point in extract_metric_points(payload, key)]
        for key in custom_keys
    }
    system_series = {
        key: [point for payload in raw_system for point in extract_metric_points(payload, key)]
        for key in system_keys
    }

    snapshot = {
        "schema_version": "shopping-swanlab-history-v1",
        "run_path": args.run_path,
        "run": run_metadata,
        "catalog": {"custom": custom_catalog, "system": system_catalog},
        "selected_keys": {"custom": custom_keys, "system": system_keys},
        "summary": json_safe(run.summary(keys=custom_keys)) if custom_keys else {},
        "raw_metrics": {"custom": raw_custom, "system": raw_system},
        "parsed_points": {
            "custom": serialize_points(custom_series),
            "system": serialize_points(system_series),
        },
    }
    report = build_markdown_report(
        run_path=args.run_path,
        run_metadata=run_metadata,
        custom_series=custom_series,
        system_series=system_series,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "swanlab-history.json"
    report_path = args.output_dir / "swanlab-analysis.md"
    history_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"SWANLAB_HISTORY={history_path.resolve()}")
    print(f"SWANLAB_ANALYSIS={report_path.resolve()}")
    print("SWANLAB BPO HISTORY EXPORT ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

