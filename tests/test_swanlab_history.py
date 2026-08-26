from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

from scripts import export_swanlab_run_metrics
from shopping_grpo.evaluation.swanlab_history import (
    MetricPoint,
    build_markdown_report,
    extract_metric_points,
    is_important_key,
    json_safe,
)


def test_extract_metric_points_accepts_structured_multi_key_response():
    payload = {
        "data": [
            {
                "key": "summary/strict_success_rate",
                "list": [
                    {"step": 0, "value": 0.60, "timestamp": 1000},
                    {"step": 50, "value": 0.65, "timestamp": 2000},
                ],
            },
            {"key": "actor/loss", "list": [{"step": 1, "value": 0.1}]},
        ]
    }
    points = extract_metric_points(payload, "summary/strict_success_rate")
    assert [(point.step, point.value) for point in points] == [
        (0, 0.60),
        (50, 0.65),
    ]


def test_extract_metric_points_accepts_direct_key_pairs():
    payload = {"actor/loss": [[1, 0.3], [2, 0.2]]}
    assert extract_metric_points(payload, "actor/loss") == [
        MetricPoint(1, 0.3),
        MetricPoint(2, 0.2),
    ]


def test_key_selection_covers_bpo_training_and_system_metrics():
    assert is_important_key("bpo_sampling/seconds_to_full_batch")
    assert is_important_key("actor/ppo_kl")
    assert is_important_key("system/gpu.0.memoryAllocated", system=True)
    assert not is_important_key("unrelated/custom")


def test_json_safe_supports_swanlab_response_objects():
    class Response:
        data = {"list": [{"step": 1, "value": 2.0}]}

    assert json_safe(Response()) == {"list": [{"step": 1, "value": 2.0}]}


def test_report_contains_decision_steps_resources_and_alerts():
    custom = {
        "val-shopping/summary/strict_success_rate": [
            MetricPoint(0, 0.60),
            MetricPoint(50, 0.66),
            MetricPoint(200, 0.67),
        ],
        "actor/grad_norm": [MetricPoint(1, 1.0), MetricPoint(2, 120.0)],
    }
    system = {"system/gpu.0.utilization": [MetricPoint(1, 80.0)]}
    report = build_markdown_report(
        run_path="mode/project/run",
        run_metadata={"name": "formal", "state": "FINISHED"},
        custom_series=custom,
        system_series=system,
    )
    assert "step 200" in report
    assert "0.67" in report
    assert "GPU 与系统资源" in report
    assert "梯度尖峰" in report


def test_export_entrypoint_writes_complete_snapshot_and_report(tmp_path, monkeypatch):
    custom_payload = {
        "data": [
            {
                "key": "val-shopping/summary/strict_success_rate",
                "list": [{"step": 0, "value": 0.60}, {"step": 200, "value": 0.67}],
            },
            {
                "key": "actor/grad_norm",
                "list": [{"step": 1, "value": 1.5}],
            },
        ]
    }
    system_payload = {
        "data": [
            {
                "key": "system/gpu.0.utilization",
                "list": [{"step": 1, "value": 88.0}],
            }
        ]
    }

    class Run:
        def json(self):
            return {"name": "formal", "state": "FINISHED"}

        def series(self, *, metric_type, metric_class):
            assert metric_type == "SCALAR"
            keys = (
                ["val-shopping/summary/strict_success_rate", "actor/grad_norm"]
                if metric_class == "CUSTOM"
                else ["system/gpu.0.utilization"]
            )
            return [SimpleNamespace(key=key) for key in keys]

        def metrics(self, *, keys, all):
            assert all is True
            return custom_payload if keys[0] != "system/gpu.0.utilization" else system_payload

        def summary(self, *, keys):
            return {key: {"latest": 0.67} for key in keys}

    module = ModuleType("swanlab")
    module.Api = lambda **kwargs: SimpleNamespace(run=lambda path: Run())
    monkeypatch.setitem(sys.modules, "swanlab", module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_swanlab_run_metrics.py",
            "--run-path",
            "mode/project/run",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert export_swanlab_run_metrics.main() == 0
    snapshot = json.loads((tmp_path / "swanlab-history.json").read_text(encoding="utf-8"))
    report = (tmp_path / "swanlab-analysis.md").read_text(encoding="utf-8")

    assert snapshot["schema_version"] == "shopping-swanlab-history-v1"
    assert snapshot["parsed_points"]["custom"]["val-shopping/summary/strict_success_rate"][-1]["step"] == 200
    assert "step 200" in report
    assert "system/gpu.0.utilization" in report
