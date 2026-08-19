import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from probes.autonomous_metrics import analyze, load_reference_records
from probes.autonomous_runner import (
    ASK_USER_TOOL_SCHEMA,
    AUTONOMOUS_ARM,
    NEUTRAL_ANSWER,
    SELECTED_TASK_IDS,
    AutonomousAskClient,
    answer_question,
    load_records,
    main as runner_main,
    run_autonomous_trajectory,
    select_pilot_tasks,
)
from probes.runner import FakeShopEnv
from probes.task_schema import load_tasks, validate_tasks


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tool_names = []

    def complete(self, messages, tools):
        self.tool_names.append([tool["function"]["name"] for tool in tools])
        if not self.responses:
            return {"role": "assistant", "content": "exhausted"}
        return deepcopy(self.responses.pop(0))


class ProbeBAutonomousProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()
        cls.task_hash = validate_tasks(cls.tasks)["task_hash"]

    def test_selection_is_frozen_and_not_outcome_driven(self):
        selected = select_pilot_tasks(self.tasks)

        self.assertEqual(tuple(int(task["task_id"]) for task in selected), SELECTED_TASK_IDS)
        self.assertEqual(
            {"budget": 4, "color": 6},
            {
                field: sum(task["latent_goal"]["field"] == field for task in selected)
                for field in ("budget", "color")
            },
        )

    def test_question_classifier_is_exactly_single_field(self):
        task = _task(self.tasks, 5496)

        correct = answer_question(task, "这次购买的预算上限是多少？")
        wrong = answer_question(task, "您偏好什么颜色？")
        compound = answer_question(task, "预算和颜色分别有什么要求？")
        unrelated = answer_question(task, "还有什么要求吗？")

        self.assertEqual(correct["classification"], "correct_ask")
        self.assertEqual(correct["answer"], task["oracle_turn"]["answer"])
        for result in (wrong, compound, unrelated):
            self.assertEqual(result["classification"], "incorrect_ask")
            self.assertEqual(result["answer"], NEUTRAL_ANSWER)
        self.assertEqual(compound["recognized_fields"], ["budget", "color"])

    def test_correct_ask_is_intercepted_before_three_environment_steps(self):
        task = _task(self.tasks, 5496)
        base_client = SequenceClient(
            [
                _assistant_tool(
                    "ask_user",
                    {"question": "这次购买的预算上限是多少？"},
                    "ask-1",
                ),
                _assistant_tool(
                    "search_products", {"query": "探针测试商品"}, "search-1"
                ),
                _assistant_tool("open_product", {"asin": "100000000001"}, "open-1"),
                _assistant_tool("buy_now", {}, "buy-1"),
            ]
        )
        env = FakeShopEnv(task)

        trajectory = run_autonomous_trajectory(
            task,
            base_client=base_client,
            env_factory=lambda current_task, url: env,
            shopsim_base_url="mock://offline",
            max_steps=35,
            run_id="test",
            task_hash=self.task_hash,
            llm_calls_before=0,
        )

        self.assertEqual(trajectory["status"], "done")
        self.assertEqual(len(trajectory["steps"]), 3)
        self.assertTrue(env.released)
        self.assertEqual(
            trajectory["probe"]["autonomous_ask"]["classification"], "correct_ask"
        )
        self.assertIn("ask_user", base_client.tool_names[0])
        self.assertTrue(
            all("ask_user" not in names for names in base_client.tool_names[1:])
        )
        visible_messages = json.dumps(
            trajectory["messages"], ensure_ascii=False, sort_keys=True
        )
        self.assertNotIn(task["clear_query"], visible_messages)
        self.assertIn(task["oracle_turn"]["answer"], visible_messages)

    def test_direct_shopping_permanently_closes_ask_tool(self):
        task = _task(self.tasks, 5496)
        base_client = SequenceClient(
            [
                _assistant_tool(
                    "search_products", {"query": "探针测试商品"}, "search-1"
                ),
                _assistant_tool("open_product", {"asin": "100000000001"}, "open-1"),
                _assistant_tool("buy_now", {}, "buy-1"),
            ]
        )
        adapter = AutonomousAskClient(base_client, task)
        env = FakeShopEnv(task)

        trajectory = run_autonomous_trajectory(
            task,
            base_client=adapter,
            env_factory=lambda current_task, url: env,
            shopsim_base_url="mock://offline",
            max_steps=35,
            run_id="test",
            task_hash=self.task_hash,
            llm_calls_before=0,
        )

        self.assertEqual(trajectory["probe"]["autonomous_ask"]["classification"], "no_ask")
        self.assertIn("ask_user", base_client.tool_names[0])
        self.assertTrue(
            all("ask_user" not in names for names in base_client.tool_names[1:])
        )

    def test_mock_resume_writes_each_selected_task_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = [
                "--mode",
                "mock",
                "--run-id",
                "resume-v3",
                "--limit-tasks",
                "2",
                "--outdir",
                temp_dir,
            ]
            self.assertEqual(runner_main(args), 0)
            self.assertEqual(runner_main(args), 0)

            records = load_records(Path(temp_dir) / "resume-v3" / "trajectories.jsonl")
            self.assertEqual(len(records), 2)
            self.assertEqual(
                [int(record["task_id"]) for record in records],
                list(SELECTED_TASK_IDS[:2]),
            )
            self.assertTrue(all(record["arm"] == AUTONOMOUS_ARM for record in records))


class ProbeBAutonomousReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()
        cls.task_hash = validate_tasks(cls.tasks)["task_hash"]

    def test_supplement_replaces_only_matching_invalid_primary(self):
        invalid = _record(18637, "oracle_ask", valid=False, strict=False)
        supplement = _record(18637, "oracle_ask", valid=True, strict=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            _write_jsonl(run_dir / "trajectories.jsonl", [invalid])
            (run_dir / "manifest.json").write_text(
                json.dumps({"task_hash": self.task_hash}), encoding="utf-8"
            )
            (run_dir / "supplemental_18637_oracle_ask.json").write_text(
                json.dumps(supplement), encoding="utf-8"
            )

            records, audit = load_reference_records(run_dir, self.task_hash)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["probe"]["valid"])
        self.assertTrue(audit["supplemental_used"])
        self.assertEqual(audit["replaced_key"], "18637:oracle_ask")

    def test_supplement_cannot_replace_valid_primary(self):
        primary = _record(18637, "oracle_ask", valid=True, strict=False)
        supplement = _record(18637, "oracle_ask", valid=True, strict=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            _write_jsonl(run_dir / "trajectories.jsonl", [primary])
            (run_dir / "manifest.json").write_text(
                json.dumps({"task_hash": self.task_hash}), encoding="utf-8"
            )
            (run_dir / "supplemental_18637_oracle_ask.json").write_text(
                json.dumps(supplement), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "invalid primary"):
                load_reference_records(run_dir, self.task_hash)

    def test_complete_mock_report_uses_three_conditions(self):
        selected = select_pilot_tasks(self.tasks)
        references = []
        autonomous = []
        for index, task in enumerate(selected):
            references.append(
                _record(task["task_id"], "no_ask", strict=index >= 5)
            )
            references.append(_record(task["task_id"], "oracle_ask", strict=True))
            autonomous.append(
                _record(
                    task["task_id"],
                    AUTONOMOUS_ARM,
                    strict=index >= 2,
                    ask_classification="correct_ask" if index < 7 else "no_ask",
                )
            )

        report = analyze(selected, autonomous, references, mode="real")

        self.assertEqual(report["conditions"][AUTONOMOUS_ARM]["strict_success_count"], 8)
        self.assertEqual(report["ask_metrics"]["correct_ask_count"], 7)
        self.assertEqual(report["oracle_gap"]["recovered_count"], 3)
        self.assertEqual(report["oracle_gap"]["recovery_rate"], 0.6)
        self.assertEqual(report["continue_gate"]["status"], "PASS")


def _task(tasks, task_id):
    return next(task for task in tasks if int(task["task_id"]) == int(task_id))


def _assistant_tool(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _record(
    task_id,
    arm,
    *,
    valid=True,
    strict=False,
    ask_classification="no_ask",
):
    reward_type = "gold_purchase" if strict else "early_abstain"
    return {
        "task_id": int(task_id),
        "arm": arm,
        "status": "done",
        "done": True,
        "steps": [],
        "terminal_result": {
            "done": True,
            "over": True,
            "purchase": {"price": 1.0},
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": reward_type,
                "reward_valid": True,
                "purchase_success": strict,
                "termination_reason": reward_type,
            },
        },
        "probe": {
            "valid": bool(valid),
            "task_hash": validate_tasks(load_tasks())["task_hash"],
            "autonomous_ask": {"classification": ask_classification},
        },
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
