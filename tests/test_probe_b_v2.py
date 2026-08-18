import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from probes.metrics import analyze, strict_success, wrong_purchase
from probes.runner import (
    CallBudget,
    CallBudgetExceeded,
    FakeShopEnv,
    MockClient,
    build_arm_inputs,
    load_records,
    main as runner_main,
    run_trajectory,
    schedule_pairs,
)
from probes.task_schema import (
    TaskValidationError,
    latent_goal_satisfied,
    load_tasks,
    validate_tasks,
)
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS


class ProbeBV2TaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()

    def test_frozen_tasks_pass_all_offline_invariants(self):
        result = validate_tasks(self.tasks)

        self.assertEqual(result["task_count"], 25)
        self.assertEqual(
            result["distribution"],
            {"brand": 3, "budget": 13, "color": 6, "origin": 3},
        )
        self.assertEqual(len(result["task_hash"]), 64)

    def test_validator_rejects_leakage_and_legacy_profile(self):
        broken = deepcopy(self.tasks)
        broken[0]["under_query"] += " 1000元"
        broken[0]["fake_profile"] = {"budget": "1000元"}

        with self.assertRaisesRegex(TaskValidationError, "leaks latent term"):
            validate_tasks(broken)

    def test_oracle_and_no_ask_share_the_exact_system_prompt(self):
        task = self.tasks[0]
        no_prompt, no_user = build_arm_inputs(task, "no_ask")
        oracle_prompt, oracle_answer = build_arm_inputs(task, "oracle_ask")

        self.assertEqual(no_prompt[0], oracle_prompt[0])
        self.assertEqual(no_user, oracle_prompt[1]["content"])
        self.assertEqual(oracle_prompt[2]["content"], task["oracle_turn"]["question"])
        self.assertEqual(oracle_answer, task["oracle_turn"]["answer"])
        self.assertNotIn(
            "ask_user",
            {schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS},
        )

    def test_mock_arms_use_equal_environment_steps_and_release(self):
        task = self.tasks[0]
        trajectories = {}
        environments = {}
        task_hash = validate_tasks(self.tasks)["task_hash"]
        for arm in ("no_ask", "oracle_ask"):
            env = FakeShopEnv(task)
            environments[arm] = env
            trajectories[arm] = run_trajectory(
                task,
                arm,
                client=MockClient(),
                env_factory=lambda current_task, url, env=env: env,
                base_url="mock://offline",
                max_steps=35,
                run_id="test",
                task_hash=task_hash,
                llm_calls_before=0,
            )

        self.assertEqual(len(trajectories["no_ask"]["steps"]), 3)
        self.assertEqual(len(trajectories["oracle_ask"]["steps"]), 3)
        self.assertTrue(environments["no_ask"].released)
        self.assertTrue(environments["oracle_ask"].released)
        self.assertEqual(
            [message["role"] for message in trajectories["oracle_ask"]["messages"][:4]],
            ["system", "user", "assistant", "user"],
        )
        self.assertTrue(strict_success(trajectories["oracle_ask"]))
        self.assertTrue(latent_goal_satisfied(task, trajectories["oracle_ask"]))

    def test_environment_is_released_when_client_fails(self):
        class FailingClient:
            def complete(self, messages, tools):
                raise OSError("model unavailable")

        task = self.tasks[0]
        env = FakeShopEnv(task)
        trajectory = run_trajectory(
            task,
            "no_ask",
            client=FailingClient(),
            env_factory=lambda current_task, url: env,
            base_url="mock://offline",
            max_steps=35,
            run_id="test",
            task_hash=validate_tasks(self.tasks)["task_hash"],
            llm_calls_before=0,
        )

        self.assertTrue(env.released)
        self.assertFalse(trajectory["probe"]["valid"])
        self.assertEqual(trajectory["status"], "error")

    def test_malformed_agent_tool_call_is_a_valid_behavioral_failure(self):
        class MalformedClient:
            def complete(self, messages, tools):
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "missing_query",
                            "type": "function",
                            "function": {
                                "name": "search_products",
                                "arguments": "{}",
                            },
                        }
                    ],
                }

        task = self.tasks[0]
        env = FakeShopEnv(task)
        trajectory = run_trajectory(
            task,
            "no_ask",
            client=MalformedClient(),
            env_factory=lambda current_task, url: env,
            base_url="mock://offline",
            max_steps=35,
            run_id="test",
            task_hash=validate_tasks(self.tasks)["task_hash"],
            llm_calls_before=0,
        )

        self.assertTrue(env.released)
        self.assertTrue(trajectory["probe"]["valid"])
        self.assertEqual(
            trajectory["probe"]["validity_reason"], "agent_malformed_tool_call"
        )

    def test_call_budget_stops_before_an_extra_request(self):
        changes = []
        budget = CallBudget(2, on_change=changes.append)

        budget.consume()
        budget.consume()
        with self.assertRaises(CallBudgetExceeded):
            budget.consume()

        self.assertEqual(budget.used, 2)
        self.assertEqual(changes, [1, 2])

    def test_schedule_is_seeded_and_contains_one_attempt_per_arm(self):
        first = schedule_pairs(self.tasks, 20260818, limit_pairs=3)
        second = schedule_pairs(self.tasks, 20260818, limit_pairs=3)

        self.assertEqual(
            [(task["task_id"], arm) for task, arm in first],
            [(task["task_id"], arm) for task, arm in second],
        )
        self.assertEqual(len(first), 6)
        counts = {}
        for task, arm in first:
            counts.setdefault(task["task_id"], set()).add(arm)
        self.assertTrue(all(arms == {"no_ask", "oracle_ask"} for arms in counts.values()))

    def test_mock_resume_does_not_duplicate_attempts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = [
                "--mode",
                "mock",
                "--run-id",
                "resume-test",
                "--limit-pairs",
                "1",
                "--outdir",
                temp_dir,
            ]
            self.assertEqual(runner_main(args), 0)
            self.assertEqual(runner_main(args), 0)

            records = load_records(Path(temp_dir) / "resume-test" / "trajectories.jsonl")
            self.assertEqual(len(records), 2)
            self.assertEqual(
                {(record["task_id"], record["arm"]) for record in records},
                {(records[0]["task_id"], "no_ask"), (records[0]["task_id"], "oracle_ask")},
            )


class ProbeBV2MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()

    def test_wrong_purchase_is_not_inferred_from_any_negative_or_failure_status(self):
        early_stop = _record(self.tasks[0]["task_id"], "no_ask", reward_type="early_abstain")
        wrong = _record(self.tasks[0]["task_id"], "oracle_ask", reward_type="wrong_purchase")

        self.assertFalse(wrong_purchase(early_stop))
        self.assertTrue(wrong_purchase(wrong))

    def test_paired_analysis_counts_directional_flips(self):
        task_a, task_b = self.tasks[:2]
        records = [
            _record(task_a["task_id"], "no_ask", status="assistant_final"),
            _record(task_a["task_id"], "oracle_ask", reward_type="gold_purchase"),
            _record(task_b["task_id"], "no_ask", reward_type="gold_purchase"),
            _record(task_b["task_id"], "oracle_ask", reward_type="wrong_purchase"),
        ]

        report = analyze([task_a, task_b], records, mode="real")

        self.assertEqual(report["valid_pair_count"], 2)
        self.assertEqual(
            report["paired_transitions"]["no_ask_fail_to_oracle_success"], 1
        )
        self.assertEqual(
            report["paired_transitions"]["no_ask_success_to_oracle_fail"], 1
        )
        self.assertEqual(report["paired_transitions"]["net_strict_wins"], 0)
        self.assertEqual(report["continue_gate"]["status"], "INCOMPLETE")

    def test_analysis_rejects_duplicate_task_arm_records(self):
        task = self.tasks[0]
        record = _record(task["task_id"], "no_ask", status="assistant_final")

        with self.assertRaisesRegex(ValueError, "duplicate task-arm"):
            analyze([task], [record, deepcopy(record)])


def _record(task_id, arm, *, reward_type=None, status="done"):
    done = status == "done"
    if reward_type is None and done:
        reward_type = "early_abstain"
    detail = {}
    if reward_type:
        detail = {
            "reward_version": "shopsimulator-reward-v3",
            "reward_type": reward_type,
            "reward_valid": True,
            "purchase_success": reward_type in {"gold_purchase", "wrong_purchase"},
            "termination_reason": reward_type,
        }
    return {
        "task_id": int(task_id),
        "arm": arm,
        "status": status,
        "done": done,
        "steps": [],
        "terminal_result": {
            "done": done,
            "over": done,
            "purchase": {"price": 1.0} if done else {},
            "reward_detail": detail,
        },
        "probe": {"valid": True},
    }


if __name__ == "__main__":
    unittest.main()
