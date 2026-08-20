import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_multiturn_sft_data import collect_until_target


def trajectory(task_id):
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "attempt_index": 0,
        "status": "done",
        "shopper_questions": [],
        "actor_llm_calls": 1,
        "shopper_llm_calls": 0,
    }


class CollectMultiturnSftDataCliTests(unittest.TestCase):
    def test_parallel_collection_stops_at_accepted_target(self):
        collected = []

        def collect_one(task, attempt_index):
            collected.append((task["task_id"], attempt_index))
            return trajectory(task["task_id"])

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "scripts.collect_multiturn_sft_data.acceptance_reasons",
            return_value=(True, []),
        ):
            written, accepted, infrastructure_failed = collect_until_target(
                tasks=[
                    {"task_id": 1},
                    {"task_id": 2},
                    {"task_id": 3},
                ],
                target_accepted=2,
                output_path=Path(tmpdir) / "raw.jsonl",
                attempts_per_task=1,
                workers=2,
                collect_one=collect_one,
            )

        self.assertEqual(len(written), 2)
        self.assertEqual(len(collected), 2)
        self.assertEqual(accepted, 2)
        self.assertFalse(infrastructure_failed)

    def test_parallel_collection_resumes_completed_attempts(self):
        collected = []

        def collect_one(task, attempt_index):
            collected.append((task["task_id"], attempt_index))
            return trajectory(task["task_id"])

        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir) / "raw.jsonl"
            raw.write_text(
                json.dumps(trajectory(1)) + "\n",
                encoding="utf-8",
            )
            with patch(
                "scripts.collect_multiturn_sft_data.acceptance_reasons",
                return_value=(True, []),
            ):
                written, accepted, _ = collect_until_target(
                    tasks=[
                        {"task_id": 1},
                        {"task_id": 2},
                        {"task_id": 3},
                    ],
                    target_accepted=2,
                    output_path=raw,
                    attempts_per_task=1,
                    workers=2,
                    collect_one=collect_one,
                )

        self.assertEqual(len(written), 1)
        self.assertEqual(collected, [(2, 0)])
        self.assertEqual(accepted, 2)


if __name__ == "__main__":
    unittest.main()
