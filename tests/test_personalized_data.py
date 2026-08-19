import gzip
import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.personalization.schema import (
    TaskValidationError,
    actor_view,
    finalize_task,
    validate_task,
)
from shopping_grpo.personalization.source import load_source_tasks


def _profile():
    return {
        "profile_id": "profile-1",
        "stable_facts": [],
        "category_preferences": [],
        "brand_preferences": [],
        "budget_preferences": [],
        "attribute_preferences": [],
        "option_preferences": [],
    }


def _clarification_task():
    return {
        "schema_version": "personalized-shopping-task-v1",
        "task_id": "pca-1",
        "source": {
            "shopsim_task_id": 7,
            "target_asin": "100000000001",
            "source_environment_version": "shopsimulator-environment-v2.1",
        },
        "scenario": "clarification_required",
        "profile": _profile(),
        "current_request": "我想给自己买一双缓震羽毛球鞋。",
        "private_goal": {
            "category": "羽毛球鞋",
            "constraints": [
                {
                    "constraint_id": "c-size",
                    "field": "size",
                    "value": "40",
                    "hardness": "hard",
                    "source": "clarification_answer",
                }
            ],
        },
        "clarification": {
            "should_ask": True,
            "max_questions": 2,
            "targets": [
                {
                    "constraint_id": "c-size",
                    "field": "size",
                    "answer": "这次需要40码。",
                    "answer_facts": {"size": "40"},
                }
            ],
        },
        "conflicts": [],
        "generation": {"generator": "test"},
        "audit": {},
    }


class PersonalizedTaskSchemaTests(unittest.TestCase):
    def test_valid_task_hashes_and_actor_view_hides_private_state(self):
        task = finalize_task(_clarification_task())
        self.assertEqual(len(task["audit"]["task_hash"]), 64)

        visible = actor_view(task)
        payload = json.dumps(visible, ensure_ascii=False)
        self.assertNotIn("private_goal", visible)
        self.assertNotIn("clarification", visible)
        self.assertNotIn("40", payload)

    def test_rejects_leakage_and_duplicate_question_fields(self):
        leaked = _clarification_task()
        leaked["current_request"] += " 我需要40码。"
        with self.assertRaises(TaskValidationError) as context:
            validate_task(leaked)
        self.assertTrue(any("leaks" in error for error in context.exception.errors))

        duplicate = _clarification_task()
        duplicate["private_goal"]["constraints"].append(
            {
                "constraint_id": "c-size-2",
                "field": "size",
                "value": "宽楦",
                "hardness": "hard",
                "source": "clarification_answer",
            }
        )
        duplicate["clarification"]["targets"].append(
            {
                "constraint_id": "c-size-2",
                "field": "size",
                "answer": "需要宽楦。",
                "answer_facts": {"size": "宽楦"},
            }
        )
        with self.assertRaises(TaskValidationError) as context:
            validate_task(duplicate)
        self.assertTrue(
            any("duplicate clarification field" in error for error in context.exception.errors)
        )


class SourceTaskExportTests(unittest.TestCase):
    def test_loads_environment_task_order_without_exporting_reference_persona(self):
        products = [
            {
                "asin": "100",
                "title": "鞋",
                "shop_name": "店",
                "category": "运动›鞋",
                "pricing": [99],
                "attribute": ["缓震"],
                "customization_options": {},
                "instructions": [
                    {
                        "instruction": "买鞋",
                        "attributes": ["缓震"],
                        "instruction_options": [],
                        "instruction_simple": "鞋",
                    }
                ],
                "user_persona": {"用户ID": "must-not-export"},
            },
            {
                "asin": "200",
                "title": "杯子",
                "shop_name": "店",
                "category": "家居›杯子",
                "pricing": [20],
                "attribute": ["保温"],
                "customization_options": {
                    "容量": [
                        {
                            "value": "500ml",
                            "price": 20,
                            "is_available": True,
                        }
                    ]
                },
                "instructions": [
                    {
                        "instruction": "买500ml保温杯",
                        "attributes": ["保温"],
                        "instruction_options": ["500ml"],
                        "instruction_simple": "买杯子",
                    }
                ],
                "user_persona": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "products.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(products, handle, ensure_ascii=False)
            rows = load_source_tasks([1, 0], product_data=path)

        self.assertEqual([row["target_asin"] for row in rows], ["200", "100"])
        self.assertEqual(rows[0]["shopsim_task_id"], 1)
        self.assertNotIn("user_persona", rows[1])
        self.assertTrue(rows[1]["has_reference_persona"])


if __name__ == "__main__":
    unittest.main()
