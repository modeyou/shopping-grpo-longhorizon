import gzip
import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.personalization.generation import (
    ARCHITECT_OUTPUT_CONTRACT,
    GenerationAPIError,
    OpenAICompatibleJSONClient,
    build_architect_task,
    extract_json_object,
    scenario_for_run,
    validate_critic_response,
)
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
                    "evidence": {"source_path": "required_options", "source_value": "40"},
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
                    "question": "请问需要什么尺码？",
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
                "evidence": {"source_path": "required_options", "source_value": "宽楦"},
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

    def test_rejects_constraint_claimed_explicit_but_absent_from_request(self):
        task = _clarification_task()
        task["scenario"] = "complete_request"
        task["private_goal"]["constraints"][0]["source"] = "request_explicit"
        task["clarification"] = {"should_ask": False, "max_questions": 2, "targets": []}
        with self.assertRaises(TaskValidationError) as context:
            validate_task(task)
        self.assertTrue(any("not grounded in current_request" in e for e in context.exception.errors))


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


class GenerationBoundaryTests(unittest.TestCase):
    def test_scenario_quotas_are_balanced_and_skip_completed_groups(self):
        accepted = ["complete_request"] * 5
        self.assertEqual(scenario_for_run(0, accepted, 20), "profile_resolvable")
        quotas = [scenario_for_run(i, [], 20) for i in range(4)]
        self.assertEqual(
            quotas,
            ["complete_request", "profile_resolvable", "clarification_required", "profile_conflict"],
        )

    def test_normalizes_common_provider_shape_without_weakening_evidence(self):
        generated = {
            "profile": {
                "preferences": [
                    {"field": "budget", "value": "100元以内", "confidence": "high"},
                    {"field": "color", "value": "黑色", "confidence": "medium"},
                ]
            },
            "current_request": "我想买一双黑色的缓震羽毛球鞋。",
            "private_goal": {
                "target_item": {"category": "羽毛球鞋", "title": "do-not-persist"},
                "constraints": [
                    {
                        "constraint_id": "c-function",
                        "field": "function",
                        "value": "缓震",
                        "hardness": "hard",
                        "source": "request_explicit",
                        "source_quote": "缓震",
                    }
                ],
            },
            "clarification": {"should_ask": False, "max_questions": 2, "targets": []},
            "conflicts": [],
        }
        task = build_architect_task(
            generated,
            source={
                "shopsim_task_id": 10,
                "target_asin": "100000000010",
                "source_hash": "source-hash",
                "category": "羽毛球鞋",
                "attributes": ["缓震"],
                "original_instruction": "我想买一双黑色的缓震羽毛球鞋。",
            },
            scenario="complete_request",
            sequence=1,
            model="test-model",
        )

        self.assertEqual(task["private_goal"]["category"], "羽毛球鞋")
        self.assertNotIn("target_item", task["private_goal"])
        self.assertEqual(task["profile"]["budget_preferences"][0]["field"], "budget")
        self.assertEqual(task["profile"]["attribute_preferences"][0]["field"], "color")
        self.assertIn('"category_preferences"', ARCHITECT_OUTPUT_CONTRACT)

    def test_api_json_parsing_and_code_owned_identity(self):
        generated = {
            "profile": {},
            "current_request": "我想买一双缓震羽毛球鞋。",
            "private_goal": {
                "category": "羽毛球鞋",
                "constraints": [
                    {
                        "constraint_id": "c-function",
                        "field": "function",
                        "value": "缓震",
                        "hardness": "hard",
                        "source": "request_explicit",
                        "source_quote": "缓震",
                    }
                ],
            },
            "clarification": {
                "should_ask": False,
                "max_questions": 2,
                "targets": [],
            },
            "conflicts": [],
        }

        def transport(endpoint, payload, headers, timeout):
            self.assertTrue(endpoint.endswith("/chat/completions"))
            self.assertEqual(headers["Authorization"], "Bearer test-key")
            self.assertEqual(timeout, 10)
            content = "```json\n" + json.dumps(generated, ensure_ascii=False) + "\n```"
            return {"choices": [{"message": {"content": content}}]}

        client = OpenAICompatibleJSONClient(
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            timeout=10,
            transport=transport,
        )
        parsed, _ = client.complete_json(system="system", user="user")
        task = build_architect_task(
            parsed,
            source={
                "shopsim_task_id": 9,
                "target_asin": "100000000009",
                "source_hash": "source-hash",
                "attributes": ["缓震"],
                "original_instruction": "我想买一双缓震羽毛球鞋。",
            },
            scenario="complete_request",
            sequence=3,
            model="test-model",
        )

        self.assertEqual(task["task_id"], "pca-000003")
        self.assertEqual(task["profile"]["profile_id"], "profile-000003")
        self.assertEqual(client.call_count, 1)
        self.assertEqual(extract_json_object('{"verdict":"accept","issues":[]}')["verdict"], "accept")
        self.assertEqual(
            validate_critic_response({"verdict": "reject", "issues": ["unnatural"]})["issues"],
            ["unnatural"],
        )

    def test_rejects_model_provenance_not_anchored_in_original_instruction(self):
        generated = {
            "profile": {},
            "current_request": "我想买缓震球鞋。",
            "private_goal": {
                "category": "球鞋",
                "constraints": [{
                    "constraint_id": "c1",
                    "field": "function",
                    "value": "缓震",
                    "hardness": "hard",
                    "source": "request_explicit",
                    "source_quote": "目标商品支持缓震",
                }],
            },
            "clarification": {"should_ask": False, "max_questions": 2, "targets": []},
            "conflicts": [],
        }
        with self.assertRaises(GenerationAPIError) as context:
            build_architect_task(
                generated,
                source={
                    "shopsim_task_id": 11,
                    "target_asin": "100000000011",
                    "category": "球鞋",
                    "original_instruction": "我想买缓震球鞋。",
                },
                scenario="complete_request",
                sequence=1,
                model="test-model",
            )
        self.assertIn("exact original_instruction span", str(context.exception))


if __name__ == "__main__":
    unittest.main()
