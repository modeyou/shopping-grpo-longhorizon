import gzip
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.persona import actor_instruction, has_persona_request  # noqa: E402


class PersonaTaskPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = SHOP_ENV / "data" / "fine_items_eval_train_all.json.gz"
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            cls.items = json.load(handle)

    def test_frozen_archive_has_stable_persona_subset(self):
        available = [
            task_id
            for task_id, item in enumerate(self.items)
            if has_persona_request(item, item["instructions"][0])
        ]

        self.assertEqual(len(self.items), 23421)
        self.assertEqual(len(available), 4666)

    def test_pilot_ids_are_persona_ready_and_not_final_evaluation(self):
        pilot = {
            json.loads(line)["task_id"]
            for line in (ROOT / "data" / "personalized" / "pilot_tasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        held_out = {
            json.loads(line)["task_id"]
            for line in (ROOT / "data" / "evaluation" / "tasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }

        self.assertEqual(len(pilot), 5)
        self.assertFalse(pilot & held_out)
        for task_id in pilot:
            item = self.items[task_id]
            instruction = item["instructions"][0]
            self.assertTrue(has_persona_request(item, instruction))
            self.assertEqual(
                actor_instruction(item, instruction, if_persona=True),
                instruction["instruction_simple"],
            )
            self.assertNotIn("instruction_sample", instruction)


if __name__ == "__main__":
    unittest.main()
