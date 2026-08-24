"""Frozen contracts for the formal GRPO A/B gate and scheduler horizon patch."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import apply_verl_scheduler_horizon_patch as scheduler_patch
from scripts.run_formal_grpo_ab import build_command


class FormalGrpoABTest(unittest.TestCase):
    def args(self, arm: str) -> SimpleNamespace:
        return SimpleNamespace(
            arm=arm,
            model=Path("model"),
            output=Path(f"output-{arm}"),
            shopper_model="shopper",
            shopper_base_url="https://shopper.example.test/v1",
            env_url="http://127.0.0.1:5700",
            seed=20260823,
            stage_end=50,
            experiment_name=None,
            resume_from_checkpoint=None,
            logger="swanlab",
            dry_run=False,
            preflight_only=True,
        )

    def test_a_and_b_only_differ_in_reward_contract_and_identity(self):
        a = build_command(self.args("a"))
        b = build_command(self.args("b"))
        self.assertIn("none", a)
        self.assertIn("bounded-v1", b)
        for command in (a, b):
            self.assertIn("trainer.n_gpus_per_node=4", command)
            self.assertIn("trainer.total_training_steps=50", command)
            self.assertIn("shopping_scheduler.total_training_steps=500", command)
            self.assertIn("trainer.save_freq=25", command)
            self.assertIn("trainer.test_freq=50", command)
            self.assertIn("trainer.val_before_train=false", command)
            self.assertIn("trainer.project_name=shopping-multiturn-agentic", command)
            self.assertIn("actor_rollout_ref.model.use_remove_padding=true", command)
            self.assertIn("actor_rollout_ref.model.use_liger=true", command)
            self.assertIn("data.dataloader_num_workers=0", command)
            self.assertIn("--preflight-only", command)

    def test_stage_end_cannot_exceed_scheduler_horizon(self):
        args = self.args("a")
        args.stage_end = 501
        with self.assertRaises(SystemExit):
            build_command(args)

    def test_scheduler_patch_is_reversible_to_pinned_input(self):
        source = (
            "import json\n\n"
            "class Trainer:\n"
            "    def configure(self):\n"
            "        total_training_steps = 50\n"
            "        try:\n"
            + scheduler_patch.OLD_SOURCE
            + "        except Exception:\n"
            "            raise\n"
        )
        expected_input = hashlib.sha256(source.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ray_trainer.py"
            target.write_text(source, encoding="utf-8", newline="\n")
            with patch.object(scheduler_patch, "EXPECTED_INPUT_SHA256", expected_input):
                scheduler_patch.apply_patch(target)
                patched = target.read_text(encoding="utf-8")
                self.assertIn(scheduler_patch.PATCH_MARKER, patched)
                scheduler_patch.verify_patched(target)
                scheduler_patch.restore_patch(target)
            self.assertEqual(target.read_text(encoding="utf-8"), source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
