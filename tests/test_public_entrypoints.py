"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_grpo import (
    _complete_checkpoint_steps,
    _diagnostic_summary,
    build_command,
    main as grpo_main,
    parse_args,
    refresh_latest_checkpoint,
    validate_launcher_owned_ray,
    write_run_contract,
)
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_completion_audit_counts_only_committed_updates_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            diagnostics = output / "training_diagnostics.jsonl"
            diagnostics.write_text(
                json.dumps(
                    {
                        "event": "optimizer_step",
                        "global_step": 25,
                        "metrics": {"training/optimizer_updated": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            actor = output / "global_step_25" / "actor"
            actor.mkdir(parents=True)
            (actor / "state.bin").write_bytes(b"state")
            incomplete = output / "global_step_50" / "actor"
            incomplete.mkdir(parents=True)
            summary = _diagnostic_summary(diagnostics)
            self.assertEqual(summary["optimizer_updates"], 1)
            self.assertEqual(summary["max_global_step"], 25)
            self.assertEqual(_complete_checkpoint_steps(output), [25])

    def test_latest_checkpoint_requires_tracker_and_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "latest_checkpointed_iteration.txt").write_text(
                "25\n", encoding="utf-8"
            )
            checkpoint = output / "global_step_25"
            checkpoint.mkdir()
            self.assertIsNone(refresh_latest_checkpoint(output))
            actor = checkpoint / "actor"
            actor.mkdir()
            (actor / "state.bin").write_bytes(b"state")
            latest = refresh_latest_checkpoint(output)
            self.assertEqual(latest["step"], 25)
            recorded = json.loads(
                (output / "latest_checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recorded["path"], str(checkpoint.resolve()))

    def test_grpo_rejects_external_ray_address(self):
        validate_launcher_owned_ray({})
        with self.assertRaisesRegex(SystemExit, "launcher-owned local Ray"):
            validate_launcher_owned_ray({"RAY_ADDRESS": "auto"})

    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            data_manifest = temporary / "manifest.json"
            data_manifest.write_text("{}", encoding="utf-8")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--data-manifest",
                    str(data_manifest),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            with patch(
                "scripts.train_grpo.validate_grpo_data_manifest",
                return_value={},
            ):
                command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model.resolve()))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train.resolve()))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation.resolve()))
        self.assertEqual(
            environment["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"], "1"
        )
        self.assertEqual(
            environment["SHOPPING_GRPO_DIAGNOSTICS_PATH"],
            str(output.resolve() / "training_diagnostics.jsonl"),
        )
        self.assertEqual(environment["SHOPPING_REWARD_SHAPING_PROFILE"], "none")
        self.assertEqual(environment["GRPO_SEED"], "20260823")
        self.assertEqual(environment["SHOPPING_ENV_CONCURRENCY_PER_WORKER"], "2")
        self.assertEqual(environment["SHOPPING_EXPECTED_SHOPSIM_SLOTS"], "20")
        self.assertIn("trainer.logger=[console]", command)

    def test_grpo_run_contract_is_hashed_and_secret_free(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            files = {}
            for name in (
                "train.parquet",
                "validation.parquet",
                "grpo.yaml",
                "agent.yaml",
                "tools.json",
                "environment.json",
                "data-manifest.json",
            ):
                path = temporary / name
                path.write_text(name, encoding="utf-8")
                files[name] = path
            output = temporary / "output"
            output.mkdir()
            environment = {
                "GRPO_TRAIN_FILE": str(files["train.parquet"]),
                "GRPO_VAL_FILE": str(files["validation.parquet"]),
                "SHOPPING_AGENT_LOOP_CONFIG": str(files["agent.yaml"]),
                "SHOPPING_TOOL_CONFIG": str(files["tools.json"]),
                "SHOPPING_ENV_MANIFEST": str(files["environment.json"]),
                "SHOPPING_GRPO_DATA_MANIFEST": str(files["data-manifest.json"]),
                "GRPO_MODEL_PATH": str(model),
                "GRPO_OUTPUT_DIR": str(output),
                "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
                "SHOP_REWARD_VERSION": "shopsimulator-reward-v4",
                "GRPO_SEED": "20260823",
                "SHOPPING_REWARD_SHAPING_PROFILE": "none",
                "SHOPPER_MODEL": "shopper",
                "SHOPPER_BASE_URL": "https://shopper.example.test/v1",
                "SHOPPER_API_KEY": "must-not-be-written",
                "CUDA_VISIBLE_DEVICES": "0,2,3,4",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "SHOPPING_ENV_CONCURRENCY_PER_WORKER": "2",
                "SHOPPING_EXPECTED_SHOPSIM_SLOTS": "20",
            }
            audit = {
                "command": ["python", "-m", "verl.trainer.main_ppo"],
                "config": str(files["grpo.yaml"]),
                "reward_profile": "none",
            }
            with patch(
                "scripts.train_grpo.subprocess.check_output",
                side_effect=["abc123\n", b"?? local-artifact", b"local diff"],
            ), patch("scripts.train_grpo._patch_inventory", return_value=[]):
                destination = write_run_contract(audit, environment)

            contract = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(contract["git"]["commit"], "abc123")
        self.assertTrue(contract["git"]["dirty"])
        self.assertEqual(
            contract["runtime_contract"]["reward_profile"],
            "none",
        )
        self.assertEqual(
            contract["runtime_contract"]["cuda_physical_to_logical"],
            {"0": 0, "2": 1, "3": 2, "4": 3},
        )
        self.assertTrue(
            contract["runtime_contract"]["ray_preserves_cuda_visible_devices"]
        )
        self.assertEqual(
            contract["runtime_contract"]["environment_concurrency_per_worker"], 2
        )
        self.assertEqual(contract["runtime_contract"]["expected_shopsim_slots"], 20)
        self.assertIn("sha256", contract["inputs"]["train_data"])
        self.assertIn("sha256", contract["inputs"]["data_manifest"])
        self.assertEqual(len(contract["inputs"]["model_artifacts"]), 1)
        self.assertIn("sha256", contract["inputs"]["model_artifacts"][0])
        self.assertNotIn("must-not-be-written", json.dumps(contract))

    def test_grpo_preflight_only_never_launches_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            data_manifest = temporary / "manifest.json"
            data_manifest.write_text("{}", encoding="utf-8")
            output = temporary / "output"
            argv = [
                "train_grpo.py",
                "--model",
                str(model),
                "--train-data",
                str(train),
                "--val-data",
                str(validation),
                "--data-manifest",
                str(data_manifest),
                "--output",
                str(output),
                "--config",
                str(root / "configs/grpo.yaml"),
                "--shopper-base-url",
                "https://shopper.example.test/v1",
                "--preflight-only",
                "--",
                "trainer.total_training_steps=1",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.dict(
                    os.environ,
                    {"SHOPPER_API_KEY": "secret", "SWANLAB_API_KEY": "swan-secret"},
                ),
                patch("scripts.train_grpo.subprocess.call", return_value=0) as call,
                patch(
                    "scripts.train_grpo.validate_grpo_data_manifest",
                    return_value={},
                ),
                patch("builtins.print"),
            ):
                grpo_main()

        self.assertEqual(call.call_count, 1)
        preflight = call.call_args.args[0]
        self.assertEqual(
            Path(preflight[1]).resolve(),
            root / "scripts/check_grpo_runtime.py",
        )
        self.assertIn("trainer.logger=[console,swanlab]", preflight)
        self.assertIn("trainer.experiment_name=shopping-agent-grpo", preflight)
        self.assertIn("trainer.total_training_steps=1", preflight)

    def test_grpo_resume_is_explicit_and_confined_to_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            train.write_bytes(b"train")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"validation")
            manifest = temporary / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            output = temporary / "output"
            checkpoint = output / "global_step_50"
            checkpoint.mkdir(parents=True)
            (output / "run_contract.json").write_text("{}", encoding="utf-8")
            argv = [
                "train_grpo.py",
                "--model",
                str(model),
                "--train-data",
                str(train),
                "--val-data",
                str(validation),
                "--data-manifest",
                str(manifest),
                "--output",
                str(output),
                "--resume-from-checkpoint",
                str(checkpoint),
                "--dry-run",
            ]
            with patch.object(sys, "argv", argv):
                args = parse_args()
            with patch(
                "scripts.train_grpo.validate_grpo_data_manifest",
                return_value={},
            ):
                command, environment = build_command(args)

        self.assertIn("trainer.resume_mode=resume_path", command)
        self.assertIn(
            f"trainer.resume_from_path={checkpoint.resolve()}",
            command,
        )
        self.assertEqual(
            environment["GRPO_RESUME_FROM_CHECKPOINT"],
            str(checkpoint.resolve()),
        )


if __name__ == "__main__":
    unittest.main()
