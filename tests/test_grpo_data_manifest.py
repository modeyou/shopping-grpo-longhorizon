import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.training.grpo.data_manifest import (
    DATASET_SCHEMA,
    REWARD_VERSION,
    deterministic_active_split,
    sha256_file,
    validate_grpo_data_manifest,
)


class GrpoDataManifestTest(unittest.TestCase):
    def test_active_split_is_deterministic_disjoint_and_validation_first(self):
        first = deterministic_active_split(
            set(range(20)),
            {2, 7},
            seed=20260823,
            train_count=10,
            validation_count=4,
        )
        second = deterministic_active_split(
            set(range(20)),
            {2, 7},
            seed=20260823,
            train_count=10,
            validation_count=4,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["validation"]), 4)
        self.assertEqual(len(first["train"]), 10)
        self.assertFalse(set(first["validation"]) & set(first["train"]))
        self.assertFalse(({2, 7} & set(first["validation"] + first["train"])))

    def test_manifest_binds_reward_environment_tasks_and_parquet_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            environment = data / "environment-v4.json"
            train = data / "train.parquet"
            validation = data / "validation.parquet"
            train_tasks = data / "train-tasks.jsonl"
            validation_tasks = data / "validation-tasks.jsonl"
            selection_source = data / "selection-manifest.json"
            reward_audit = data / "reward-audit.jsonl"
            reservoir = data / "reservoir.jsonl"
            opening = data / "openings.jsonl"
            exclusion = data / "excluded.jsonl"
            environment.write_text("environment", encoding="utf-8")
            train.write_bytes(b"train")
            validation.write_bytes(b"validation")
            train_tasks.write_text(
                '{"task_id": 1}\n{"task_id": 2}\n',
                encoding="utf-8",
            )
            validation_tasks.write_text('{"task_id": 3}\n', encoding="utf-8")
            selection_source.write_text("selection", encoding="utf-8")
            reward_audit.write_text(
                '{"task_id": 1, "eligible": true}\n',
                encoding="utf-8",
            )
            reservoir.write_text('{"task_id": 1}\n', encoding="utf-8")
            opening.write_text('{"task_id": 1}\n', encoding="utf-8")
            exclusion.write_text('{"task_id": 9}\n', encoding="utf-8")

            def detail(path, **extra):
                return {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    **extra,
                }

            manifest = {
                "schema_version": DATASET_SCHEMA,
                "status": "accepted",
                "reward_version": REWARD_VERSION,
                "environment": detail(environment),
                "selection_source": detail(selection_source),
                "reward_reachability_audit": detail(reward_audit, rows=1),
                "source_reservoir": detail(reservoir),
                "selection": {
                    "train": detail(train_tasks, tasks=2),
                    "validation": detail(validation_tasks, tasks=1),
                },
                "artifacts": {
                    "train": detail(train, rows=4, tasks=2),
                    "validation": detail(validation, rows=2, tasks=1),
                },
                "openings": {"train_gap": detail(opening)},
                "exclusions": [
                    {"label": "final", **detail(exclusion, tasks=1)}
                ],
                "audit": {
                    "train_validation_overlap_count": 0,
                    "selected_exclusion_overlap_count": 0,
                    "all_selected_tasks_reward_reachable": True,
                },
            }
            manifest_path = data / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            accepted = validate_grpo_data_manifest(
                manifest_path,
                train_data=train,
                validation_data=validation,
                environment_manifest=environment,
                root=root,
            )
            self.assertEqual(accepted["status"], "accepted")

            reward_audit.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Reward v4 reachability audit hash mismatch"
            ):
                validate_grpo_data_manifest(
                    manifest_path,
                    train_data=train,
                    validation_data=validation,
                    environment_manifest=environment,
                    root=root,
                )
            reward_audit.write_text(
                '{"task_id": 1, "eligible": true}\n',
                encoding="utf-8",
            )

            train.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "train parquet hash mismatch"):
                validate_grpo_data_manifest(
                    manifest_path,
                    train_data=train,
                    validation_data=validation,
                    environment_manifest=environment,
                    root=root,
                )


if __name__ == "__main__":
    unittest.main()
