import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_grpo import (
    DEFAULT_DATA_MANIFEST,
    DEFAULT_TRAIN_DATA,
    DEFAULT_VAL_DATA,
)
from shopping_grpo.training.sft.data_promotion import (
    REWARD_VERSION,
    SFT_DATA_SCHEMA,
    promote_formal_sft_data,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


class FormalDataLayoutTest(unittest.TestCase):
    def test_reference_and_formal_namespaces_are_unambiguous(self):
        forbidden = (
            ROOT / "data/sft/train.jsonl",
            ROOT / "data/sft/validation.jsonl",
            ROOT / "data/grpo/train.jsonl",
            ROOT / "data/grpo/validation.jsonl",
            ROOT / "data/grpo/train.parquet",
            ROOT / "data/grpo/validation.parquet",
        )
        self.assertFalse([str(path) for path in forbidden if path.exists()])
        for path in (
            ROOT / "data/reference/sft-v1/train.jsonl",
            ROOT / "data/reference/sft-v1/validation.jsonl",
            ROOT / "data/reference/grpo-v1/train.jsonl",
            ROOT / "data/reference/grpo-v1/validation.jsonl",
        ):
            self.assertTrue(path.is_file(), path)

        self.assertEqual(
            DEFAULT_TRAIN_DATA,
            ROOT / "data/grpo/formal-v2/multiturn-train.parquet",
        )
        self.assertEqual(
            DEFAULT_VAL_DATA,
            ROOT / "data/grpo/formal-v2/multiturn-validation.parquet",
        )
        self.assertEqual(
            DEFAULT_DATA_MANIFEST,
            ROOT / "data/grpo/formal-v2/manifest.json",
        )

    def test_frozen_split_reference_paths_still_exist(self):
        metadata = json.loads(
            (ROOT / "data/multiturn/tasks/metadata.json").read_text(
                encoding="utf-8"
            )
        )
        for detail in metadata["exclusions"]["sources"]:
            path = ROOT / detail["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(len(detail["sha256"]), 64)

    def test_formal_sft_promotion_is_byte_identical_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "outputs/source"
            destination = root / "data/sft/formal-v2"
            source.mkdir(parents=True)
            train = source / "train.jsonl"
            validation = source / "validation.jsonl"
            train.write_text('{"task_id": 1}\n', encoding="utf-8")
            validation.write_text('{"task_id": 2}\n', encoding="utf-8")
            manifest = {
                "schema_version": SFT_DATA_SCHEMA,
                "reward": REWARD_VERSION,
                "split": {"task_disjoint": True},
                "evaluation_exclusion": {"selected_overlap_count": 0},
                "artifacts": {
                    "train.jsonl": {
                        "rows": 1,
                        "sha256": sha256_file(train),
                    },
                    "validation.jsonl": {
                        "rows": 1,
                        "sha256": sha256_file(validation),
                    },
                },
            }
            manifest_path = source / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            expected_manifest_hash = sha256_file(manifest_path)

            promotion = promote_formal_sft_data(
                source,
                destination,
                repo_root=root,
                expected_manifest_sha256=expected_manifest_hash,
            )

            self.assertEqual(promotion["status"], "accepted")
            for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
                self.assertEqual(
                    (source / name).read_bytes(),
                    (destination / name).read_bytes(),
                )
            with self.assertRaisesRegex(ValueError, "new or empty"):
                promote_formal_sft_data(
                    source,
                    destination,
                    repo_root=root,
                    expected_manifest_sha256=expected_manifest_hash,
                )


if __name__ == "__main__":
    unittest.main()
