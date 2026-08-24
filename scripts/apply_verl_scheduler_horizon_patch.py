#!/usr/bin/env python3
"""Decouple a staged run's stop step from veRL's optimizer scheduler horizon."""

from __future__ import annotations

import argparse
import hashlib
import py_compile
import shutil
import sys
from pathlib import Path

if __package__:
    from scripts import apply_verl_dynamic_sampling_patch as dynamic_patch
else:  # Direct execution: the scripts directory, not the repository root, is sys.path[0].
    import apply_verl_dynamic_sampling_patch as dynamic_patch


PATCH_MARKER = "SHOPPING_GRPO_SCHEDULER_HORIZON_PATCH_V1"
EXPECTED_INPUT_SHA256 = dynamic_patch.EXPECTED_PATCHED_SHA256
BACKUP_SUFFIX = ".shopping-grpo-scheduler-horizon.orig"

OLD_SOURCE = '''                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
'''

NEW_SOURCE = '''                scheduler_config = self.config.get("shopping_scheduler", {})
                scheduler_total_training_steps = int(
                    scheduler_config.get("total_training_steps", total_training_steps)
                )
                if scheduler_total_training_steps < total_training_steps:
                    raise ValueError(
                        "shopping_scheduler.total_training_steps must be greater than or "
                        "equal to trainer.total_training_steps"
                    )
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        scheduler_total_training_steps
                    )
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = scheduler_total_training_steps
                print(
                    "SHOPPING_GRPO_SCHEDULER_HORIZON "
                    + json.dumps(
                        {
                            "stage_total_training_steps": int(total_training_steps),
                            "scheduler_total_training_steps": scheduler_total_training_steps,
                        },
                        sort_keys=True,
                    )
                )
                # SHOPPING_GRPO_SCHEDULER_HORIZON_PATCH_V1
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def patched_source(source: str) -> str:
    if source.count(OLD_SOURCE) != 1:
        raise RuntimeError("veRL scheduler assignment source does not match the pinned contract")
    if PATCH_MARKER in source:
        raise RuntimeError("scheduler-horizon patch marker exists in an unexpected source")
    return source.replace(OLD_SOURCE, NEW_SOURCE, 1)


def verify_patched(target: Path) -> str:
    source = target.read_text(encoding="utf-8")
    if source.count(NEW_SOURCE) != 1 or source.count(PATCH_MARKER) != 1:
        raise RuntimeError("scheduler-horizon patch is incomplete")
    recovered = source.replace(NEW_SOURCE, OLD_SOURCE, 1)
    recovered_sha256 = sha256_bytes(recovered.encode("utf-8"))
    if recovered_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            "scheduler-horizon patch base hash mismatch: "
            f"expected {EXPECTED_INPUT_SHA256}, got {recovered_sha256}"
        )
    py_compile.compile(str(target), doraise=True)
    return sha256(target)


def apply_patch(target: Path) -> None:
    source = target.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        digest = verify_patched(target)
        print(f"veRL scheduler-horizon patch already applied: {target}")
        print(f"patched_sha256: {digest}")
        return
    current_sha256 = sha256(target)
    if current_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            "refusing to patch unknown ray_trainer.py: expected dynamic-sampling "
            f"SHA256 {EXPECTED_INPUT_SHA256}, got {current_sha256}"
        )
    backup = Path(str(target) + BACKUP_SUFFIX)
    if backup.exists() and sha256(backup) != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"refusing invalid scheduler-horizon backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)
    updated = patched_source(source)
    temporary = target.with_name(target.name + ".shopping-scheduler-horizon.tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="\n")
        temporary.replace(target)
        digest = verify_patched(target)
    except Exception:
        shutil.copy2(backup, target)
        raise
    print(f"applied veRL scheduler-horizon patch: {target}")
    print(f"backup: {backup}")
    print(f"patched_sha256: {digest}")


def restore_patch(target: Path) -> None:
    if PATCH_MARKER not in target.read_text(encoding="utf-8"):
        if sha256(target) == EXPECTED_INPUT_SHA256:
            print(f"veRL scheduler-horizon patch is already restored: {target}")
            return
        raise RuntimeError("target is neither the pinned input nor a verified patched source")
    verify_patched(target)
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.is_file() or sha256(backup) != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"cannot restore without verified backup: {backup}")
    shutil.copy2(backup, target)
    print(f"restored scheduler-horizon base: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--target", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.restore and args.check:
        raise SystemExit("--restore and --check are mutually exclusive")
    try:
        target = dynamic_patch.validate_runtime_and_target(args.target)
        if args.restore:
            restore_patch(target)
        elif args.check:
            digest = verify_patched(target)
            print(f"verified veRL scheduler-horizon patch: {target}")
            print(f"patched_sha256: {digest}")
        else:
            apply_patch(target)
    except (OSError, RuntimeError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL scheduler-horizon patch error: {exc}") from exc


if __name__ == "__main__":
    main()
