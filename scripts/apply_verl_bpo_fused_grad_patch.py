#!/usr/bin/env python3
"""Apply/check/restore the veRL 0.8 fused-PPO gradient backport."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import sys
from pathlib import Path

from shopping_grpo.training.bpo.fused_ppo_grad_patch import (
    PATCH_MARKER,
    patch_source,
)

EXPECTED_VERL_VERSION = "0.8.0"
BACKUP_SUFFIX = ".shopping-bpo-fused-grad.orig"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_target():
    if importlib.metadata.version("verl") != EXPECTED_VERL_VERSION:
        raise RuntimeError(f"BPO requires verl=={EXPECTED_VERL_VERSION}")
    import verl

    root = Path(verl.__file__).resolve().parent
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"verl is outside the active environment: {root}")
    target = root / "utils/experimental/torch_functional.py"
    if not target.is_file():
        raise RuntimeError(f"veRL fused-PPO source is missing: {target}")
    return target


def expected_patched_sha256(target):
    backup = Path(str(Path(target).resolve()) + BACKUP_SUFFIX)
    if not backup.is_file():
        raise RuntimeError(f"verified fused-PPO backup is missing: {backup}")
    patched = patch_source(backup.read_text(encoding="utf-8"))
    return hashlib.sha256(patched.encode("utf-8")).hexdigest()


def verify(target):
    target = Path(target).resolve()
    expected = expected_patched_sha256(target)
    actual = sha256(target)
    if actual != expected:
        raise RuntimeError(
            f"patched torch_functional.py hash mismatch: expected {expected}, got {actual}"
        )
    if target.read_text(encoding="utf-8").count(PATCH_MARKER) != 1:
        raise RuntimeError("BPO fused-PPO gradient marker must occur exactly once")
    py_compile.compile(str(target), doraise=True)


def apply(target):
    target = Path(target).resolve()
    source = target.read_text(encoding="utf-8")
    backup = Path(str(target) + BACKUP_SUFFIX)
    if PATCH_MARKER in source:
        verify(target)
        print(f"veRL BPO fused-PPO gradient patch already applied: {target}")
        return
    patched = patch_source(source)
    if backup.exists() and backup.read_text(encoding="utf-8") != source:
        raise RuntimeError(f"refusing to overwrite mismatched fused-PPO backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)
    temporary = target.with_name(target.name + ".shopping-bpo-fused-grad.tmp")
    temporary.write_text(patched, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(temporary), doraise=True)
        temporary.replace(target)
        verify(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        shutil.copy2(backup, target)
        raise
    print(f"applied veRL BPO fused-PPO gradient patch: {target}")
    print(f"backup: {backup}")


def restore(target):
    target = Path(target).resolve()
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.is_file():
        raise RuntimeError(f"verified fused-PPO backup is missing: {backup}")
    patch_source(backup.read_text(encoding="utf-8"))
    shutil.copy2(backup, target)
    py_compile.compile(str(target), doraise=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.check and args.restore:
        raise SystemExit("--check and --restore are mutually exclusive")
    try:
        target = args.target.resolve() if args.target else resolve_target()
        if args.restore:
            restore(target)
        elif args.check:
            verify(target)
            print(f"verified veRL BPO fused-PPO gradient patch: {target}")
        else:
            apply(target)
    except (OSError, RuntimeError, ValueError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL BPO fused-PPO gradient patch error: {exc}") from exc


if __name__ == "__main__":
    main()
