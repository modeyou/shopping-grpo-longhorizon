#!/usr/bin/env python3
"""Apply/check/restore the exact-entropy patch to pinned veRL 0.8."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import sys
from pathlib import Path

from shopping_grpo.training.bpo.entropy_patch import (
    LEGACY_PATCH_MARKERS,
    PATCH_MARKER,
    patch_source,
)

EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_ORIGINAL_SHA256 = "c7aafaa923edb7ab19c6a3d147643013be687df76d79ef38e855958d8382c68c"
BACKUP_SUFFIX = ".shopping-bpo-entropy.orig"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_patched_sha256(target):
    """Derive the one accepted V2 hash from the verified pinned source backup."""
    backup = Path(str(Path(target).resolve()) + BACKUP_SUFFIX)
    if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"verified BPO patch backup is missing: {backup}")
    patched = patch_source(backup.read_text(encoding="utf-8"))
    return hashlib.sha256(patched.encode("utf-8")).hexdigest()


def resolve_target():
    if importlib.metadata.version("verl") != EXPECTED_VERL_VERSION:
        raise RuntimeError(f"BPO requires verl=={EXPECTED_VERL_VERSION}")
    import verl

    root = Path(verl.__file__).resolve().parent
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"verl is outside the active environment: {root}")
    target = root / "workers/rollout/vllm_rollout/vllm_async_server.py"
    if not target.is_file():
        raise RuntimeError(f"veRL vLLM server source is missing: {target}")
    return target


def verify(target):
    actual = sha256(target)
    expected = expected_patched_sha256(target)
    if actual != expected:
        raise RuntimeError(
            "patched vllm_async_server.py hash mismatch: "
            f"expected {expected}, got {actual}"
        )
    source = target.read_text(encoding="utf-8")
    if PATCH_MARKER not in source:
        raise RuntimeError(f"BPO patch marker is missing: {PATCH_MARKER}")
    if source.count(PATCH_MARKER) != 1:
        raise RuntimeError("BPO patch marker must occur exactly once")
    py_compile.compile(str(target), doraise=True)


def apply(target):
    target = Path(target).resolve()
    source = target.read_text(encoding="utf-8")
    backup = Path(str(target) + BACKUP_SUFFIX)
    has_current_marker = PATCH_MARKER in source
    has_legacy_marker = any(marker in source for marker in LEGACY_PATCH_MARKERS)
    if has_current_marker:
        try:
            verify(target)
        except RuntimeError:
            pass
        else:
            print(f"veRL BPO entropy patch already applied: {target}")
            return
    if has_current_marker or has_legacy_marker:
        if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
            raise RuntimeError(
                "cannot upgrade an unknown BPO entropy patch without its "
                f"verified original backup: {backup}"
            )
        source = backup.read_text(encoding="utf-8")
        print(f"upgrading BPO entropy patch from verified backup: {target}")
    actual = sha256(target)
    if not (has_current_marker or has_legacy_marker) and actual != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to patch unknown vllm_async_server.py: "
            f"expected {EXPECTED_ORIGINAL_SHA256}, got {actual}"
        )
    if backup.exists() and sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"invalid BPO patch backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)
    patched = patch_source(source)
    temporary = target.with_name(target.name + ".shopping-bpo.tmp")
    temporary.write_text(patched, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(temporary), doraise=True)
        temporary.replace(target)
        verify(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        shutil.copy2(backup, target)
        raise
    print(f"applied veRL BPO entropy patch: {target}")
    print(f"backup: {backup}")


def restore(target):
    target = Path(target).resolve()
    if sha256(target) == EXPECTED_ORIGINAL_SHA256:
        print(f"veRL BPO entropy source is already original: {target}")
        return
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"verified BPO patch backup is missing: {backup}")
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
            print(f"verified veRL BPO entropy patch: {target}")
        else:
            apply(target)
    except (OSError, RuntimeError, ValueError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL BPO patch error: {exc}") from exc


if __name__ == "__main__":
    main()
