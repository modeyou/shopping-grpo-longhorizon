#!/usr/bin/env python3
"""Apply/check/restore the tracking-finish patch to pinned veRL 0.8."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import sys
from pathlib import Path

from shopping_grpo.training.grpo.tracking_patch import (
    LEGACY_PATCH_MARKERS,
    PATCH_MARKER,
    patch_source,
)

EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_ORIGINAL_SHA256 = "a96d48404c53425d4c6f44eb164e72d7a55edfea3337e4279ad9fd8f4695db77"
BACKUP_SUFFIX = ".shopping-grpo-tracking.orig"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_target():
    if importlib.metadata.version("verl") != EXPECTED_VERL_VERSION:
        raise RuntimeError(f"tracking patch requires verl=={EXPECTED_VERL_VERSION}")
    import verl

    root = Path(verl.__file__).resolve().parent
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"verl is outside the active environment: {root}")
    target = root / "utils" / "tracking.py"
    if not target.is_file():
        raise RuntimeError(f"veRL tracking source is missing: {target}")
    return target


def expected_patched_sha256(target):
    target = Path(target).resolve()
    backup = Path(str(target) + BACKUP_SUFFIX)
    original = next(
        (item for item in (backup, target) if item.is_file() and sha256(item) == EXPECTED_ORIGINAL_SHA256),
        None,
    )
    if original is None:
        raise RuntimeError("verified original veRL tracking.py is unavailable")
    patched = patch_source(original.read_text(encoding="utf-8"))
    return hashlib.sha256(patched.encode("utf-8")).hexdigest()


def verify(target):
    target = Path(target).resolve()
    expected = expected_patched_sha256(target)
    actual = sha256(target)
    if actual != expected:
        raise RuntimeError(f"patched tracking.py hash mismatch: expected {expected}, got {actual}")
    source = target.read_text(encoding="utf-8")
    finish_contract = (
        '    def finish(self):\n'
        '        """Finish tracking once, from the trainer thread when possible."""\n'
    )
    dashboard_contract = '                if default_backend == "swanlab":\n'
    if (
        source.count(PATCH_MARKER) != 1
        or source.count(finish_contract) != 1
        or source.count(dashboard_contract) != 1
    ):
        raise RuntimeError("veRL Tracking dashboard/finish contract is missing")
    py_compile.compile(str(target), doraise=True)


def apply(target):
    target = Path(target).resolve()
    backup = Path(str(target) + BACKUP_SUFFIX)
    actual = sha256(target)
    expected = expected_patched_sha256(target)
    if actual == expected:
        verify(target)
        print(f"veRL tracking finish patch already applied: {target}")
        return
    source = target.read_text(encoding="utf-8")
    if any(marker in source for marker in LEGACY_PATCH_MARKERS):
        if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
            raise RuntimeError(
                "cannot upgrade legacy tracking patch without its verified original backup"
            )
        shutil.copy2(backup, target)
        actual = EXPECTED_ORIGINAL_SHA256
        print(f"upgrading legacy veRL tracking patch from verified backup: {target}")
    if actual != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing to patch unknown tracking.py: expected {EXPECTED_ORIGINAL_SHA256}, got {actual}"
        )
    if backup.exists() and sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"invalid tracking patch backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)
    temporary = target.with_name(target.name + ".shopping-tracking.tmp")
    temporary.write_text(
        patch_source(target.read_text(encoding="utf-8")), encoding="utf-8", newline="\n"
    )
    try:
        py_compile.compile(str(temporary), doraise=True)
        temporary.replace(target)
        verify(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        shutil.copy2(backup, target)
        raise
    print(f"applied veRL tracking finish patch: {target}")
    print(f"backup: {backup}")


def restore(target):
    target = Path(target).resolve()
    if sha256(target) == EXPECTED_ORIGINAL_SHA256:
        print(f"veRL tracking source is already original: {target}")
        return
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"verified tracking patch backup is missing: {backup}")
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
            print(f"verified veRL tracking finish patch: {target}")
        else:
            apply(target)
    except (OSError, RuntimeError, ValueError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL tracking patch error: {exc}") from exc


if __name__ == "__main__":
    main()
