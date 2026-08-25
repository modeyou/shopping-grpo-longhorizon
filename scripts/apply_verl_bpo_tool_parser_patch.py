#!/usr/bin/env python3
"""Apply/check/restore the anchored veRL 0.8 Qwen XML parser patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import sys
from pathlib import Path

from shopping_grpo.training.bpo.xml_tool_parser_patch import PATCH_MARKER, patch_source


EXPECTED_VERL_VERSION = "0.8.0"
BACKUP_SUFFIX = ".shopping-bpo-xml.orig"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_target():
    if importlib.metadata.version("verl") != EXPECTED_VERL_VERSION:
        raise RuntimeError(f"BPO requires verl=={EXPECTED_VERL_VERSION}")
    import verl

    root = Path(verl.__file__).resolve().parent
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"verl is outside the active environment: {root}")
    target = root / "experimental/agent_loop/tool_parser.py"
    if not target.is_file():
        raise RuntimeError(f"veRL tool parser source is missing: {target}")
    return target


def expected_patched_sha256(target):
    backup = Path(str(Path(target).resolve()) + BACKUP_SUFFIX)
    if not backup.is_file():
        raise RuntimeError(f"verified XML parser backup is missing: {backup}")
    original = backup.read_text(encoding="utf-8")
    patched = patch_source(original)
    return hashlib.sha256(patched.encode("utf-8")).hexdigest()


def verify(target):
    expected = expected_patched_sha256(target)
    actual = sha256(target)
    if actual != expected:
        raise RuntimeError(
            f"patched tool_parser.py hash mismatch: expected {expected}, got {actual}"
        )
    source = Path(target).read_text(encoding="utf-8")
    if source.count(PATCH_MARKER) != 1:
        raise RuntimeError("BPO XML parser patch marker must occur exactly once")
    py_compile.compile(str(target), doraise=True)


def apply(target):
    target = Path(target).resolve()
    source = target.read_text(encoding="utf-8")
    backup = Path(str(target) + BACKUP_SUFFIX)
    if PATCH_MARKER in source:
        verify(target)
        print(f"veRL BPO XML parser patch already applied: {target}")
        return
    # Exact anchor validation is performed before preserving or changing bytes.
    patched = patch_source(source)
    if backup.exists() and backup.read_text(encoding="utf-8") != source:
        raise RuntimeError(f"refusing to overwrite mismatched XML parser backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)
    temporary = target.with_name(target.name + ".shopping-bpo-xml.tmp")
    temporary.write_text(patched, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(temporary), doraise=True)
        temporary.replace(target)
        verify(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        shutil.copy2(backup, target)
        raise
    print(f"applied veRL BPO XML parser patch: {target}")
    print(f"backup: {backup}")


def restore(target):
    target = Path(target).resolve()
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.is_file():
        raise RuntimeError(f"verified XML parser backup is missing: {backup}")
    # The backup must still satisfy the exact original anchor contract.
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
            print(f"verified veRL BPO XML parser patch: {target}")
        else:
            apply(target)
    except (OSError, RuntimeError, ValueError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL BPO XML parser patch error: {exc}") from exc


if __name__ == "__main__":
    main()
