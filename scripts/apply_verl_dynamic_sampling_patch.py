#!/usr/bin/env python3
"""Apply or restore the pinned veRL 0.8 dynamic-sampling patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_ORIGINAL_SHA256 = "de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66"
SUPERSEDED_PATCHED_SHA256S = frozenset(
    {
        "a2132ecbce6ca55fcd3a61f615b925b4a0c7a2192c69cd3e4faf8046124b334b",
        "9fc8bff440199e062236fc86f2a6f01eae4238e3cd8026f87e88d9c93fc4da82",
        "684b491e20ba9d41e91d5010186d4d08b01a01fc67f8a77d17c086b0381e00a3",
        "fc3564cc5680a9fa92ca7b0a9bc3ae87ccdc90c498ab1bfe34c6796d6c54fb5a",
        "c0c38f8c7c5d1f0376c3c0a236864a334e0466176410f6e6fa3cbf66524c8435",
        "19b9bc29e0bda8a6aa56e63199cc5737ae77c52ba41b92eaa9d15a38f24a32b8",
    }
)
PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V4"
BACKUP_SUFFIX = ".shopping-grpo-dynamic-sampling.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"
BPO_COMPATIBILITY_OLD = '''            if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                raise ValueError("shopping_dynamic_sampling only supports Vanilla GRPO")
'''
BPO_COMPATIBILITY_NEW = '''            dynamic_adv_estimator = (
                str(self.config.algorithm.adv_estimator)
                .rsplit(".", 1)[-1]
                .lower()
            )
            if (
                self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO
                and dynamic_adv_estimator != "bpo"
            ):
                raise ValueError(
                    "shopping_dynamic_sampling only supports GRPO or BPO"
                )
'''

HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


def load_tracking_patcher():
    """Load the sibling tracking lifecycle installer by path."""
    path = Path(__file__).resolve().with_name(
        "apply_verl_tracking_finish_patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_shopping_grpo_tracking_patcher", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load veRL tracking patcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unified_diff_hunks(path: Path) -> None:
    """Reject stale hunk counts/offsets before GNU patch can misplace code."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cumulative_delta = 0
    index = 0
    hunk_count = 0
    while index < len(lines):
        match = HUNK_HEADER_RE.match(lines[index])
        if match is None:
            index += 1
            continue
        hunk_count += 1
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) is not None else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) is not None else 1
        actual_old = 0
        actual_new = 0
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if line.startswith(("--- ", "+++ ")):
                break
            if line.startswith((" ", "-")):
                actual_old += 1
            if line.startswith((" ", "+")):
                actual_new += 1
            index += 1
        if (old_count, new_count) != (actual_old, actual_new):
            raise RuntimeError(
                "malformed dynamic-sampling patch hunk counts: "
                f"{match.group(0)!r} declares {old_count}/{new_count}, "
                f"contains {actual_old}/{actual_new}"
            )
        expected_new_start = old_start + cumulative_delta
        if old_count == 0:
            expected_new_start += 1
        if new_start != expected_new_start:
            raise RuntimeError(
                "malformed dynamic-sampling patch hunk offset: "
                f"{match.group(0)!r} targets {new_start}, "
                f"expected {expected_new_start}"
            )
        cumulative_delta += new_count - old_count
    if hunk_count == 0:
        raise RuntimeError(f"dynamic-sampling patch contains no hunks: {path}")


def resolve_installed_ray_trainer() -> Path:
    installed_version = importlib.metadata.version("verl")
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
        )

    import verl

    verl_source = Path(verl.__file__).resolve()
    expected_environment = Path(sys.prefix).resolve()
    if not verl_source.is_relative_to(expected_environment):
        raise RuntimeError(f"verl.__file__ is not from the project environment: {verl_source}")

    target = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not target.is_file():
        raise RuntimeError(f"installed ray_trainer.py does not exist: {target}")
    return target.resolve()


def validate_runtime_and_target(target_override: Path | None) -> Path:
    installed_target = resolve_installed_ray_trainer()
    if target_override is None:
        return installed_target
    target = target_override.resolve()
    if not target.is_file():
        raise RuntimeError(f"target ray_trainer.py does not exist: {target}")
    return target


def verify_patched(target: Path) -> None:
    target_hash = sha256(target)
    expected_hash = expected_patched_sha256(target)
    if target_hash != expected_hash:
        raise RuntimeError(
            "patched ray_trainer.py hash mismatch: "
            f"expected {expected_hash}, got {target_hash}"
        )
    source = target.read_text(encoding="utf-8")
    if PATCH_MARKER not in source:
        raise RuntimeError(f"patched ray_trainer.py is missing marker {PATCH_MARKER}")
    if BPO_COMPATIBILITY_OLD in source or source.count(BPO_COMPATIBILITY_NEW) != 1:
        raise RuntimeError("patched ray_trainer.py is missing BPO compatibility")
    py_compile.compile(str(target), doraise=True)


def add_bpo_compatibility_source(source: str) -> str:
    """Return the BPO-compatible form of the pinned dynamic-sampling source."""
    if source.count(BPO_COMPATIBILITY_OLD) != 1:
        raise RuntimeError(
            "the pinned dynamic-sampling patch has an unexpected estimator guard"
        )
    return source.replace(BPO_COMPATIBILITY_OLD, BPO_COMPATIBILITY_NEW, 1)


def add_bpo_compatibility(target: Path) -> None:
    """Extend the pinned GRPO patch without weakening its source/hash checks."""
    upgraded = add_bpo_compatibility_source(target.read_text(encoding="utf-8"))
    temporary = target.with_name(target.name + ".shopping-bpo-compat.tmp")
    shutil.copy2(target, temporary)
    try:
        temporary.write_text(upgraded, encoding="utf-8", newline="\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def expected_patched_sha256(target: Path) -> str:
    """Derive the exact patched hash from the verified veRL original."""
    backup = Path(str(target) + BACKUP_SUFFIX)
    original = None
    for candidate in (backup, target):
        if candidate.is_file() and sha256(candidate) == EXPECTED_ORIGINAL_SHA256:
            original = candidate
            break
    if original is None:
        raise RuntimeError(
            "cannot derive patched hash without the verified original ray_trainer.py"
        )
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")
    validate_unified_diff_hunks(PATCH_FILE)
    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError("required system 'patch' executable is unavailable")
    with tempfile.TemporaryDirectory(prefix="shopping-grpo-patch-") as directory:
        derived = Path(directory) / "ray_trainer.py"
        shutil.copy2(original, derived)
        subprocess.run(
            [
                patch_program,
                "--batch",
                "--forward",
                "--silent",
                str(derived),
                str(PATCH_FILE),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        derived.write_text(
            add_bpo_compatibility_source(derived.read_text(encoding="utf-8")),
            encoding="utf-8",
            newline="\n",
        )
        py_compile.compile(str(derived), doraise=True)
        return sha256(derived)


def apply_patch(target: Path) -> None:
    target_hash = sha256(target)
    backup = Path(str(target) + BACKUP_SUFFIX)
    has_verified_original = (
        target_hash == EXPECTED_ORIGINAL_SHA256
        or (backup.is_file() and sha256(backup) == EXPECTED_ORIGINAL_SHA256)
    )
    if not has_verified_original:
        if target_hash in SUPERSEDED_PATCHED_SHA256S:
            raise RuntimeError(
                "cannot upgrade the previous patch without its verified original backup"
            )
        raise RuntimeError(
            "refusing to patch unknown ray_trainer.py: "
            f"expected original SHA256 {EXPECTED_ORIGINAL_SHA256}, got {target_hash}"
        )
    expected_hash = expected_patched_sha256(target)
    if target_hash == expected_hash:
        verify_patched(target)
        print(f"veRL dynamic-sampling patch already applied: {target}")
        return
    if target_hash in SUPERSEDED_PATCHED_SHA256S:
        if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
            raise RuntimeError(
                "cannot upgrade the previous patch without its verified original backup"
            )
        shutil.copy2(backup, target)
        target_hash = EXPECTED_ORIGINAL_SHA256
    if target_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to patch unknown ray_trainer.py: "
            f"expected original SHA256 {EXPECTED_ORIGINAL_SHA256}, got {target_hash}"
        )
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")

    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError("required system 'patch' executable is unavailable")

    if backup.exists() and sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"refusing to overwrite invalid backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)

    rollback_source = backup

    try:
        subprocess.run(
            [patch_program, "--batch", "--forward", "--silent", str(target), str(PATCH_FILE)],
            check=True,
            cwd=PROJECT_ROOT,
        )
        add_bpo_compatibility(target)
        verify_patched(target)
    except Exception:
        shutil.copy2(rollback_source, target)
        raise

    print(f"applied veRL dynamic-sampling patch: {target}")
    print(f"backup: {backup}")
    print(f"patched_sha256: {sha256(target)}")


def restore_patch(target: Path) -> None:
    backup = Path(str(target) + BACKUP_SUFFIX)
    target_hash = sha256(target)
    if target_hash == EXPECTED_ORIGINAL_SHA256:
        print(f"veRL ray_trainer.py is already original: {target}")
        return
    if not backup.is_file():
        raise RuntimeError(f"cannot restore without backup: {backup}")
    backup_hash = sha256(backup)
    if backup_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing invalid backup: expected {EXPECTED_ORIGINAL_SHA256}, got {backup_hash}"
        )

    restore_temp = target.with_name(target.name + ".shopping-grpo-restore.tmp")
    shutil.copy2(backup, restore_temp)
    restore_temp.replace(target)
    if sha256(target) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"restore verification failed: {target}")
    py_compile.compile(str(target), doraise=True)
    print(f"restored original veRL ray_trainer.py: {target}")
    print(f"original_sha256: {sha256(target)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore the verified original file from the automatic backup",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the target is already patched without modifying it",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="override ray_trainer.py target for isolated patch-script tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.restore, args.check)) > 1:
        raise SystemExit("--restore and --check are mutually exclusive")
    try:
        target = validate_runtime_and_target(args.target)
        if args.restore:
            restore_patch(target)
        elif args.check:
            verify_patched(target)
            print(f"verified veRL dynamic-sampling patch: {target}")
        else:
            apply_patch(target)
        if args.target is None:
            tracking_patch = load_tracking_patcher()
            tracking_target = tracking_patch.resolve_target()
            if args.restore:
                tracking_patch.restore(tracking_target)
            elif args.check:
                tracking_patch.verify(tracking_target)
                print(
                    "verified veRL tracking finish patch: "
                    f"{tracking_target}"
                )
            else:
                tracking_patch.apply(tracking_target)
    except (OSError, RuntimeError, subprocess.CalledProcessError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL dynamic-sampling patch error: {exc}") from exc


if __name__ == "__main__":
    main()
