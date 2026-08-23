#!/usr/bin/env python3
"""Promote the accepted multi-turn SFT files into data/sft/formal-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.training.sft.data_promotion import (
    FORMAL_SFT_MANIFEST_SHA256,
    promote_formal_sft_data,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            ROOT
            / "outputs/multiturn-sft/mix-formal-1800-v4-seed20260822"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / "data/sft/formal-v2",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=FORMAL_SFT_MANIFEST_SHA256,
    )
    args = parser.parse_args()
    try:
        result = promote_formal_sft_data(
            args.source,
            args.destination,
            repo_root=ROOT,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
