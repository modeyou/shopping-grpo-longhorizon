#!/usr/bin/env python3
"""Freeze complete openings and G+/G-/C+ mappings for an evaluated task set."""

import argparse
import json
import os
from pathlib import Path

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.multiturn.evaluation_openings import (
    freeze_evaluation_openings,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--gap-openings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = freeze_evaluation_openings(
        task_manifest=args.tasks,
        gap_openings=args.gap_openings,
        output_dir=args.output_dir,
        env_factory=ShopAgentEnv,
        base_url=args.base_url,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
