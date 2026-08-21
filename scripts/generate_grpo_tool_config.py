#!/usr/bin/env python3
"""Generate the veRL tool registry from the repository's canonical schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.environment.tools import MULTITURN_SHOP_TOOL_SCHEMAS


def build_config():
    return {
        "tools": [
            {
                "class_name": "shopping_grpo.training.grpo.adapter.tools.ShopSimulatorTool",
                "config": {"type": "native"},
                "tool_schema": schema,
            }
            for schema in MULTITURN_SHOP_TOOL_SCHEMAS
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/tools.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = json.dumps(build_config(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"tool config is stale: {args.output}")
        print(f"tool config is current: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(f"wrote {args.output} tools={len(MULTITURN_SHOP_TOOL_SCHEMAS)}")


if __name__ == "__main__":
    main()
