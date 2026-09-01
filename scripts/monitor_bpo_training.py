#!/usr/bin/env python3
"""Continuously summarize an active CARL-BPO diagnostics JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from shopping_grpo.training.bpo.live_monitor import (
    BpoLiveMonitor,
    aggregate_records,
    read_complete_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _print_compact(payload: dict) -> None:
    progress = payload["progress"]
    sampling = payload["sampling"]
    selected = payload["tasks"]["selected_task"]
    token_mass = payload["token_mass"]
    alerts = payload["alerts"]
    print(
        json.dumps(
            {
                "step": progress["latest_generation_step"],
                "accepted_groups": sampling["accepted_groups_total"],
                "accepted_sibling_terminal_outcomes": sampling[
                    "accepted_sibling_terminal_outcomes_total"
                ],
                "selected_unique_tasks": selected["unique"],
                "selected_task_repeat_rate": selected["repeat_rate"],
                "exact_token_mass": token_mass["exact_actor_tokens_available"],
                "action_span_max_share": token_mass[
                    "selected_action_span_tokens"
                ]["max_share"],
                "slow_warnings": alerts["slow_full_batch_warning_total"],
                "blocking": alerts["blocking"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.once:
        payload = aggregate_records(read_complete_jsonl(args.diagnostics))
        _print_compact(payload)
        if args.output:
            _write_atomic(args.output, payload)
        return 0

    monitor = BpoLiveMonitor()
    pending = ""
    last_emit = time.monotonic()
    caught_up = False
    with args.diagnostics.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if chunk:
                pending += chunk
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    if line.strip():
                        monitor.consume(json.loads(line))
            now = time.monotonic()
            should_emit = (not chunk and not caught_up) or (
                caught_up and now - last_emit >= args.interval
            )
            if should_emit:
                payload = monitor.snapshot()
                _print_compact(payload)
                if args.output:
                    _write_atomic(args.output, payload)
                last_emit = now
                caught_up = True
            if not chunk:
                time.sleep(min(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
