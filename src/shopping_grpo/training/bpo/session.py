"""Coroutine-local bindings for cloned BPO ShopSimulator branches."""

from __future__ import annotations

import asyncio

from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    current_shopper,
)


class ClonedBranchSession:
    def __init__(self, source_env, snapshot_id, state, shopper):
        self.source_env = source_env
        self.snapshot_id = snapshot_id
        self.state = state
        self.shopper = shopper
        self.env = None
        self._tokens = []

    async def start(self):
        def state_flag(name):
            if isinstance(self.state, dict):
                return bool(self.state.get(name))
            return bool(getattr(self.state, name, False))

        if state_flag("done") or state_flag("terminate"):
            raise RuntimeError(
                "BPO cannot restore a branch from terminal runtime state"
            )
        self.env = await asyncio.to_thread(self.source_env.clone, self.snapshot_id)
        if bool(getattr(self.env, "done", False)):
            await asyncio.to_thread(self.env.release)
            self.env = None
            raise RuntimeError(
                "BPO snapshot restored a terminal environment at a "
                "non-terminal branch boundary"
            )
        self._tokens = [
            (current_environment, current_environment.set(self.env)),
            (current_runtime_state, current_runtime_state.set(self.state)),
            (current_shopper, current_shopper.set(self.shopper)),
        ]
        return self

    async def close(self):
        try:
            if self.env is not None:
                await asyncio.to_thread(self.env.release)
        finally:
            for variable, token in reversed(self._tokens):
                variable.reset(token)
            self._tokens = []
            self.env = None
