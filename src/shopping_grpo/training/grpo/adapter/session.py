"""把一条 veRL trajectory 绑定到一个 ShopSimulator 环境租约。

session 负责异步训练框架与同步 HTTP 客户端之间的边界：在线程中执行网络调用，
在 coroutine-local context 中暴露当前环境和运行状态，并在任何退出路径释放租约。
"""

from __future__ import annotations

import asyncio
import json

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.multiturn.tasks import source_goal_hash
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    current_shopper,
    make_runtime_state,
)
from shopping_grpo.training.grpo.adapter.shopper import ControlledShopper


class ShopSimulatorSession:
    """负责 reset、绑定 coroutine-local 状态，并保证 release。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5700",
        timeout: int = 60,
        max_steps: int = 35,
        required_environment_version: str | None = None,
        required_reward_version: str | None = None,
        multiturn_enable: bool = False,
        shopper_model: str | None = None,
        shopper_base_url: str | None = None,
        shopper_api_key: str | None = None,
        shopper_timeout: int = 120,
        shopper_max_tokens: int = 512,
        max_shopper_questions: int = 2,
        shopper_factory=None,
        env_factory=None,
    ):
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.required_environment_version = required_environment_version
        self.required_reward_version = required_reward_version
        self.multiturn_enable = bool(multiturn_enable)
        self.shopper_model = shopper_model
        self.shopper_base_url = shopper_base_url
        self.shopper_api_key = shopper_api_key
        self.shopper_timeout = int(shopper_timeout)
        self.shopper_max_tokens = int(shopper_max_tokens)
        self.max_shopper_questions = int(max_shopper_questions)
        self.shopper_factory = shopper_factory
        self.env_factory = env_factory or ShopAgentEnv
        self.env = None
        self.state = None
        self._environment_token = None
        self._state_token = None
        self._shopper_token = None

    async def start(self, task_id: int, *, multiturn_spec=None) -> dict:
        """启动一条 trajectory，并把首个 observation 放进运行状态。"""
        if self.env is not None:
            raise RuntimeError("ShopSimulator session has already started")
        env_kwargs = {"base_url": self.base_url, "timeout": self.timeout}
        if self.multiturn_enable:
            env_kwargs["multiturn"] = True
        self.env = self.env_factory(**env_kwargs)
        try:
            # ShopAgentEnv 使用阻塞 urllib；放到线程中不会阻塞 veRL 的事件循环。
            if self.multiturn_enable:
                initial = await asyncio.to_thread(
                    self.env.reset, int(task_id), initial_request=""
                )
            else:
                initial = await asyncio.to_thread(self.env.reset, int(task_id))
        except Exception:
            try:
                await asyncio.to_thread(self.env.release)
            finally:
                self.env = None
            raise

        mode = (multiturn_spec or {}).get("interaction_mode", "single")
        self.state = make_runtime_state(
            task_id=task_id,
            max_steps=self.max_steps,
            interaction_mode=mode,
        )
        actual_version = (
            initial.get("environment_version") if isinstance(initial, dict) else None
        )
        if (
            self.required_environment_version is not None
            and actual_version != self.required_environment_version
        ):
            try:
                await asyncio.to_thread(self.env.release)
            finally:
                self.env = None
            raise RuntimeError(
                "ShopSimulator environment version mismatch: "
                f"expected {self.required_environment_version!r}, got {actual_version!r}"
            )
        actual_reward_version = (
            initial.get("reward_version")
            if isinstance(initial, dict)
            else None
        )
        if (
            self.required_reward_version is not None
            and actual_reward_version != self.required_reward_version
        ):
            try:
                await asyncio.to_thread(self.env.release)
            finally:
                self.env = None
            raise RuntimeError(
                "ShopSimulator reward version mismatch: "
                f"expected {self.required_reward_version!r}, "
                f"got {actual_reward_version!r}"
            )
        self.state["environment_version"] = actual_version
        self.state["expected_reward_version"] = actual_reward_version
        shopper = None
        if self.multiturn_enable:
            if not isinstance(multiturn_spec, dict):
                await self._release_after_start_failure()
                raise ValueError("multiturn session requires validated opening metadata")
            private_context = getattr(self.env, "shopper_context", None)
            if not isinstance(private_context, dict):
                await self._release_after_start_failure()
                raise RuntimeError("multiturn reset did not return private shopper context")
            if source_goal_hash(private_context) != multiturn_spec["source_goal_hash"]:
                await self._release_after_start_failure()
                raise RuntimeError("multiturn source goal hash mismatch")
            private_text = private_context["instruction_full"] + json.dumps(
                private_context.get("goal_options") or [], ensure_ascii=False
            )
            if any(fact not in private_text for fact in multiturn_spec["omitted_facts"]):
                await self._release_after_start_failure()
                raise RuntimeError("opening omitted fact is not grounded in the private goal")
            try:
                shopper = self._build_shopper(multiturn_spec)
            except Exception:
                await self._release_after_start_failure()
                raise
            self.state["ask_shopper_enabled"] = True
            self.state["max_shopper_questions"] = self.max_shopper_questions
        if isinstance(initial, dict) and initial.get("observation_state") is not None:
            self.state["latest_observation"] = render_structured_observation(
                initial["observation_state"]
            )
        else:
            self.state["latest_observation"] = str(
                initial.get("instruction", initial.get("observation", ""))
                if isinstance(initial, dict)
                else initial
            )
        # ContextVar 让并发 trajectory 互不串状态，比共享全局 current_env 安全。
        self._environment_token = current_environment.set(self.env)
        self._state_token = current_runtime_state.set(self.state)
        if shopper is not None:
            self._shopper_token = current_shopper.set(shopper)
        return self.state

    def _build_shopper(self, spec):
        if self.shopper_factory is not None:
            return self.shopper_factory(
                initial_request=spec["initial_request"],
                allowed_facts=spec["omitted_facts"],
                max_questions=self.max_shopper_questions,
            )
        if not self.shopper_model or not self.shopper_base_url or not self.shopper_api_key:
            raise ValueError(
                "multiturn harness requires shopper_model, shopper_base_url, and shopper_api_key"
            )
        from shopping_grpo.evaluation.rollout import OpenAIChatClient

        client = OpenAIChatClient(
            model=self.shopper_model,
            base_url=self.shopper_base_url,
            api_key=self.shopper_api_key,
            temperature=0,
            timeout=self.shopper_timeout,
            max_tokens=self.shopper_max_tokens,
        )
        return ControlledShopper(
            client,
            initial_request=spec["initial_request"],
            allowed_facts=spec["omitted_facts"],
            max_questions=self.max_shopper_questions,
        )

    async def _release_after_start_failure(self):
        try:
            await asyncio.to_thread(self.env.release)
        finally:
            self.env = None

    async def close(self) -> None:
        """释放环境并恢复 ContextVar；释放失败仍会清理本地绑定。"""
        if self.env is None:
            return
        try:
            await asyncio.to_thread(self.env.release)
        except Exception as exc:
            if self.state is not None:
                self.state["error"] = f"release_error:{exc.__class__.__name__}:{exc}"
            raise
        finally:
            if self._shopper_token is not None:
                current_shopper.reset(self._shopper_token)
            if self._state_token is not None:
                current_runtime_state.reset(self._state_token)
            if self._environment_token is not None:
                current_environment.reset(self._environment_token)
            self.env = None
            self._state_token = None
            self._environment_token = None
            self._shopper_token = None
