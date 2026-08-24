"""Full one-boundary Branching Policy Optimization trajectory generator."""

from __future__ import annotations

import asyncio
from copy import copy, deepcopy
import math
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState

from shopping_grpo.training.bpo.branching import (
    BranchCandidate,
    first_decision_token,
    select_branch_candidate,
)
from shopping_grpo.training.bpo.session import ClonedBranchSession
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.runtime import (
    current_runtime_state,
    current_shopper,
    multiturn_spec_from_kwargs,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession


def clone_agent_data(source):
    """Copy token/history state while retaining immutable tool registry references."""
    cloned = copy(source)
    for name, value in vars(source).items():
        if name == "_active_tools":
            setattr(cloned, name, value)
        else:
            setattr(cloned, name, deepcopy(value))
    cloned.request_id = uuid4().hex
    cloned.metrics = {}
    cloned.tool_calls = []
    return cloned


class ShoppingBPOAgentLoop(ShoppingToolAgentLoop):
    """Generate K sibling returns from one entropy-selected action boundary."""

    def __init__(
        self,
        *args,
        sibling_count=4,
        branch_count=1,
        entropy_probe="exact-full-vocabulary",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sibling_count = int(sibling_count)
        self.branch_count = int(branch_count)
        self.entropy_probe = str(entropy_probe)
        if self.sibling_count != 4:
            raise ValueError("formal BPO v1 requires sibling_count=4")
        if self.branch_count != 1:
            raise ValueError("formal BPO v1 requires exactly one branch boundary")
        if self.entropy_probe != "exact-full-vocabulary":
            raise ValueError("formal BPO v1 requires exact full-vocabulary entropy")
        if getattr(self, "reward_shaping_profile", "none") != "none":
            raise ValueError("formal BPO v1 uses native Reward v4 without shaping")

    async def _new_agent_data(self, kwargs):
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=self._get_mm_processor_kwargs(audios),
            metrics={},
            request_id=uuid4().hex,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        extra_info = kwargs.get("extra_info", {}) or {}
        selected_names = extra_info.get("tool_selection")
        if selected_names and self.tools:
            selected = {
                name: self.tools[name]
                for name in selected_names
                if name in self.tools
            }
            data._active_tools = selected
            data._active_tool_schemas = [
                tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                for tool in selected.values()
            ]
        else:
            data._active_tools = self.tools
            data._active_tool_schemas = self.tool_schemas
        return data

    def _make_output(self, data):
        response_ids = data.prompt_ids[-len(data.response_mask):]
        prompt_ids = data.prompt_ids[:len(data.prompt_ids) - len(data.response_mask)]
        multi_modal_data = {}
        for source, destination in (
            (data.image_data, "images"),
            (data.video_data, "videos"),
            (data.audio_data, "audios"),
        ):
            if source is not None:
                multi_modal_data[destination] = source
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=data.response_mask[:self.response_length],
            response_logprobs=(
                data.response_logprobs[:self.response_length]
                if data.response_logprobs else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=data.mm_processor_kwargs,
            num_turns=data.user_turns + data.assistant_turns + 1,
            metrics=data.metrics,
            routed_experts=(
                data.routed_experts[:len(prompt_ids) + self.response_length]
                if data.routed_experts is not None else None
            ),
            extra_fields=data.extra_fields,
        )
        output.extra_fields.update(
            {"turn_scores": data.turn_scores, "tool_rewards": data.tool_rewards}
        )
        return output

    async def _probe_entropy(self, prompt_ids, sampling_params):
        probe = dict(sampling_params)
        probe.update(
            {
                "max_tokens": 1,
                "logprobs": True,
                "bpo_entropy_probe": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            }
        )
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=list(prompt_ids),
            sampling_params=probe,
        )
        entropy = (output.extra_fields or {}).get("bpo_full_vocab_entropy")
        if entropy is None:
            raise RuntimeError(
                "BPO entropy probe metadata is missing; apply the pinned veRL BPO patch"
            )
        entropy = float(entropy)
        if not math.isfinite(entropy):
            raise RuntimeError(
                "BPO entropy probe produced NaN/Inf; verify the V2 entropy patch"
            )
        return entropy

    async def _drive(self, data, sampling_params, *, action_starts):
        state = AgentState.GENERATING
        at_restored_boundary = True
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                if at_restored_boundary:
                    at_restored_boundary = False
                else:
                    action_starts.append(len(data.response_mask))
                state = await self._handle_generating_state(data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(data)
            else:
                raise RuntimeError(f"invalid BPO continuation state: {state}")
        return data

    async def _run_clone(self, candidate, sampling_params, task_id, sibling_index):
        payload = candidate.payload
        shopper = payload["shopper"].clone()
        session = ClonedBranchSession(
            payload["source_env"],
            candidate.snapshot_id,
            deepcopy(payload["runtime_state"]),
            shopper,
        )
        await session.start()
        try:
            data = clone_agent_data(payload["agent_data"])
            action_starts = list(payload["action_starts"])
            await self._drive(data, sampling_params, action_starts=action_starts)
            output = self._finalize_shopping_output(
                self._make_output(data), current_runtime_state.get(), task_id
            )
            self._attach_bpo_metadata(
                output,
                candidate,
                action_starts,
                sibling_index,
                payload["group_id"],
            )
            return output
        finally:
            await session.close()

    @staticmethod
    def _attach_bpo_metadata(output, candidate, action_starts, sibling_index, group_id):
        output.extra_fields.update(
            {
                "bpo_group_id": group_id,
                "bpo_sibling_index": int(sibling_index),
                "bpo_branch_action": int(candidate.action_index),
                "bpo_branch_entropy": float(candidate.entropy),
                "bpo_action_token_starts": list(action_starts),
                "bpo_return_budget": 4,
            }
        )

    async def run_tree(self, sampling_params, **kwargs):
        """Return one backbone and three continuations from one shared prefix."""
        spec = multiturn_spec_from_kwargs(kwargs, enabled=True)
        task_id = spec["task_id"]
        session = ShopSimulatorSession(
            base_url=self.base_url,
            timeout=self.timeout,
            max_steps=self.max_steps,
            required_environment_version=self.required_environment_version,
            required_reward_version=self.required_reward_version,
            multiturn_enable=True,
            shopper_model=self.shopper_model,
            shopper_base_url=self.shopper_base_url,
            shopper_api_key=self.shopper_api_key,
            shopper_timeout=self.shopper_timeout,
            shopper_max_tokens=self.shopper_max_tokens,
            max_shopper_questions=self.max_shopper_questions,
            shopper_factory=self.shopper_factory,
            env_factory=self.env_factory,
        )
        state = await session.start(task_id, multiturn_spec=spec)
        source_env = session.env
        group_id = uuid4().hex
        candidate = None
        action_starts = []
        data = await self._new_agent_data(kwargs)
        current = await self._handle_pending_state(data, sampling_params)
        try:
            try:
                while current != AgentState.TERMINATED:
                    if current == AgentState.GENERATING:
                        action_index = len(action_starts)
                        action_starts.append(len(data.response_mask))
                        snapshot_id = await asyncio.to_thread(source_env.snapshot)
                        prefix_data = clone_agent_data(data)
                        prefix_state = deepcopy(state)
                        prefix_shopper = current_shopper.get().clone()
                        prompt_before = list(data.prompt_ids)
                        current = await self._handle_generating_state(
                            data, sampling_params
                        )
                        try:
                            pieces = self.tokenizer.convert_ids_to_tokens(
                                data.response_ids
                            )
                            token_offset = first_decision_token(pieces)
                            entropy = await self._probe_entropy(
                                prompt_before + data.response_ids[:token_offset],
                                sampling_params,
                            )
                        except ValueError:
                            await asyncio.to_thread(
                                source_env.drop_snapshot, snapshot_id
                            )
                        except Exception:
                            await asyncio.to_thread(
                                source_env.drop_snapshot, snapshot_id
                            )
                            raise
                        else:
                            proposed = BranchCandidate(
                                action_index=action_index,
                                token_offset=token_offset,
                                entropy=entropy,
                                snapshot_id=snapshot_id,
                                payload={
                                    "agent_data": prefix_data,
                                    "runtime_state": prefix_state,
                                    "shopper": prefix_shopper,
                                    "source_env": source_env,
                                    "action_starts": list(action_starts),
                                    "group_id": group_id,
                                },
                            )
                            selected = select_branch_candidate(
                                [
                                    value
                                    for value in (candidate, proposed)
                                    if value is not None
                                ]
                            )
                            rejected = proposed if selected is candidate else candidate
                            if rejected is not None:
                                await asyncio.to_thread(
                                    source_env.drop_snapshot, rejected.snapshot_id
                                )
                            candidate = selected
                    elif current == AgentState.PROCESSING_TOOLS:
                        current = await self._handle_processing_tools_state(data)
                    else:
                        raise RuntimeError(f"invalid BPO backbone state: {current}")
                if candidate is None:
                    raise RuntimeError(
                        "BPO backbone has no valid semantic decision boundary"
                    )
                backbone = self._finalize_shopping_output(
                    self._make_output(data), state, task_id
                )
                self._attach_bpo_metadata(
                    backbone, candidate, action_starts, 0, group_id
                )
            finally:
                await session.close()

            branches = await asyncio.gather(
                *[
                    self._run_clone(candidate, sampling_params, task_id, sibling)
                    for sibling in range(1, self.sibling_count)
                ]
            )
            return [backbone, *branches]
        finally:
            if candidate is not None:
                await asyncio.to_thread(
                    source_env.drop_snapshot, candidate.snapshot_id
                )
