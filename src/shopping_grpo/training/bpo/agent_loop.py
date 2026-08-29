"""Full one-boundary Branching Policy Optimization trajectory generator."""

from __future__ import annotations

import asyncio
from copy import copy, deepcopy
import hashlib
import math
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState

from shopping_grpo.training.bpo.branching import (
    BranchCandidate,
    retain_branch_candidates,
    select_nonterminal_branch_candidate,
    validate_tree_outputs,
)
from shopping_grpo.training.bpo.session import ClonedBranchSession
from shopping_grpo.training.bpo.reward import completion_aligned_train_return
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.runtime import (
    current_runtime_state,
    current_shopper,
    multiturn_spec_from_kwargs,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.dynamic_sampling import (
    BPO_DIAGNOSTIC_FIELDS,
    aggregate_bpo_tree_metrics,
    build_rollout_diagnostics,
    summarize_bpo_group_diagnostics,
)


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

def attach_bpo_tree_metrics(outputs):
    """Attach one validated tree summary to every sibling shopping record."""
    values = list(outputs)
    if not values:
        raise ValueError("BPO tree metrics require sibling outputs")
    group_ids = [str(output.extra_fields["bpo_group_id"]) for output in values]
    if len(set(group_ids)) != 1:
        raise ValueError("BPO tree metrics require one group id")
    shopping_infos = [output.extra_fields["shopping"] for output in values]
    aligned_fields = {
        name: [output.extra_fields[name] for output in values]
        for name in BPO_DIAGNOSTIC_FIELDS
        if all(name in output.extra_fields for output in values)
    }
    records = build_rollout_diagnostics(
        group_ids,
        shopping_infos,
        aligned_fields=aligned_fields,
    )
    summaries = summarize_bpo_group_diagnostics(records)
    rewards = []
    for output, shopping in zip(values, shopping_infos, strict=True):
        score = output.reward_score
        if score is None:
            score = shopping["reward"]["total"]
        rewards.append(float(score))
    metrics = aggregate_bpo_tree_metrics(
        summaries,
        [{"uid": group_ids[0], "rewards": rewards}],
    )
    for shopping in shopping_infos:
        shopping["bpo_metrics"] = dict(metrics)
    return metrics


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
            raise ValueError("CARL-BPO requires sibling_count=4")
        if self.branch_count != 1:
            raise ValueError("CARL-BPO requires exactly one branch boundary")
        if self.entropy_probe != "exact-full-vocabulary":
            raise ValueError("CARL-BPO requires exact full-vocabulary entropy")

    @staticmethod
    def _attach_training_return(output):
        """Expose the CARL score on outputs finalized by either run path."""
        shopping = output.extra_fields.get("shopping")
        if not isinstance(shopping, dict) or not isinstance(
            shopping.get("reward"), dict
        ):
            raise ValueError("CARL-BPO output is missing shopping reward diagnostics")
        train_return = completion_aligned_train_return(
            shopping["reward"], reward_type=shopping.get("reward_type")
        )
        shopping["train_return"] = train_return
        shopping["reward"]["train_return"] = train_return
        # veRL builds token_level_rewards from AgentLoopOutput.reward_score.  The
        # native Reward-v4 result remains in shopping["reward"] for evaluation.
        output.reward_score = train_return
        return output

    def _finalize_shopping_output(self, output, state, task_id):
        """Keep Reward v4 diagnostics and expose the CARL training score."""
        output = super()._finalize_shopping_output(output, state, task_id)
        return self._attach_training_return(output)

    @staticmethod
    def _stage_for_prefix(state):
        """Classify a local boundary from the actions already in the prefix."""
        steps = state.get("steps") or []
        last_tool = str(steps[-1].get("tool", "")) if steps else ""
        lowered = last_tool.lower()
        observation = str(state.get("latest_observation", "")).lower()
        if any(
            marker in observation
            for marker in ("被本地动作守卫拒绝", "error", "失败", "无效")
        ):
            return "search_recovery"
        if any(
            marker in observation
            for marker in ("颜色", "尺码", "尺寸", "容量", "套装", "select_option")
        ):
            return "option"
        if lowered in {"search_products", "back_to_search", "prev_page"}:
            return "product"
        if any(
            word in lowered
            for word in ("option", "variant", "select", "configure", "size", "color")
        ):
            return "option"
        return "product"

    @staticmethod
    def _retain_local_candidates(candidates):
        """Retain one high-entropy snapshot per semantic stage plus the global best."""
        values = list(candidates)
        best_by_stage = {}
        for candidate in values:
            current = best_by_stage.get(candidate.stage)
            if current is None or (
                float(candidate.entropy), -int(candidate.action_index)
            ) > (float(current.entropy), -int(current.action_index)):
                best_by_stage[candidate.stage] = candidate
        retained = list(best_by_stage.values())
        global_best = max(
            values,
            key=lambda item: (float(item.entropy), -int(item.action_index)),
            default=None,
        )
        if global_best is not None and global_best not in retained:
            retained.append(global_best)
        return retained

    @staticmethod
    def _attach_root_metadata(output, *, group_id, sibling_index, prompt_ids):
        prefix_digest = hashlib.sha256(
            ",".join(str(value) for value in prompt_ids).encode("ascii")
        ).hexdigest()
        response_digest = hashlib.sha256(
            ",".join(str(value) for value in output.response_ids).encode("ascii")
        ).hexdigest()
        output.extra_fields.update(
            {
                "bpo_group_id": group_id,
                "bpo_group_type": "root",
                "bpo_sibling_index": int(sibling_index),
                "bpo_branch_action": -1,
                "bpo_branch_entropy": 0.0,
                "bpo_action_token_starts": [0],
                "bpo_return_budget": 4,
                "bpo_env_idx": 0,
                "bpo_branch_prefix_sha256": prefix_digest,
                "bpo_branch_action_sha256": response_digest,
                "bpo_backbone_action_count": 1,
                "bpo_branch_relative_position": 0.0,
                "bpo_branch_prefix_steps": 0,
                "bpo_branch_prefix_shopper_calls": 0,
                "bpo_branch_prefix_environment_transitions": 0,
                "bpo_local_stage": "root",
                "bpo_local_stage_target": "root",
                "bpo_local_stage_fallback": False,
                "bpo_local_stage_unavailable": False,
            }
        )

    async def _run_root(self, sampling_params, **kwargs):
        """Generate four independent episode-level Root rollouts."""
        group_id = uuid4().hex
        outputs = []
        for sibling_index in range(self.sibling_count):
            output = await ShoppingToolAgentLoop.run(
                self, sampling_params, **kwargs
            )
            # The adapter's parent ``run`` owns a second finalization path and
            # therefore does not dispatch through this subclass override.
            self._attach_training_return(output)
            self._attach_root_metadata(
                output,
                group_id=group_id,
                sibling_index=sibling_index,
                prompt_ids=output.prompt_ids,
            )
            outputs.append(output)
        validate_tree_outputs(outputs, sibling_count=self.sibling_count)
        attach_bpo_tree_metrics(outputs)
        return outputs

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
                session.env.env_idx,
            )
            return output
        finally:
            await session.close()

    @staticmethod
    def _attach_bpo_metadata(
        output,
        candidate,
        action_starts,
        sibling_index,
        group_id,
        env_idx,
    ):
        branch_start = int(action_starts[int(candidate.action_index)])
        branch_end = (
            int(action_starts[int(candidate.action_index) + 1])
            if int(candidate.action_index) + 1 < len(action_starts)
            else len(output.response_ids)
        )
        prefix_ids = [int(value) for value in output.response_ids[:branch_start]]
        prefix_digest = hashlib.sha256(
            ",".join(str(value) for value in prefix_ids).encode("ascii")
        ).hexdigest()
        branch_action_ids = [
            int(value)
            for value, enabled in zip(
                output.response_ids[branch_start:branch_end],
                output.response_mask[branch_start:branch_end],
                strict=True,
            )
            if int(enabled) != 0
        ]
        branch_action_digest = hashlib.sha256(
            ",".join(str(value) for value in branch_action_ids).encode("ascii")
        ).hexdigest()
        output.extra_fields.update(
            {
                "bpo_group_id": group_id,
                "bpo_group_type": "local",
                "bpo_sibling_index": int(sibling_index),
                "bpo_branch_action": int(candidate.action_index),
                "bpo_branch_entropy": float(candidate.entropy),
                "bpo_action_token_starts": list(action_starts),
                "bpo_return_budget": 4,
                "bpo_env_idx": int(env_idx),
                "bpo_branch_prefix_sha256": prefix_digest,
                "bpo_branch_action_sha256": branch_action_digest,
                "bpo_backbone_action_count": int(
                    candidate.payload["backbone_action_count"]
                ),
                "bpo_branch_relative_position": float(
                    int(candidate.action_index)
                    / max(1, int(candidate.payload["backbone_action_count"]) - 1)
                ),
                "bpo_local_stage": str(candidate.stage),
                "bpo_local_stage_target": str(
                    candidate.payload.get("stage_target", "auto")
                ),
                "bpo_local_stage_fallback": bool(
                    candidate.payload.get("stage_fallback", False)
                ),
                "bpo_local_stage_unavailable": bool(
                    candidate.payload.get("stage_fallback", False)
                ),
                "bpo_branch_prefix_steps": int(
                    candidate.payload["branch_prefix_steps"]
                ),
                "bpo_branch_prefix_shopper_calls": int(
                    candidate.payload["branch_prefix_shopper_calls"]
                ),
                "bpo_branch_prefix_environment_transitions": int(
                    candidate.payload["branch_prefix_environment_transitions"]
                ),
            }
        )

    async def run_tree(self, sampling_params, **kwargs):
        """Return either an independent Root group or a Local sibling group."""
        group_type = str(kwargs.pop("bpo_group_type", "local"))
        if group_type == "root":
            return await self._run_root(sampling_params, **kwargs)
        if group_type != "local":
            raise ValueError(f"unknown CARL-BPO group type: {group_type!r}")
        stage_target = str(kwargs.pop("bpo_stage_target", "auto"))
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
        retained_candidates = []
        action_starts = []
        data = await self._new_agent_data(kwargs)
        current = await self._handle_pending_state(data, sampling_params)
        try:
            while current != AgentState.TERMINATED:
                if current == AgentState.GENERATING:
                    action_index = len(action_starts)
                    action_starts.append(len(data.response_mask))
                    prefix_data = clone_agent_data(data)
                    prefix_state = deepcopy(state)
                    prefix_shopper = current_shopper.get().clone()
                    prompt_before = list(data.prompt_ids)
                    # Entropy probing is read-only.  Run it before creating
                    # the environment snapshot so an incompatible live vLLM
                    # entropy contract fails without allocating an opaque
                    # ShopSimulator snapshot.  The snapshot remains at the
                    # same pre-action semantic boundary.
                    entropy = await self._probe_entropy(
                        prompt_before,
                        sampling_params,
                    )
                    snapshot_id = await asyncio.to_thread(source_env.snapshot)
                    try:
                        # The paper defines H_t from the first-token
                        # distribution at the exact action boundary s_t.
                        # The snapshot and every sibling resume from this
                        # same prompt, without any backbone action token.
                        current = await self._handle_generating_state(
                            data, sampling_params
                        )
                    except Exception:
                        await asyncio.to_thread(
                            source_env.drop_snapshot, snapshot_id
                        )
                        raise
                    else:
                        proposed = BranchCandidate(
                            action_index=action_index,
                            token_offset=0,
                            entropy=entropy,
                            snapshot_id=snapshot_id,
                            stage=self._stage_for_prefix(prefix_state),
                            payload={
                                "agent_data": prefix_data,
                                "runtime_state": prefix_state,
                                "shopper": prefix_shopper,
                                "source_env": source_env,
                                "action_starts": list(action_starts),
                                "group_id": group_id,
                                "branch_prefix_steps": len(prefix_state["steps"]),
                                "branch_prefix_shopper_calls": int(
                                    prefix_shopper.call_count
                                ),
                                "branch_prefix_environment_transitions": sum(
                                    step.get("tool") != "ask_shopper"
                                    for step in prefix_state["steps"]
                                ),
                            },
                        )
                        previous = list(retained_candidates)
                        retained_candidates = self._retain_local_candidates(
                            [*previous, proposed]
                        )
                        retained_ids = {
                            value.snapshot_id for value in retained_candidates
                        }
                        for rejected in [*previous, proposed]:
                            if rejected.snapshot_id in retained_ids:
                                continue
                            await asyncio.to_thread(
                                source_env.drop_snapshot, rejected.snapshot_id
                            )
                elif current == AgentState.PROCESSING_TOOLS:
                    current = await self._handle_processing_tools_state(data)
                else:
                    raise RuntimeError(f"invalid BPO backbone state: {current}")
            try:
                eligible = [
                    value
                    for value in retained_candidates
                    if int(value.action_index) < len(action_starts) - 1
                ]
                if stage_target != "auto":
                    targeted = [
                        value for value in eligible if value.stage == stage_target
                    ]
                    if targeted:
                        eligible = targeted
                candidate = select_nonterminal_branch_candidate(
                    retain_branch_candidates(eligible, limit=max(1, len(eligible))),
                    action_count=len(action_starts),
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            candidate.payload["backbone_action_count"] = len(action_starts)
            candidate.payload["stage_target"] = stage_target
            candidate.payload["stage_fallback"] = bool(
                stage_target != "auto" and candidate.stage != stage_target
            )
            for rejected in retained_candidates:
                if rejected is candidate:
                    continue
                await asyncio.to_thread(
                    source_env.drop_snapshot, rejected.snapshot_id
                )
            retained_candidates = [candidate]
            backbone = self._finalize_shopping_output(
                self._make_output(data), state, task_id
            )
            self._attach_bpo_metadata(
                backbone,
                candidate,
                action_starts,
                0,
                group_id,
                source_env.env_idx,
            )
            branch_results = await asyncio.gather(
                *[
                    self._run_clone(candidate, sampling_params, task_id, sibling)
                    for sibling in range(1, self.sibling_count)
                ],
                return_exceptions=True,
            )
            branch_errors = [
                value for value in branch_results if isinstance(value, BaseException)
            ]
            if branch_errors:
                raise RuntimeError(
                    "one or more BPO clone continuations failed after all clone "
                    "leases were joined"
                ) from branch_errors[0]
            branches = list(branch_results)
            outputs = [backbone, *branches]
            validate_tree_outputs(outputs, sibling_count=self.sibling_count)
            attach_bpo_tree_metrics(outputs)
            return outputs
        finally:
            for retained in retained_candidates:
                await asyncio.to_thread(
                    source_env.drop_snapshot, retained.snapshot_id
                )
            # Keep the source lease alive until every snapshot clone has been
            # restored and completed.  This prevents server-side slot reuse
            # from racing with the three sibling restorations.
            await session.close()
