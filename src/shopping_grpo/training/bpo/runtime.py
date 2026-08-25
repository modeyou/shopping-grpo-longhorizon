"""Pinned veRL 0.8 runtime adapters for full BPO grouping and advantages."""

from __future__ import annotations

import asyncio
import json
import os

from shopping_grpo.training.bpo.advantage import (
    audit_bpo_rollout_batch,
    compute_bpo_advantage,
)
from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

_INSTALLED = False
_SPARSE_CUDA_MAPPING_MARKER = "SHOPPING_BPO_SPARSE_CUDA_MAPPING_V1"


def _is_bpo(value):
    return str(value).rsplit(".", 1)[-1].lower() == "bpo"


def sibling_group_starts(index, sibling_count):
    """Validate contiguous repeated prompts before any branch rollout starts."""
    values = list(index)
    sibling_count = int(sibling_count)
    if sibling_count < 2 or len(values) % sibling_count:
        raise ValueError("BPO worker batch must be divisible by sibling_count")
    starts = list(range(0, len(values), sibling_count))
    for start in starts:
        group = values[start:start + sibling_count]
        if len({str(value) for value in group}) != 1:
            raise ValueError("BPO siblings must be contiguous repeats of one prompt")
    return starts


def cuda_logical_ordinal(accelerator_id, visible_devices):
    """Map a Ray physical accelerator id into CUDA's masked logical namespace."""
    devices = [item.strip() for item in str(visible_devices).split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique non-empty devices")
    accelerator_id = str(accelerator_id).strip()
    if accelerator_id in devices:
        return devices.index(accelerator_id)
    try:
        logical = int(accelerator_id)
    except ValueError as exc:
        raise ValueError(
            f"Ray accelerator id {accelerator_id!r} is absent from CUDA_VISIBLE_DEVICES"
        ) from exc
    if 0 <= logical < len(devices):
        return logical
    raise ValueError(
        f"Ray accelerator id {accelerator_id!r} is outside the logical CUDA namespace"
    )


def install_sparse_cuda_mapping():
    """Make veRL colocated workers support non-contiguous physical GPU ids."""
    import ray
    from verl.single_controller.base.worker import Worker
    from verl.utils.device import (
        get_torch_device,
        get_visible_devices_keyword,
    )
    from verl.utils.ray_utils import ray_noset_visible_devices

    current = Worker._setup_env_cuda_visible_devices
    if getattr(current, "_shopping_bpo_marker", None) == _SPARSE_CUDA_MAPPING_MARKER:
        return

    def setup_env_cuda_visible_devices(worker):
        if not ray_noset_visible_devices():
            return current(worker)
        keyword = get_visible_devices_keyword().upper()
        visible_devices = os.environ.get(keyword)
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get(
            "GPU", []
        )
        if not visible_devices or len(accelerator_ids) != 1:
            return current(worker)
        logical_rank = cuda_logical_ordinal(accelerator_ids[0], visible_devices)
        os.environ["LOCAL_RANK"] = str(logical_rank)
        get_torch_device().set_device(logical_rank)

    setup_env_cuda_visible_devices._shopping_bpo_marker = (
        _SPARSE_CUDA_MAPPING_MARKER
    )
    Worker._setup_env_cuda_visible_devices = setup_env_cuda_visible_devices


async def _generate_bpo_sequences(worker, batch):
    import hydra
    import numpy as np
    from verl.experimental.agent_loop import agent_loop as module

    config = worker.rollout_config
    sibling_count = int(worker.config.shopping_bpo.sibling_count)
    sampling_params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": 1.0,
        "logprobs": config.calculate_log_probs,
    }
    if "agent_name" not in batch.non_tensor_batch:
        batch.non_tensor_batch["agent_name"] = np.array(
            [config.agent.default_agent_loop] * len(batch), dtype=object
        )
    index = batch.non_tensor_batch.get("index", np.arange(len(batch)))
    trajectory_info = await module.get_trajectory_info(
        batch.meta_info.get("global_steps", -1),
        index.tolist(),
        False,
    )

    async def run_group(start):
        rows = list(range(start, start + sibling_count))
        kwargs = {
            key: value[start]
            for key, value in batch.non_tensor_batch.items()
            if key != "__do_sample__"
        }
        agent_name = str(kwargs.pop("agent_name"))
        registry = module._agent_loop_registry
        if agent_name not in registry:
            raise ValueError(f"BPO agent loop is not registered: {agent_name}")
        loop = hydra.utils.instantiate(
            config=registry[agent_name],
            trainer_config=module.DictConfigWrap(config=worker.config),
            server_manager=worker.llm_client,
            tokenizer=worker.tokenizer,
            processor=worker.processor,
            dataset_cls=worker.dataset_cls,
            data_config=module.DictConfigWrap(worker.config.data),
            tools=module.ToolListWrap(worker.tools),
        )
        if not hasattr(loop, "run_tree"):
            raise TypeError("formal BPO agent loop must implement run_tree")
        outputs = await loop.run_tree(dict(sampling_params), **kwargs)
        if len(outputs) != sibling_count:
            raise RuntimeError("BPO run_tree returned the wrong sibling count")
        processed = []
        for local_index, (row, output) in enumerate(zip(rows, outputs, strict=True)):
            if int(output.extra_fields.get("bpo_sibling_index", -1)) != local_index:
                raise RuntimeError("BPO sibling output order is invalid")
            row_kwargs = {
                key: value[row]
                for key, value in batch.non_tensor_batch.items()
                if key not in {"__do_sample__", "agent_name"}
            }
            processed.append(
                await worker._agent_loop_postprocess(
                    output, trajectory_info[row]["validate"], **row_kwargs
                )
            )
        return processed

    starts = sibling_group_starts(index, sibling_count)
    groups = await asyncio.gather(*[run_group(start) for start in starts])
    outputs = [item for group in groups for item in group]
    return worker._postprocess(
        outputs,
        input_non_tensor_batch=batch.non_tensor_batch,
        validate=False,
    )


def install_bpo_runtime():
    """Install idempotent BPO hooks in the driver and every Ray worker."""
    global _INSTALLED
    install_torch_padding_fallback()
    install_sparse_cuda_mapping()
    if _INSTALLED:
        return

    from verl.experimental.agent_loop import agent_loop as agent_module
    from verl.trainer.ppo import ray_trainer

    original_generate = agent_module.AgentLoopWorker.generate_sequences
    original_advantage = ray_trainer.compute_advantage

    async def generate_sequences(worker, batch):
        enabled = bool(worker.config.get("shopping_bpo", {}).get("enable", False))
        validate = bool(batch.meta_info.get("validate", False))
        if not enabled or validate:
            return await original_generate(worker, batch)
        return await _generate_bpo_sequences(worker, batch)

    def compute_advantage(data, adv_estimator, *args, config=None, **kwargs):
        if not _is_bpo(adv_estimator):
            return original_advantage(
                data, adv_estimator, *args, config=config, **kwargs
            )
        bpo_config = (config or {}).get("bpo", {})
        metadata = {
            name: data.non_tensor_batch[name]
            for name in (
                "bpo_group_id",
                "bpo_sibling_index",
                "bpo_branch_action",
                "bpo_action_token_starts",
                "bpo_branch_entropy",
                "bpo_return_budget",
                "bpo_env_idx",
                "bpo_branch_prefix_sha256",
            )
        }
        audits = audit_bpo_rollout_batch(
            data.batch["prompts"],
            data.batch["responses"],
            data.batch["response_mask"],
            metadata=metadata,
            sibling_count=int(bpo_config.get("sibling_count", 4)),
        )
        advantages, returns = compute_bpo_advantage(
            data.batch["token_level_rewards"],
            data.batch["response_mask"],
            metadata=metadata,
            sibling_count=int(bpo_config.get("sibling_count", 4)),
            upstream_lambda=float(bpo_config.get("upstream_lambda", 0.95)),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        scores = data.batch["token_level_rewards"].sum(dim=-1)
        for audit in audits:
            rows = [
                index
                for index, value in enumerate(metadata["bpo_group_id"])
                if str(value) == audit["group_id"]
            ]
            group_returns = [float(scores[index]) for index in rows]
            total_return = sum(group_returns)
            loo = [
                value - (total_return - value) / (len(group_returns) - 1)
                for value in group_returns
            ]
            audit["returns"] = group_returns
            audit["loo_advantages"] = loo
            audit["loo_sum"] = sum(loo)
        print("BPO tree audit passed: " + json.dumps(audits, sort_keys=True))
        return data

    agent_module.AgentLoopWorker.generate_sequences = generate_sequences
    ray_trainer.compute_advantage = compute_advantage
    _INSTALLED = True


def worker_process_setup_hook():
    install_bpo_runtime()
