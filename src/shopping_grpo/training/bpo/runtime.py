"""Pinned veRL 0.8 runtime adapters for full BPO grouping and advantages."""

from __future__ import annotations

import asyncio

from shopping_grpo.training.bpo.advantage import compute_bpo_advantage
from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

_INSTALLED = False


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
            )
        }
        advantages, returns = compute_bpo_advantage(
            data.batch["token_level_rewards"],
            data.batch["response_mask"],
            metadata=metadata,
            sibling_count=int(bpo_config.get("sibling_count", 4)),
            upstream_lambda=float(bpo_config.get("upstream_lambda", 0.95)),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    agent_module.AgentLoopWorker.generate_sequences = generate_sequences
    ray_trainer.compute_advantage = compute_advantage
    _INSTALLED = True


def worker_process_setup_hook():
    install_bpo_runtime()
