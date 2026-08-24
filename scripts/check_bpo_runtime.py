#!/usr/bin/env python3
"""Preflight the pinned full-BPO runtime without loading model weights."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from scripts import check_grpo_runtime as common
from shopping_grpo.training.bpo.entropy_patch import PATCH_MARKER


def validate_bpo_config(config):
    bpo = config.shopping_bpo
    algorithm = config.algorithm
    model = config.actor_rollout_ref.model
    rollout = config.actor_rollout_ref.rollout
    if str(algorithm.adv_estimator) != "bpo":
        raise SystemExit("BPO requires algorithm.adv_estimator=bpo")
    expected = {
        "enable": True,
        "sibling_count": 4,
        "branch_count": 1,
        "return_budget": 4,
        "selection": "maximum_exact_entropy",
        "entropy_probe": "exact-full-vocabulary",
    }
    for key, value in expected.items():
        if bpo.get(key) != value:
            raise SystemExit(f"formal BPO requires shopping_bpo.{key}={value!r}")
    if int(rollout.n) != 4:
        raise SystemExit("formal BPO requires rollout.n=4")
    if int(rollout.agent.num_workers) != int(config.data.train_batch_size):
        raise SystemExit("BPO workers must equal train_batch_size so K siblings stay together")
    if int(rollout.engine_kwargs.vllm.max_logprobs) != -1:
        raise SystemExit("exact BPO entropy requires vLLM max_logprobs=-1")
    if bool(model.use_fused_kernels):
        raise SystemExit("formal BPO requires use_fused_kernels=false")
    if not bool(model.use_liger) or not bool(model.use_remove_padding):
        raise SystemExit(
            "formal BPO requires use_liger=true and use_remove_padding=true"
        )
    if float(algorithm.bpo.upstream_lambda) != 0.95:
        raise SystemExit("formal BPO requires upstream_lambda=0.95")
    if float(config.actor_rollout_ref.actor.clip_ratio_low) != 0.2:
        raise SystemExit("formal BPO requires PPO clip ratio 0.2")
    if int(config.trainer.n_gpus_per_node) != 4:
        raise SystemExit("formal BPO requires four GPUs")


def validate_entropy_patch(verl_source):
    from scripts.apply_verl_bpo_patch import EXPECTED_PATCHED_SHA256

    target = verl_source.parent / "workers/rollout/vllm_rollout/vllm_async_server.py"
    if not target.is_file():
        raise SystemExit(f"BPO exact-entropy patch target is missing: {target}")
    actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual_sha256 != EXPECTED_PATCHED_SHA256:
        raise SystemExit(
            "BPO exact-entropy patch hash mismatch: "
            f"expected {EXPECTED_PATCHED_SHA256}, got {actual_sha256}; "
            "run scripts/apply_verl_bpo_patch.py first"
        )
    if PATCH_MARKER not in target.read_text(encoding="utf-8"):
        raise SystemExit(
            "BPO exact-entropy patch is missing; run "
            "scripts/apply_verl_bpo_patch.py first"
        )
    print(
        "BPO exact-entropy patch preflight passed: "
        + json.dumps(
            {"path": str(target), "sha256": actual_sha256},
            sort_keys=True,
        )
    )


def validate_bpo_runtime_hooks(config, *, validate_official_config=True):
    """Exercise the real veRL dispatcher with a tiny CPU-only sibling group."""
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    from verl.experimental.agent_loop import agent_loop as agent_module
    from verl.trainer.ppo import ray_trainer
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    from shopping_grpo.training.bpo.runtime import install_bpo_runtime

    install_bpo_runtime()
    use_critic = need_critic(config)
    use_reference_policy = need_reference_policy(config)
    if use_critic:
        raise SystemExit("formal BPO unexpectedly enabled a critic")
    if validate_official_config:
        validate_config(
            config=config,
            use_reference_policy=use_reference_policy,
            use_critic=use_critic,
        )
    expected_module = "shopping_grpo.training.bpo.runtime"
    if agent_module.AgentLoopWorker.generate_sequences.__module__ != expected_module:
        raise SystemExit("BPO AgentLoopWorker hook was not installed")
    if ray_trainer.compute_advantage.__module__ != expected_module:
        raise SystemExit("BPO advantage dispatcher hook was not installed")

    rewards = torch.tensor(
        [[0.0, 0.0, score, 0.0] for score in (1.0, 0.0, -1.0, 2.0)],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [[1.0, 0.0, 1.0, 1.0]] * 4,
        dtype=torch.float32,
    )
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": rewards,
                "response_mask": response_mask,
            },
            batch_size=[4],
        ),
        non_tensor_batch={
            "bpo_group_id": np.asarray(["preflight"] * 4, dtype=object),
            "bpo_sibling_index": np.asarray([0, 1, 2, 3]),
            "bpo_branch_action": np.asarray([1] * 4),
            "bpo_action_token_starts": np.asarray([[0, 2]] * 4, dtype=object),
        },
    )
    try:
        result = ray_trainer.compute_advantage(
            data,
            adv_estimator="bpo",
            config=config.algorithm,
        )
    except Exception as exc:
        raise SystemExit(
            "BPO veRL dispatcher CPU preflight failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    advantages = result.batch.get("advantages")
    returns = result.batch.get("returns")
    if advantages is None or returns is None:
        raise SystemExit("BPO veRL dispatcher did not produce advantages and returns")
    if tuple(advantages.shape) != (4, 4) or tuple(returns.shape) != (4, 4):
        raise SystemExit("BPO veRL dispatcher produced invalid tensor shapes")
    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise SystemExit("BPO veRL dispatcher produced NaN or Inf")
    if not torch.equal(advantages, returns):
        raise SystemExit("BPO veRL dispatcher returns do not match BPO advantages")
    print(
        "BPO veRL dispatcher CPU preflight passed: "
        + json.dumps(
            {
                "advantage_shape": list(advantages.shape),
                "agent_hook": agent_module.AgentLoopWorker.generate_sequences.__module__,
                "advantage_hook": ray_trainer.compute_advantage.__module__,
                "finite": True,
                "official_config_validation": bool(validate_official_config),
                "use_critic": use_critic,
                "use_reference_policy": use_reference_policy,
                "sibling_count": 4,
            },
            sort_keys=True,
        )
    )


def validate_swanlab(config):
    backends = list(config.trainer.get("logger", []))
    if "swanlab" not in backends:
        return
    if os.environ.get("SWANLAB_MODE") != "online" or not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("BPO SwanLab requires online mode and SWANLAB_API_KEY")
    if str(config.trainer.project_name) != "shopping-bpo":
        raise SystemExit("BPO SwanLab project must be shopping-bpo")


def main():
    config = common.compose_runtime_config(__import__("sys").argv[1:])
    common.validate_environment_contract()
    common.validate_reward_shaping_profile()
    common.validate_grpo_seeds(
        config, label="BPO", environment_name="BPO_SEED"
    )
    common.validate_training_memory_budget(config, label="BPO")
    common.validate_transformers_revision()
    installed = {}
    for package, expected in common.EXPECTED_VERSIONS.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise SystemExit(f"missing BPO dependency: {package}=={expected}") from exc
        if installed[package].split("+", 1)[0] != expected:
            raise SystemExit(
                f"incompatible BPO dependency: {package}={installed[package]}, "
                f"expected {expected}"
            )
    import torch
    import verl
    from shopping_grpo.training.bpo.agent_loop import ShoppingBPOAgentLoop

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise SystemExit("formal BPO requires four visible CUDA devices")
    verl_source = Path(verl.__file__).resolve()
    validate_bpo_config(config)
    validate_entropy_patch(verl_source)
    common.validate_dynamic_sampling(config, verl_source, installed)
    validate_swanlab(config)
    validate_bpo_runtime_hooks(config)
    print(
        "BPO runtime preflight passed: "
        + json.dumps(
            {
                "algorithm": "full-bpo-v1",
                "agent_loop": ShoppingBPOAgentLoop.__name__,
                "branch_count": 1,
                "sibling_count": 4,
                "return_budget": 4,
                "upstream_lambda": 0.95,
                "use_fused_kernels": False,
                "use_liger": True,
                "use_remove_padding": True,
                "gpus": 4,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
