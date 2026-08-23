#!/usr/bin/env python3
"""Preflight the pinned full-BPO runtime without loading model weights."""

from __future__ import annotations

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
    if not bool(model.use_fused_kernels) or not bool(model.use_liger):
        raise SystemExit("formal BPO requires use_fused_kernels=true and use_liger=true")
    if float(algorithm.bpo.upstream_lambda) != 0.95:
        raise SystemExit("formal BPO requires upstream_lambda=0.95")
    if float(config.actor_rollout_ref.actor.clip_ratio_low) != 0.2:
        raise SystemExit("formal BPO requires PPO clip ratio 0.2")
    if int(config.trainer.n_gpus_per_node) != 4:
        raise SystemExit("formal BPO requires four GPUs")


def validate_entropy_patch(verl_source):
    target = (
        verl_source.parent
        / "workers/rollout/vllm_rollout/vllm_async_server.py"
    )
    if not target.is_file() or PATCH_MARKER not in target.read_text(encoding="utf-8"):
        raise SystemExit(
            "BPO exact-entropy patch is missing; run "
            "scripts/apply_verl_bpo_patch.py first"
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
                f"incompatible BPO dependency: {package}={installed[package]}, expected {expected}"
            )
    import torch
    import verl
    from shopping_grpo.training.bpo.agent_loop import ShoppingBPOAgentLoop
    from shopping_grpo.training.bpo.runtime import install_bpo_runtime

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise SystemExit("formal BPO requires four visible CUDA devices")
    verl_source = Path(verl.__file__).resolve()
    validate_bpo_config(config)
    validate_entropy_patch(verl_source)
    common.validate_dynamic_sampling(config, verl_source, installed)
    validate_swanlab(config)
    install_bpo_runtime()
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
                "use_fused_kernels": True,
                "use_liger": True,
                "gpus": 4,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
