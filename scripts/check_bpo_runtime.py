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
        "entropy_state": "action-boundary-first-token",
        "rollout_audit": "exact-tree-v1",
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
    if float(rollout.gpu_memory_utilization) != 0.45:
        raise SystemExit("formal BPO requires vLLM gpu_memory_utilization=0.45")
    if int(rollout.max_num_seqs) != 8:
        raise SystemExit("formal BPO requires vLLM max_num_seqs=8")
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
    from scripts.apply_verl_bpo_patch import expected_patched_sha256

    target = verl_source.parent / "workers/rollout/vllm_rollout/vllm_async_server.py"
    if not target.is_file():
        raise SystemExit(f"BPO exact-entropy patch target is missing: {target}")
    actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        expected_sha256 = expected_patched_sha256(target)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "BPO exact-entropy patch hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}; "
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

    from shopping_grpo.training.bpo.runtime import (
        _SPARSE_CUDA_MAPPING_MARKER,
        cuda_logical_ordinal,
        install_bpo_runtime,
    )

    install_bpo_runtime()
    from verl.single_controller.base.worker import Worker

    if (
        getattr(
            Worker._setup_env_cuda_visible_devices,
            "_shopping_bpo_marker",
            None,
        )
        != _SPARSE_CUDA_MAPPING_MARKER
    ):
        raise SystemExit("BPO sparse CUDA worker hook was not installed")
    sparse_cuda_mapping = [
        cuda_logical_ordinal(value, "0,2,3,4")
        for value in ("0", "2", "3", "4")
    ]
    if sparse_cuda_mapping != [0, 1, 2, 3]:
        raise SystemExit("BPO sparse CUDA physical-to-logical mapping is invalid")
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
                "responses": torch.tensor(
                    [[10, 11, 20 + row, 0] for row in range(4)],
                    dtype=torch.long,
                ),
                "prompts": torch.tensor([[1, 2]] * 4, dtype=torch.long),
            },
            batch_size=[4],
        ),
        non_tensor_batch={
            "bpo_group_id": np.asarray(["preflight"] * 4, dtype=object),
            "bpo_sibling_index": np.asarray([0, 1, 2, 3]),
            "bpo_branch_action": np.asarray([1] * 4),
            "bpo_action_token_starts": np.asarray([[0, 2]] * 4, dtype=object),
            "bpo_branch_entropy": np.asarray([2.0] * 4),
            "bpo_return_budget": np.asarray([4] * 4),
            "bpo_env_idx": np.asarray([0, 0, 1, 2]),
            "bpo_branch_prefix_sha256": np.asarray(["same"] * 4, dtype=object),
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
                "sparse_cuda_mapping": sparse_cuda_mapping,
            },
            sort_keys=True,
        )
    )


def validate_snapshot_fidelity():
    """Exercise a formal task through warm state, restore and branch continuation."""
    from shopping_grpo.environment.client import ShopAgentEnv

    base_url = os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700")
    task_manifest = (
        Path(__file__).resolve().parents[1]
        / "data/grpo/formal-v2/multiturn-train-tasks.jsonl"
    )
    try:
        task = next(
            json.loads(line)
            for line in task_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        task_id = int(task["task_id"])
    except (OSError, StopIteration, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"BPO snapshot preflight cannot read a formal train task: {task_manifest}"
        ) from exc
    source = ShopAgentEnv(base_url=base_url, timeout=60, multiturn=True)
    clones = []
    snapshot_id = None
    source_released = False
    try:
        source.reset(task_id, initial_request="")
        source.step("search[bpo snapshot warm state]")
        snapshot_id = source.snapshot()
        source_env_idx = source.env_idx
        action = "search[bpo snapshot fidelity probe]"
        source_transition = source.step(action)
        # Match formal rollout: release the completed backbone lease before
        # restoring K-1 siblings from the still-live opaque snapshot.
        source.release()
        source_released = True
        clones = [source.clone(snapshot_id) for _ in range(3)]
        env_indices = [source_env_idx, *[clone.env_idx for clone in clones]]
        if len(set(env_indices[1:])) != 3:
            raise SystemExit("BPO snapshot preflight clone leases are not isolated")
        transitions = [source_transition, *[clone.step(action) for clone in clones]]
        comparable = []
        for transition in transitions:
            normalized = dict(transition)
            normalized.pop("env_idx", None)
            normalized.pop("idx", None)
            comparable.append(normalized)
        canonical = json.dumps(
            comparable[0], ensure_ascii=False, sort_keys=True, default=str
        )
        if any(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            != canonical
            for value in comparable[1:]
        ):
            raise SystemExit(
                "BPO snapshot fidelity failed: identical actions from one snapshot "
                "produced different transitions"
            )
        divergent_transitions = [
            clone.step(f"search[bpo branch continuation {index}]")
            for index, clone in enumerate(clones, start=1)
        ]
        if not all(isinstance(value, dict) for value in divergent_transitions):
            raise SystemExit("BPO snapshot clones cannot continue independently")
        print(
            "BPO ShopSimulator snapshot fidelity preflight passed: "
            + json.dumps(
                {
                    "task_id": task_id,
                    "source_env_idx": env_indices[0],
                    "clone_env_indices": env_indices[1:],
                    "identical_transition_count": len(transitions),
                    "divergent_continuation_count": len(divergent_transitions),
                },
                sort_keys=True,
            )
        )
    finally:
        for env in reversed(clones):
            env.release()
        if not source_released:
            source.release()
        if snapshot_id is not None:
            source.drop_snapshot(snapshot_id)


def validate_swanlab(config):
    backends = list(config.trainer.get("logger", []))
    if "swanlab" not in backends:
        return
    if os.environ.get("SWANLAB_MODE") != "online" or not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("BPO SwanLab requires online mode and SWANLAB_API_KEY")
    if str(config.trainer.project_name) != "shopping-bpo":
        raise SystemExit("BPO SwanLab project must be shopping-bpo")


def validate_visible_gpu_headroom(torch, *, minimum_free_gib=20.0):
    device_count = int(torch.cuda.device_count())
    if device_count != 4:
        raise SystemExit(
            "formal BPO requires exactly four visible CUDA devices; "
            "set CUDA_VISIBLE_DEVICES to four clean GPUs"
        )
    free_gib = []
    for index in range(device_count):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        available = free_bytes / (1024 ** 3)
        free_gib.append(round(available, 2))
        if available < minimum_free_gib:
            raise SystemExit(
                f"visible CUDA device {index} has only {available:.2f} GiB free; "
                f"formal BPO requires at least {minimum_free_gib:.2f} GiB per GPU"
            )
    print(
        "BPO visible-GPU headroom preflight passed: "
        + json.dumps(
            {
                "device_count": device_count,
                "free_gib": free_gib,
                "minimum_free_gib": minimum_free_gib,
            },
            sort_keys=True,
        )
    )
    return free_gib


def main():
    config = common.compose_runtime_config(__import__("sys").argv[1:])
    common.validate_environment_contract()
    validate_snapshot_fidelity()
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

    if not torch.cuda.is_available():
        raise SystemExit("formal BPO requires CUDA")
    visible_gpu_free_gib = validate_visible_gpu_headroom(torch)
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
                "gpu_memory_utilization": 0.45,
                "max_num_seqs": 8,
                "minimum_free_gpu_memory_gib": 20.0,
                "sparse_cuda_mapping": "physical-to-logical-v1",
                "visible_gpu_free_gib": visible_gpu_free_gib,
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
