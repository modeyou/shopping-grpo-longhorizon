#!/usr/bin/env python3
"""Preflight the pinned CARL-BPO runtime without loading model weights."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tempfile
from types import SimpleNamespace

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
        "algorithm": "carl-bpo-v3",
        "selection": "maximum_exact_entropy_with_semantic_gate",
        "entropy_probe": "exact-full-vocabulary",
        "entropy_state": "action-boundary-first-token",
        "rollout_audit": "exact-tree-v2",
        "semantic_action_contract": "canonical-tool-arguments-v1",
        "local_credit_support": "branch-action-only-v1",
        "policy_loss": "action-balanced-root-local-v1",
    }
    for key, value in expected.items():
        if bpo.get(key) != value:
            raise SystemExit(f"formal BPO requires shopping_bpo.{key}={value!r}")
    if list(bpo.group_schedule) != ["root", "local"]:
        raise SystemExit("CARL-BPO requires group_schedule=[root, local]")
    if dict(bpo.local_stage_weights) != {
        "product": 8,
        "option": 7,
        "search_strategy": 5,
    }:
        raise SystemExit("CARL-BPO local stage weights are invalid")
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
    dynamic = config.shopping_dynamic_sampling
    if int(dynamic.minimum_accepted_prompts) != 2:
        raise SystemExit(
            "formal BPO requires shopping_dynamic_sampling."
            "minimum_accepted_prompts=2"
        )
    if not bool(dynamic.require_full_batch):
        raise SystemExit("formal BPO requires strict full-tree batches")
    if int(dynamic.quality_search_gen_batches) != 10:
        raise SystemExit("formal BPO requires a 10-generation-batch quality window")
    if int(dynamic.max_num_gen_batches) != 120:
        raise SystemExit("formal BPO requires a 120-generation-batch hard limit")
    expected_checkpoint_steps = list(range(25, 501, 25))
    if list(dynamic.checkpoint_steps) != expected_checkpoint_steps:
        raise SystemExit("CARL-BPO checkpoint steps are invalid")
    expected_validation_steps = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    if list(dynamic.validation_steps) != expected_validation_steps:
        raise SystemExit("CARL-BPO validation steps are invalid")
    if int(config.data.dataloader_num_workers) != 0:
        raise SystemExit("formal BPO requires single-process parquet loading")
    if int(config.data.train_batch_size) != 2:
        raise SystemExit("formal BPO requires train_batch_size=2")
    if not bool(model.use_fused_kernels):
        raise SystemExit("formal BPO requires use_fused_kernels=true")
    if not bool(model.use_liger) or not bool(model.use_remove_padding):
        raise SystemExit(
            "formal BPO requires use_liger=true and use_remove_padding=true"
        )
    if bool(config.actor_rollout_ref.actor.calculate_entropy):
        raise SystemExit(
            "formal BPO requires actor.calculate_entropy=false; branch entropy "
            "comes from the exact one-token vLLM probe"
        )
    if float(config.actor_rollout_ref.actor.clip_ratio_low) != 0.2:
        raise SystemExit("formal BPO requires PPO clip ratio 0.2")
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "seq-mean-token-mean":
        raise SystemExit(
            "CARL-BPO v3 requires actor.loss_agg_mode=seq-mean-token-mean"
        )
    if int(config.trainer.n_gpus_per_node) != 4:
        raise SystemExit("formal BPO requires four GPUs")
    if int(bpo.effective_return_budget) != 4000:
        raise SystemExit("CARL-BPO requires effective_return_budget=4000")
    if int(bpo.effective_tree_budget) != 1000:
        raise SystemExit("CARL-BPO requires effective_tree_budget=1000")
    diagnostic_steps = str(
        os.environ.get("SHOPPING_BPO_DIAGNOSTIC_STEPS", "")
    ).strip()
    if diagnostic_steps:
        if diagnostic_steps != "1":
            raise SystemExit("BPO diagnostic mode supports exactly one step")
        if int(config.trainer.total_training_steps) != 1:
            raise SystemExit("BPO diagnostic mode requires exactly one global step")
        if bool(config.trainer.val_before_train):
            raise SystemExit("BPO diagnostic mode must disable validation")
        if int(config.trainer.save_freq) != -1 or int(config.trainer.test_freq) != -1:
            raise SystemExit("BPO diagnostic mode must disable save/test")
    else:
        if int(config.trainer.total_training_steps) != 500:
            raise SystemExit("CARL-BPO requires at most 500 global steps")
        if int(config.trainer.save_freq) != 25 or int(config.trainer.test_freq) != 50:
            raise SystemExit(
                "CARL-BPO requires checkpoints every 25 steps and validation every 50 steps"
            )
    if int(config.trainer.max_actor_ckpt_to_keep) != 20:
        raise SystemExit("CARL-BPO must retain all registered checkpoints")
    if int(config.data.seed) != 20260823:
        raise SystemExit("formal BPO requires data seed 20260823")
    optim = config.actor_rollout_ref.actor.optim
    if float(optim.lr) != 1.0e-6:
        raise SystemExit("formal BPO requires actor learning rate 1e-6")
    if int(optim.lr_warmup_steps) != 10:
        raise SystemExit("formal BPO requires 10 explicit warmup steps")
    if str(optim.lr_scheduler_type) != "cosine" or float(optim.min_lr_ratio) != 0.1:
        raise SystemExit("formal BPO requires cosine scheduling with min_lr_ratio=0.1")
    scheduler_environment = {
        "SHOPPING_BPO_SCHEDULER_HORIZON": "500",
        "SHOPPING_BPO_WARMUP_STEPS": "10",
        "SHOPPING_BPO_MIN_LR_RATIO": "0.1",
    }
    for name, expected_value in scheduler_environment.items():
        if os.environ.get(name) != expected_value:
            raise SystemExit(f"formal BPO requires {name}={expected_value}")


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


def validate_xml_tool_parser_patch(verl_source):
    from scripts.apply_verl_bpo_tool_parser_patch import expected_patched_sha256
    from shopping_grpo.training.bpo.xml_tool_parser_patch import PATCH_MARKER

    target = verl_source.parent / "experimental/agent_loop/tool_parser.py"
    if not target.is_file():
        raise SystemExit(f"BPO XML parser patch target is missing: {target}")
    actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        expected_sha256 = expected_patched_sha256(target)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "BPO XML parser patch hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}; "
            "run scripts/apply_verl_bpo_patch.py first"
        )
    if target.read_text(encoding="utf-8").count(PATCH_MARKER) != 1:
        raise SystemExit("BPO XML parser patch marker is missing")
    print(
        "BPO tolerant XML parser patch preflight passed: "
        + json.dumps(
            {"path": str(target), "sha256": actual_sha256}, sort_keys=True
        )
    )


def validate_fused_ppo_gradient_patch(verl_source):
    """Verify the backport and reproduce the formerly silent grad drop on CPU."""
    import torch

    from scripts.apply_verl_bpo_fused_grad_patch import expected_patched_sha256
    from shopping_grpo.training.bpo.fused_ppo_grad_patch import PATCH_MARKER

    target = verl_source.parent / "utils/experimental/torch_functional.py"
    if not target.is_file():
        raise SystemExit(f"BPO fused-PPO source is missing: {target}")
    actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        expected_sha256 = expected_patched_sha256(target)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "BPO fused-PPO gradient patch hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}; "
            "run scripts/apply_verl_bpo_patch.py first"
        )
    if target.read_text(encoding="utf-8").count(PATCH_MARKER) != 1:
        raise SystemExit("BPO fused-PPO gradient patch marker is missing")

    from verl.utils.experimental import torch_functional as fused

    # A transpose makes flatten(0, 1) allocate a new tensor inside the custom
    # autograd forward.  veRL 0.8.0 incorrectly inspected that saved copy's
    # requires_grad flag and returned no hidden-state gradient.
    base = torch.randn(2, 3, 5, dtype=torch.float32, requires_grad=True)
    hidden_states = base.transpose(0, 1)
    if hidden_states.is_contiguous():
        raise SystemExit("BPO fused-PPO gradient probe must be non-contiguous")
    vocab_weights = torch.randn(11, 5, dtype=torch.float32)
    input_ids = torch.randint(0, 11, hidden_states.shape[:2])
    flash_available = fused._FLASH_ATTN_CROSS_ENTROPY_AVAILABLE
    fused._FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False
    try:
        log_probs, _ = fused.FusedLinearForPPO(chunk_size=4)(
            hidden_states, vocab_weights, input_ids
        )
        (-log_probs.mean()).backward()
    finally:
        fused._FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = flash_available
    if base.grad is None or not torch.isfinite(base.grad).all():
        raise SystemExit("BPO fused-PPO gradient probe produced no finite gradient")
    gradient_abs_sum = float(base.grad.abs().sum().item())
    if gradient_abs_sum <= 0.0:
        raise SystemExit("BPO fused-PPO gradient probe produced an all-zero gradient")
    print(
        "BPO fused-PPO input-gradient preflight passed: "
        + json.dumps(
            {
                "gradient_abs_sum": gradient_abs_sum,
                "noncontiguous_hidden_states": True,
                "path": str(target),
                "sha256": actual_sha256,
            },
            sort_keys=True,
        )
    )


def build_scheduler_probe_engine(config, *, optimizer_config_class=None):
    """Build the frozen receiver used by veRL 0.8.0's scheduler."""
    if optimizer_config_class is None:
        from verl.workers.config.optimizer import FSDPOptimizerConfig

        optimizer_config_class = FSDPOptimizerConfig
    source = config.actor_rollout_ref.actor.optim
    optimizer_config = optimizer_config_class(
        lr=float(source.lr),
        lr_warmup_steps_ratio=float(source.lr_warmup_steps_ratio),
        total_training_steps=int(source.total_training_steps),
        lr_warmup_steps=int(source.lr_warmup_steps),
        min_lr_ratio=float(source.min_lr_ratio),
        lr_scheduler_type=str(source.lr_scheduler_type),
        num_cycles=float(source.num_cycles),
        zero_indexed_step=bool(source.zero_indexed_step),
    )
    return SimpleNamespace(
        # FSDPEngine._build_lr_scheduler logs its resolved horizon on rank 0.
        # Keep the probe on that path so preflight exercises every branch of
        # the pinned upstream method instead of bypassing rank-dependent code.
        rank=0,
        optimizer_config=optimizer_config,
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
        _SCHEDULER_CONTRACT_MARKER,
        _SPARSE_CUDA_MAPPING_MARKER,
        cuda_logical_ordinal,
        install_bpo_runtime,
    )

    install_bpo_runtime()
    from verl.single_controller.base.worker import Worker
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

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
    if (
        getattr(
            FSDPEngine._build_lr_scheduler,
            "_shopping_bpo_marker",
            None,
        )
        != _SCHEDULER_CONTRACT_MARKER
    ):
        raise SystemExit("BPO scheduler contract hook was not installed")
    # Exercise the exact veRL 0.8.0 call shape on CPU.  A marker/signature
    # check alone cannot prove that a wrapper forwards the optimizer argument.
    probe_parameter = torch.nn.Parameter(torch.tensor([0.0]))
    probe_optimizer = torch.optim.AdamW([probe_parameter], lr=1.0e-6)
    probe_engine = build_scheduler_probe_engine(config)
    try:
        probe_scheduler = FSDPEngine._build_lr_scheduler(
            probe_engine, probe_optimizer
        )
    except Exception as exc:
        raise SystemExit(
            "BPO scheduler hook failed the real veRL CPU call contract: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if int(probe_engine.optimizer_config.total_training_steps) != 500:
        raise SystemExit("BPO scheduler hook did not install the 500-step horizon")
    if not hasattr(probe_scheduler, "step"):
        raise SystemExit("BPO scheduler hook returned an invalid scheduler")
    print(
        "BPO scheduler CPU call-contract preflight passed: "
        + json.dumps(
            {
                "call": "FSDPEngine._build_lr_scheduler(engine, optimizer)",
                "horizon": 500,
                "scheduler_type": type(probe_scheduler).__name__,
            },
            sort_keys=True,
        )
    )
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

    sibling_scores = (1.0, 0.0, -1.0, 2.0)
    rewards = torch.tensor(
        [
            [0.0, 0.0, score, 0.0]
            for score in (*sibling_scores, *sibling_scores)
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [[1.0, 0.0, 1.0, 1.0]] * 8,
        dtype=torch.float32,
    )
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": rewards,
                "response_mask": response_mask,
                "responses": torch.tensor(
                    [[10, 11, 20 + row, 0] for row in range(8)],
                    dtype=torch.long,
                ),
                "prompts": torch.tensor([[1, 2]] * 8, dtype=torch.long),
            },
            batch_size=[8],
        ),
        non_tensor_batch={
            "bpo_group_id": np.asarray(
                ["preflight-root"] * 4 + ["preflight-local"] * 4,
                dtype=object,
            ),
            "bpo_group_type": np.asarray(["root"] * 4 + ["local"] * 4),
            "bpo_stage_target": np.asarray(["root"] * 4 + ["product"] * 4),
            "bpo_sibling_index": np.asarray([0, 1, 2, 3] * 2),
            "bpo_branch_action": np.asarray([-1] * 4 + [1] * 4),
            "bpo_action_token_starts": np.asarray(
                [[0]] * 4 + [[0, 2]] * 4,
                dtype=object,
            ),
            "bpo_action_token_ends": np.asarray(
                [[4]] * 4 + [[2, 4]] * 4,
                dtype=object,
            ),
            "bpo_action_metadata_valid": np.asarray([True] * 8),
            "bpo_branch_entropy": np.asarray([0.0] * 4 + [2.0] * 4),
            "bpo_return_budget": np.asarray([4] * 8),
            "bpo_env_idx": np.asarray([-1] * 4 + [0, 1, 2, 3]),
            "bpo_branch_prefix_sha256": np.asarray(
                ["root"] * 4 + ["local"] * 4,
                dtype=object,
            ),
            "bpo_backbone_action_count": np.asarray([1] * 4 + [3] * 4),
            "bpo_branch_relative_position": np.asarray([-1.0] * 4 + [0.5] * 4),
            "bpo_branch_semantic_action_sha256": np.asarray(
                [""] * 4 + ["product-a", "product-a", "product-b", "product-b"],
                dtype=object,
            ),
            "bpo_branch_semantic_valid": np.asarray(
                [False] * 4 + [True] * 4
            ),
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
    if tuple(advantages.shape) != (8, 4) or tuple(returns.shape) != (8, 4):
        raise SystemExit("BPO veRL dispatcher produced invalid tensor shapes")
    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise SystemExit("BPO veRL dispatcher produced NaN or Inf")
    if not torch.equal(advantages, returns):
        raise SystemExit("BPO veRL dispatcher returns do not match BPO advantages")
    advantage_support = advantages.ne(0)
    sequence_advantage_mass = advantages.abs().sum(dim=-1) / advantage_support.sum(
        dim=-1
    )
    root_advantage_mass = sequence_advantage_mass[:4].mean()
    local_advantage_mass = sequence_advantage_mass[4:].mean()
    if not torch.isclose(root_advantage_mass, local_advantage_mass):
        raise SystemExit(
            "BPO Root/Local sequence-normalized advantage mass is not balanced"
        )
    actor_metrics = result.meta_info.get("shopping_bpo_actor_metrics") or {}
    required_actor_metrics = {
        "bpo_action/active_tokens",
        "bpo_action/original_actor_tokens",
        "bpo_action/active_token_ratio",
        "bpo_action/root_actions",
        "bpo_action/local_actions",
        "bpo_action/root_policy_weight_mass",
        "bpo_action/local_policy_weight_mass",
    }
    missing_actor_metrics = required_actor_metrics.difference(actor_metrics)
    if missing_actor_metrics:
        raise SystemExit(
            "BPO veRL dispatcher is missing action metrics: "
            + ", ".join(sorted(missing_actor_metrics))
        )
    if any(
        abs(float(actor_metrics[f"bpo_action/{group}_policy_weight_mass"]) - 0.5)
        > 1e-6
        for group in ("root", "local")
    ):
        raise SystemExit("BPO Root/Local policy weight mass is not 0.5/0.5")
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
                "group_types": ["root", "local"],
                "root_advantage_mass": float(root_advantage_mass.item()),
                "local_advantage_mass": float(local_advantage_mass.item()),
                "action_metrics": actor_metrics,
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
    if str(config.trainer.project_name) != "shopping-multiturn-agentic":
        raise SystemExit(
            "BPO SwanLab project must be shopping-multiturn-agentic"
        )
    try:
        import swanlab

        authenticated = swanlab.login(os.environ["SWANLAB_API_KEY"])
    except Exception as exc:
        raise SystemExit(
            "BPO SwanLab authentication failed "
            f"({type(exc).__name__}); verify SWANLAB_API_KEY"
        ) from exc
    if authenticated is False:
        raise SystemExit(
            "BPO SwanLab authentication failed; verify SWANLAB_API_KEY"
        )
    print(
        "BPO SwanLab authentication preflight passed: "
        + json.dumps(
            {
                "mode": "online",
                "project": str(config.trainer.project_name),
            },
            sort_keys=True,
        )
    )


def validate_shopper_api():
    try:
        from scripts.run_sft_checkpoint_sweep import validate_shopper_api as probe

        probe(
            base_url=os.environ["SHOPPER_BASE_URL"],
            api_key=os.environ["SHOPPER_API_KEY"],
            model=os.environ["SHOPPER_MODEL"],
            timeout=30,
        )
    except (KeyError, RuntimeError) as exc:
        raise SystemExit(f"BPO Shopper API authentication failed: {exc}") from exc


def validate_finalize_hook():
    from shopping_grpo.training.grpo.adapter.agent_loop import (
        ShoppingToolAgentLoop,
    )
    from shopping_grpo.training.grpo.adapter.runtime import (
        current_shopper,
        make_runtime_state,
    )

    loop = ShoppingToolAgentLoop.__new__(ShoppingToolAgentLoop)
    loop.reward_mode = "native"
    loop.reward_shaping_profile = "none"
    loop.reward_length_shaping_enable = False
    loop.reward_length_soft_threshold = 20
    loop.reward_length_penalty_per_step = 0.01
    loop.reward_length_max_penalty = 0.15
    output = SimpleNamespace(reward_score=None, extra_fields={})
    state = make_runtime_state(task_id=3, max_steps=35, interaction_mode="gap")
    shopper_token = current_shopper.set(SimpleNamespace(call_count=2))
    try:
        finalized = loop._finalize_shopping_output(output, state, task_id=3)
    except Exception as exc:
        raise SystemExit(
            "BPO GRPO-finalize hook preflight failed "
            f"({type(exc).__name__}): {exc}"
        ) from exc
    finally:
        current_shopper.reset(shopper_token)
    shopping = finalized.extra_fields.get("shopping") or {}
    if shopping.get("shopper_llm_calls") != 2:
        raise SystemExit(
            "BPO GRPO-finalize hook did not preserve Shopper call diagnostics"
        )
    print(
        "BPO GRPO-finalize hook preflight passed: "
        + json.dumps(
            {
                "shopper_llm_calls": shopping["shopper_llm_calls"],
                "task_id": shopping["task_id"],
            },
            sort_keys=True,
        )
    )


def validate_step0_validation_cache():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    from shopping_grpo.training.bpo.step0_validation import (
        CACHE_PATH_ENV,
        CONTRACT_SHA256_ENV,
        REFRESH_ENV,
        freeze_validation_cache,
        install_step0_validation_cache,
        validate_contract,
    )

    try:
        contract = json.loads(os.environ["SHOPPING_BPO_STEP0_CONTRACT_JSON"])
        expected_sha256 = os.environ[CONTRACT_SHA256_ENV]
        cache_path = Path(os.environ[CACHE_PATH_ENV]).resolve()
    except (KeyError, json.JSONDecodeError) as exc:
        raise SystemExit("BPO step-0 validation contract environment is invalid") from exc
    actual_sha256 = validate_contract(contract)
    if actual_sha256 != expected_sha256:
        raise SystemExit("BPO step-0 validation contract environment hash mismatch")
    if cache_path.name != f"{actual_sha256}.json":
        raise SystemExit("BPO step-0 validation cache is not content-addressed")

    original_environment = {
        name: os.environ.get(name)
        for name in (CACHE_PATH_ENV, CONTRACT_SHA256_ENV, REFRESH_ENV)
    }
    with tempfile.TemporaryDirectory(prefix="shopping-bpo-step0-preflight-") as directory:
        probe_cache = Path(directory) / f"{actual_sha256}.json"
        freeze_validation_cache(
            probe_cache,
            contract_sha256_value=actual_sha256,
            metrics={"val-shopping/reward/strict_mean": 0.5},
        )
        os.environ[CACHE_PATH_ENV] = str(probe_cache)
        os.environ[CONTRACT_SHA256_ENV] = actual_sha256
        os.environ[REFRESH_ENV] = "0"
        try:
            install_step0_validation_cache()
            metrics = RayPPOTrainer._validate(SimpleNamespace(global_steps=0))
        finally:
            for name, value in original_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    if metrics.get("val-shopping/reward/strict_mean") != 0.5:
        raise SystemExit("BPO step-0 validation cache changed a frozen metric")
    if metrics.get("val-step0/cache_hit") != 1.0:
        raise SystemExit("BPO step-0 validation cache hook did not reuse the probe")
    print(
        "BPO step-0 validation cache preflight passed: "
        + json.dumps(
            {
                "cache_path": str(cache_path),
                "contract_sha256": actual_sha256,
                "swanlab_replay": True,
            },
            sort_keys=True,
        )
    )

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
    validate_swanlab(config)
    validate_shopper_api()
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
    validate_xml_tool_parser_patch(verl_source)
    validate_fused_ppo_gradient_patch(verl_source)
    common.validate_dynamic_sampling(config, verl_source, installed)
    common.validate_tracking_finish_patch(verl_source)
    validate_step0_validation_cache()
    validate_finalize_hook()
    validate_bpo_runtime_hooks(config)
    print(
        "BPO runtime preflight passed: "
        + json.dumps(
            {
                "algorithm": "carl-bpo-v3",
                "agent_loop": ShoppingBPOAgentLoop.__name__,
                "branch_count": 1,
                "sibling_count": 4,
                "return_budget": 4,
                "effective_tree_budget": 1000,
                "effective_return_budget": 4000,
                "trees_per_optimizer_step": 2,
                "returns_per_optimizer_step": 8,
                "maximum_optimizer_steps": 500,
                "scheduler": "cosine",
                "scheduler_horizon": 500,
                "warmup_steps": 10,
                "minimum_lr_ratio": 0.1,
                "swanlab_project": str(config.trainer.project_name),
                "dynamic_target_prompts": int(config.data.train_batch_size),
                "dynamic_minimum_accepted_prompts": int(
                    config.shopping_dynamic_sampling.minimum_accepted_prompts
                ),
                "dynamic_require_full_batch": bool(
                    config.shopping_dynamic_sampling.require_full_batch
                ),
                "dynamic_quality_search_generation_batches": int(
                    config.shopping_dynamic_sampling.quality_search_gen_batches
                ),
                "dynamic_max_generation_batches": int(
                    config.shopping_dynamic_sampling.max_num_gen_batches
                ),
                "gpu_memory_utilization": 0.45,
                "max_num_seqs": 8,
                "minimum_free_gpu_memory_gib": 20.0,
                "sparse_cuda_mapping": "physical-to-logical-v1",
                "visible_gpu_free_gib": visible_gpu_free_gib,
                "use_fused_kernels": True,
                "use_liger": True,
                "use_remove_padding": True,
                "actor_calculate_entropy": False,
                "gpus": 4,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
