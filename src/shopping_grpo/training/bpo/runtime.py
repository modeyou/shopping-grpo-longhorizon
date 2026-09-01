"""Pinned veRL 0.8 runtime adapters for CARL-BPO grouping and advantages."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os

from shopping_grpo.training.bpo.advantage import (
    audit_bpo_rollout_batch,
    compute_bpo_advantage,
    summarize_bpo_actor_batch,
)
from shopping_grpo.training.bpo.step0_validation import install_step0_validation_cache
from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
from shopping_grpo.training.grpo.dynamic_sampling import append_training_diagnostic

_INSTALLED = False
_SPARSE_CUDA_MAPPING_MARKER = "SHOPPING_BPO_SPARSE_CUDA_MAPPING_V1"
_OPTIMIZER_AUDIT_MARKER = "SHOPPING_BPO_OPTIMIZER_AUDIT_V2"
_SCHEDULER_CONTRACT_MARKER = "SHOPPING_BPO_SCHEDULER_CONTRACT_V1"
_FORWARD_BACKWARD_AUDIT_MARKER = "SHOPPING_BPO_FORWARD_BACKWARD_AUDIT_V1"
_FORWARD_BACKWARD_AUDIT_COUNT = 0


def _diagnostic_global_step(meta_info):
    """Normalize veRL's scalar/list global-step metadata for diagnostics."""
    raw = meta_info.get("global_steps", meta_info.get("global_step", -1))
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else -1
    item = getattr(raw, "item", None)
    if callable(item):
        raw = item()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _append_runtime_diagnostic(event, global_step, **payload):
    """Persist diagnostics without making logging failures change training."""
    path = os.environ.get("SHOPPING_GRPO_DIAGNOSTICS_PATH")
    if not path:
        return
    try:
        append_training_diagnostic(path, event, global_step, **payload)
    except (OSError, TypeError, ValueError) as exc:
        print(
            "BPO diagnostic write skipped: "
            + json.dumps(
                {"event": event, "error": f"{type(exc).__name__}:{exc}"},
                sort_keys=True,
            )
        )


def _tensor_support(value):
    """Summarize a tensor without retaining it after the forward call."""
    import torch

    if not torch.is_tensor(value):
        return None
    tensor = value.detach()
    flat = tensor.float()
    return {
        "shape": [int(item) for item in tensor.shape],
        "numel": int(tensor.numel()),
        "nonzero": int(tensor.ne(0).sum().item()),
        "sum": float(flat.sum().item()),
        "finite": bool(torch.isfinite(flat).all().item()),
    }


def _extract_loss_scalar(value):
    """Best-effort extraction that never changes the upstream loss object."""
    import torch
    from collections.abc import Mapping

    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().float().item())
        return None
    if isinstance(value, Mapping):
        for key in ("loss", "actor_loss", "policy_loss"):
            if key in value:
                scalar = _extract_loss_scalar(value[key])
                if scalar is not None:
                    return scalar
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            scalar = _extract_loss_scalar(item)
            if scalar is not None:
                return scalar
    return None


def install_forward_backward_audit():
    """Capture the actual actor loss-mask support before remove-padding."""
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    current = FSDPEngine.forward_backward_batch
    if (
        getattr(current, "_shopping_bpo_marker", None)
        == _FORWARD_BACKWARD_AUDIT_MARKER
    ):
        return
    _require_pinned_signature(
        current,
        ("self", "data", "loss_function"),
        "FSDPEngine.forward_backward_batch",
    )

    def forward_backward_batch(engine, data, loss_function, forward_only=False):
        global _FORWARD_BACKWARD_AUDIT_COUNT
        if forward_only:
            return current(
                engine,
                data,
                loss_function,
                forward_only=forward_only,
            )

        loss_values = []

        def audited_loss_function(*args, **kwargs):
            result = loss_function(*args, **kwargs)
            scalar = _extract_loss_scalar(result)
            if scalar is not None:
                loss_values.append(scalar)
            return result

        result = current(
            engine,
            data,
            audited_loss_function,
            forward_only=forward_only,
        )
        limit = int(os.environ.get("SHOPPING_BPO_LOSS_AUDIT_LIMIT", "8"))
        if _FORWARD_BACKWARD_AUDIT_COUNT < max(0, limit):
            _FORWARD_BACKWARD_AUDIT_COUNT += 1
            loss_mask = None
            attention_mask = None
            try:
                loss_mask = data["loss_mask"]
            except (KeyError, TypeError):
                pass
            try:
                attention_mask = data["attention_mask"]
            except (KeyError, TypeError):
                pass
            diagnostics = {
                "loss_mask": _tensor_support(loss_mask),
                "attention_mask": _tensor_support(attention_mask),
                "loss_values": [
                    float(value)
                    for value in loss_values[: max(1, limit)]
                ],
                "forward_only": False,
            }
            _append_runtime_diagnostic(
                "bpo_actor_loss_batch",
                -1,
                diagnostics=diagnostics,
            )
            print(
                "BPO actor loss batch diagnostics: "
                + json.dumps(diagnostics, sort_keys=True)
            )
        return result

    forward_backward_batch._shopping_bpo_marker = _FORWARD_BACKWARD_AUDIT_MARKER
    FSDPEngine.forward_backward_batch = forward_backward_batch


def _require_pinned_signature(function, expected_prefix, label):
    """Reject an incompatible veRL hook target before replacing it."""
    parameters = list(inspect.signature(function).parameters.values())
    actual_prefix = [item.name for item in parameters[:len(expected_prefix)]]
    if actual_prefix != list(expected_prefix):
        raise RuntimeError(
            f"pinned veRL {label} signature mismatch: expected prefix "
            f"{list(expected_prefix)!r}, got {actual_prefix!r}"
        )
    for parameter in parameters[:len(expected_prefix)]:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise RuntimeError(
                f"pinned veRL {label} parameter {parameter.name!r} "
                "is not positional"
            )


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
    _require_pinned_signature(
        current, ("self",), "Worker._setup_env_cuda_visible_devices"
    )

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


def _local_tensor(value):
    """Return the local tensor represented by a Tensor or DTensor."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _global_audit_counts(torch, values, device):
    """Sum optimizer audit counters across the active FSDP process group."""
    counts = torch.tensor(values, dtype=torch.float64, device=device)
    distributed = getattr(torch, "distributed", None)
    if (
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    ):
        distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)
    return [float(value) for value in counts.cpu().tolist()]


def install_optimizer_update_audit():
    """Hard-gate startup gradients and the first positive-LR parameter delta."""
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    current = FSDPEngine.optimizer_step
    if getattr(current, "_shopping_bpo_marker", None) == _OPTIMIZER_AUDIT_MARKER:
        return
    _require_pinned_signature(current, ("self",), "FSDPEngine.optimizer_step")

    def optimizer_step(engine):
        enabled = os.environ.get("SHOPPING_BPO_REQUIRE_PARAMETER_UPDATE") == "1"
        already_audited = bool(
            getattr(engine, "_shopping_bpo_optimizer_audited", False)
        )
        if not enabled or already_audited:
            return current(engine)

        import torch

        learning_rates = [
            float(group.get("lr", float("nan")))
            for group in engine.optimizer.param_groups
        ]
        learning_rates_valid = bool(learning_rates) and all(
            math.isfinite(value) and value >= 0.0 for value in learning_rates
        )
        maximum_learning_rate = max(learning_rates) if learning_rates_valid else None
        parameter_delta_required = bool(
            learning_rates_valid and maximum_learning_rate > 0.0
        )

        tracked = []
        trainable_tensors = 0
        grad_tensors = 0
        nonzero_grad_tensors = 0
        nonfinite_grad_values = 0
        grad_squared_sum = 0.0
        audit_device = None

        for group in engine.optimizer.param_groups:
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                trainable_tensors += 1
                local_parameter = _local_tensor(parameter.detach())
                if audit_device is None:
                    audit_device = local_parameter.device
                gradient = parameter.grad
                if gradient is None:
                    continue
                grad_tensors += 1
                local_gradient = _local_tensor(gradient.detach())
                finite = torch.isfinite(local_gradient)
                nonfinite_grad_values += int((~finite).sum().item())
                local_gradient_float = local_gradient.float()
                finite_gradient = torch.where(
                    finite,
                    local_gradient_float,
                    torch.zeros_like(local_gradient_float),
                )
                squared_sum = float(finite_gradient.square().sum().item())
                grad_squared_sum += squared_sum
                if squared_sum > 0.0:
                    nonzero_grad_tensors += 1
                    tracked.append((parameter, local_parameter.clone()))

        if audit_device is None:
            audit_device = next(engine.module.parameters()).device

        reported_grad_norm = current(engine)

        changed_tensors = 0
        for parameter, before in tracked:
            after = _local_tensor(parameter.detach())
            if not torch.equal(before, after):
                changed_tensors += 1

        (
            global_trainable,
            global_grad,
            global_nonzero_grad,
            global_nonfinite,
            global_grad_squared_sum,
            global_changed,
        ) = _global_audit_counts(
            torch,
            (
                trainable_tensors,
                grad_tensors,
                nonzero_grad_tensors,
                nonfinite_grad_values,
                grad_squared_sum,
                changed_tensors,
            ),
            audit_device,
        )
        audit = {
            "changed_parameter_tensors": int(global_changed),
            "grad_squared_sum": global_grad_squared_sum,
            "grad_tensors": int(global_grad),
            "nonfinite_grad_values": int(global_nonfinite),
            "nonzero_grad_tensors": int(global_nonzero_grad),
            "reported_grad_norm": float(reported_grad_norm),
            "trainable_parameter_tensors": int(global_trainable),
            "learning_rates": learning_rates,
            "maximum_learning_rate": maximum_learning_rate,
            "parameter_delta_required": parameter_delta_required,
        }
        warning_reasons = []
        if not learning_rates_valid:
            warning_reasons.append("invalid_learning_rate")
        if global_trainable <= 0:
            warning_reasons.append("no_trainable_parameters")
        if global_nonfinite > 0:
            warning_reasons.append("nonfinite_gradients")
        if not math.isfinite(float(reported_grad_norm)):
            warning_reasons.append("nonfinite_reported_grad_norm")
        if global_nonzero_grad <= 0 or global_grad_squared_sum <= 0.0:
            warning_reasons.append("no_nonzero_gradients")
        if parameter_delta_required and global_changed <= 0:
            warning_reasons.append("no_parameter_delta_at_positive_lr")
        audit["accepted"] = not warning_reasons
        audit["warning_reasons"] = warning_reasons
        print("BPO optimizer update audit: " + json.dumps(audit, sort_keys=True))
        if warning_reasons:
            print(
                "BPO optimizer audit warning: "
                + json.dumps(
                    {
                        "blocking": True,
                        "reasons": warning_reasons,
                    },
                    sort_keys=True,
                )
            )
        rank = 0
        distributed = getattr(torch, "distributed", None)
        if (
            distributed is not None
            and distributed.is_available()
            and distributed.is_initialized()
        ):
            rank = int(distributed.get_rank())
        if rank == 0:
            _append_runtime_diagnostic(
                "bpo_optimizer_backward",
                -1,
                audit=audit,
                phase="optimizer_step",
            )
        if warning_reasons:
            raise RuntimeError(
                "CARL-BPO startup optimizer gate failed: "
                + ", ".join(warning_reasons)
            )
        gradient_accepted = (
            global_trainable > 0
            and global_nonfinite <= 0
            and global_nonzero_grad > 0
            and global_grad_squared_sum > 0.0
        )
        if gradient_accepted:
            engine._shopping_bpo_gradient_audited = True
        if (
            gradient_accepted
            and parameter_delta_required
            and global_changed > 0
        ):
            engine._shopping_bpo_optimizer_audited = True
        return reported_grad_norm

    optimizer_step._shopping_bpo_marker = _OPTIMIZER_AUDIT_MARKER
    FSDPEngine.optimizer_step = optimizer_step


def install_scheduler_contract():
    """Keep the CARL 500-step LR curve fixed and resumable."""
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    current = FSDPEngine._build_lr_scheduler
    if getattr(current, "_shopping_bpo_marker", None) == _SCHEDULER_CONTRACT_MARKER:
        return
    _require_pinned_signature(
        current, ("self", "optimizer"), "FSDPEngine._build_lr_scheduler"
    )

    def build_lr_scheduler(engine, *args, **kwargs):
        horizon = int(os.environ.get("SHOPPING_BPO_SCHEDULER_HORIZON", "500"))
        warmup = int(os.environ.get("SHOPPING_BPO_WARMUP_STEPS", "10"))
        min_lr_ratio = float(
            os.environ.get("SHOPPING_BPO_MIN_LR_RATIO", "0.1")
        )
        if horizon != 500 or warmup != 10 or min_lr_ratio != 0.1:
            raise RuntimeError("formal BPO scheduler contract was modified")
        if not 0 <= warmup < horizon:
            raise RuntimeError("formal BPO scheduler warmup/horizon is invalid")
        optimizer_config = engine.optimizer_config
        # veRL BaseConfig freezes fields that are not explicitly listed in
        # _mutable_fields.  In v0.8.0 min_lr_ratio is frozen, while the formal
        # YAML already binds the warmup/scheduler/minimum ratio.  Validate those
        # immutable recipe values and only override the officially mutable
        # horizon required by the CARL run contract.
        if int(optimizer_config.lr_warmup_steps) != warmup:
            raise RuntimeError(
                "formal BPO scheduler warmup does not match the contract"
            )
        if str(optimizer_config.lr_scheduler_type) != "cosine":
            raise RuntimeError(
                "formal BPO scheduler type does not match the contract"
            )
        if float(optimizer_config.min_lr_ratio) != min_lr_ratio:
            raise RuntimeError(
                "formal BPO minimum LR ratio does not match the contract"
            )
        optimizer_config.total_training_steps = horizon
        # veRL 0.8.0 passes the freshly-built optimizer positionally.  Forward
        # every argument so the contract remains compatible with later veRL
        # signatures instead of shadowing the upstream method shape.
        scheduler = current(engine, *args, **kwargs)
        print(
            "BPO scheduler contract: "
            + json.dumps(
                {
                    "effective_return_budget": 4000,
                    "maximum_optimizer_steps": 500,
                    "horizon": horizon,
                    "min_lr_ratio": min_lr_ratio,
                    "scheduler": "cosine",
                    "warmup_steps": warmup,
                },
                sort_keys=True,
            )
        )
        return scheduler

    build_lr_scheduler._shopping_bpo_marker = _SCHEDULER_CONTRACT_MARKER
    FSDPEngine._build_lr_scheduler = build_lr_scheduler


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

    group_schedule = tuple(
        str(value)
        for value in worker.config.shopping_bpo.get(
            "group_schedule", ["root", "local"]
        )
    )
    raw_group_types = batch.non_tensor_batch.get("bpo_group_type")
    if raw_group_types is None:
        raise RuntimeError(
            "CARL-BPO requires driver-assigned bpo_group_type before worker sharding"
        )
    raw_stage_targets = batch.non_tensor_batch.get("bpo_stage_target")
    if raw_stage_targets is None:
        raise RuntimeError(
            "CARL-BPO requires driver-assigned bpo_stage_target before worker sharding"
        )

    async def run_group(start):
        rows = list(range(start, start + sibling_count))
        kwargs = {
            key: value[start]
            for key, value in batch.non_tensor_batch.items()
            if key != "__do_sample__"
        }
        agent_name = str(kwargs.pop("agent_name"))
        scheduled_types = {str(raw_group_types[row]) for row in rows}
        if len(scheduled_types) != 1:
            raise RuntimeError(
                "CARL-BPO sibling rows disagree on driver-assigned group type"
            )
        group_type = str(kwargs.pop("bpo_group_type"))
        if scheduled_types != {group_type} or group_type not in group_schedule:
            raise RuntimeError(
                "CARL-BPO worker received an invalid driver-assigned group type"
            )
        kwargs["bpo_group_type"] = group_type
        if group_type == "local":
            stage_targets = {str(raw_stage_targets[row]) for row in rows}
            if len(stage_targets) != 1:
                raise RuntimeError(
                    "CARL-BPO Local sibling rows disagree on driver-assigned stage target"
                )
            kwargs["bpo_stage_target"] = next(iter(stage_targets))
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
            raise TypeError("CARL-BPO agent loop must implement run_tree")
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
    install_optimizer_update_audit()
    install_forward_backward_audit()
    install_scheduler_contract()
    install_step0_validation_cache()
    if _INSTALLED:
        return

    from verl.experimental.agent_loop import agent_loop as agent_module
    from verl.trainer.ppo import ray_trainer

    original_generate = agent_module.AgentLoopWorker.generate_sequences
    original_advantage = ray_trainer.compute_advantage
    _require_pinned_signature(
        original_generate,
        ("self", "batch"),
        "AgentLoopWorker.generate_sequences",
    )
    _require_pinned_signature(
        original_advantage,
        ("data", "adv_estimator"),
        "ray_trainer.compute_advantage",
    )

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
                "bpo_group_type",
                "bpo_sibling_index",
                "bpo_branch_action",
                "bpo_action_token_starts",
                "bpo_branch_entropy",
                "bpo_return_budget",
                "bpo_env_idx",
                "bpo_branch_prefix_sha256",
                "bpo_backbone_action_count",
                "bpo_branch_relative_position",
            )
            if name in data.non_tensor_batch
        }
        if "bpo_group_type" not in metadata:
            raise RuntimeError(
                "CARL-BPO optimizer batch is missing driver-assigned bpo_group_type"
            )
        audits = audit_bpo_rollout_batch(
            data.batch["prompts"],
            data.batch["responses"],
            data.batch["response_mask"],
            metadata=metadata,
            sibling_count=int(bpo_config.get("sibling_count", 4)),
        )
        advantages, returns, internals = compute_bpo_advantage(
            data.batch["token_level_rewards"],
            data.batch["response_mask"],
            metadata=metadata,
            sibling_count=int(bpo_config.get("sibling_count", 4)),
            return_diagnostics=True,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        actor_batch = summarize_bpo_actor_batch(
            data.batch["token_level_rewards"],
            data.batch["response_mask"],
            advantages,
            returns,
            internals["policy_weights"],
        )
        global_step = _diagnostic_global_step(data.meta_info)
        actor_payload = {
            "diagnostics": actor_batch,
            "tree_count": len(audits),
            "sibling_count": int(bpo_config.get("sibling_count", 4)),
            "tree_audits": audits,
        }
        _append_runtime_diagnostic(
            "bpo_actor_batch",
            global_step,
            **actor_payload,
        )
        print(
            "BPO actor batch diagnostics: "
            + json.dumps(actor_payload, sort_keys=True)
        )
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
