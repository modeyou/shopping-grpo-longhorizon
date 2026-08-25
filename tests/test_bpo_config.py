import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.check_bpo_runtime import validate_visible_gpu_headroom
from shopping_grpo.training.bpo.runtime import (
    _OPTIMIZER_AUDIT_MARKER,
    _SCHEDULER_CONTRACT_MARKER,
    _SPARSE_CUDA_MAPPING_MARKER,
    _require_pinned_signature,
    cuda_logical_ordinal,
    install_optimizer_update_audit,
    install_scheduler_contract,
    install_sparse_cuda_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_signature_guard_rejects_incompatible_hook():
    def compatible(self, optimizer):
        return self, optimizer

    def incompatible(self):
        return self

    _require_pinned_signature(
        compatible, ("self", "optimizer"), "scheduler fixture"
    )
    with pytest.raises(RuntimeError, match="signature mismatch"):
        _require_pinned_signature(
            incompatible, ("self", "optimizer"), "scheduler fixture"
        )


def test_formal_bpo_config_is_independent_and_frozen():
    config = yaml.safe_load((ROOT / "configs/bpo.yaml").read_text(encoding="utf-8"))
    model = config["actor_rollout_ref"]["model"]
    rollout = config["actor_rollout_ref"]["rollout"]
    assert config["algorithm"]["adv_estimator"] == "bpo"
    assert config["shopping_bpo"] == {
        "enable": True,
        "sibling_count": 4,
        "branch_count": 1,
        "return_budget": 4,
        "selection": "maximum_exact_entropy",
        "entropy_tie_break": "earliest_action",
        "entropy_probe": "exact-full-vocabulary",
        "entropy_state": "action-boundary-first-token",
        "rollout_audit": "exact-tree-v1",
        "effective_return_budget": 1600,
    }
    assert rollout["n"] == 4
    assert rollout["agent"]["num_workers"] == 2
    assert rollout["engine_kwargs"]["vllm"]["max_logprobs"] == -1
    assert rollout["gpu_memory_utilization"] == 0.45
    assert rollout["max_num_seqs"] == 8
    assert model["use_fused_kernels"] is True
    assert model["use_liger"] is True
    assert model["use_remove_padding"] is True
    assert config["actor_rollout_ref"]["actor"]["calculate_entropy"] is False
    assert config["trainer"]["n_gpus_per_node"] == 4
    assert config["trainer"]["project_name"] == "shopping-multiturn-agentic"
    assert config["trainer"]["total_training_steps"] == 200
    assert config["trainer"]["save_freq"] == 25
    assert config["trainer"]["max_actor_ckpt_to_keep"] == 9
    assert config["trainer"]["test_freq"] == 50
    assert config["shopping_bpo"]["effective_return_budget"] == 1600
    optim = config["actor_rollout_ref"]["actor"]["optim"]
    assert optim == {
        "lr": 1.0e-6,
        "lr_warmup_steps": 10,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
    }
    assert config["data"]["train_batch_size"] == 2
    assert config["data"]["dataloader_num_workers"] == 0
    assert config["shopping_dynamic_sampling"]["minimum_accepted_prompts"] == 2
    assert config["shopping_dynamic_sampling"]["require_full_batch"] is True
    assert config["shopping_dynamic_sampling"]["soft_warning_gen_batches"] == 10
    assert config["shopping_dynamic_sampling"]["max_num_gen_batches"] == 30
    assert config["shopping_dynamic_sampling"]["checkpoint_steps"] == [10]
    assert config["shopping_dynamic_sampling"]["validation_steps"] == [10]


class _FakeCuda:
    def __init__(self, free_gib):
        self.free_gib = list(free_gib)

    def device_count(self):
        return len(self.free_gib)

    def mem_get_info(self, index):
        gib = 1024 ** 3
        return int(self.free_gib[index] * gib), 24 * gib


def test_visible_gpu_headroom_accepts_four_clean_devices():
    torch = SimpleNamespace(cuda=_FakeCuda([23.5, 22.0, 21.5, 24.0]))
    assert validate_visible_gpu_headroom(torch) == [23.5, 22.0, 21.5, 24.0]


def test_visible_gpu_headroom_rejects_busy_or_wrong_device_count():
    with pytest.raises(SystemExit, match="only 15.50 GiB free"):
        validate_visible_gpu_headroom(
            SimpleNamespace(cuda=_FakeCuda([23.5, 15.5, 23.5, 23.5]))
        )
    with pytest.raises(SystemExit, match="exactly four visible CUDA devices"):
        validate_visible_gpu_headroom(
            SimpleNamespace(cuda=_FakeCuda([23.5, 23.5, 23.5]))
        )


def test_sparse_physical_gpu_ids_map_to_masked_cuda_ordinals():
    visible_devices = "0,2,3,4"
    assert [
        cuda_logical_ordinal(value, visible_devices)
        for value in ("0", "2", "3", "4")
    ] == [0, 1, 2, 3]
    assert cuda_logical_ordinal("1", visible_devices) == 1
    with pytest.raises(ValueError, match="outside the logical CUDA namespace"):
        cuda_logical_ordinal("5", visible_devices)
    with pytest.raises(ValueError, match="unique non-empty devices"):
        cuda_logical_ordinal("0", "0,2,2,4")


def test_sparse_cuda_worker_hook_sets_masked_logical_device(monkeypatch):
    selected = []

    class _Worker:
        def _setup_env_cuda_visible_devices(self):
            raise AssertionError("the pinned veRL implementation must be replaced")

    ray = types.ModuleType("ray")
    ray.get_runtime_context = lambda: SimpleNamespace(
        get_accelerator_ids=lambda: {"GPU": ["4"]}
    )
    worker = types.ModuleType("verl.single_controller.base.worker")
    worker.Worker = _Worker
    device = types.ModuleType("verl.utils.device")
    device.get_torch_device = lambda: SimpleNamespace(
        set_device=lambda value: selected.append(value)
    )
    device.get_visible_devices_keyword = lambda: "CUDA_VISIBLE_DEVICES"
    ray_utils = types.ModuleType("verl.utils.ray_utils")
    ray_utils.ray_noset_visible_devices = lambda: True

    for name, module in {
        "ray": ray,
        "verl": types.ModuleType("verl"),
        "verl.single_controller": types.ModuleType("verl.single_controller"),
        "verl.single_controller.base": types.ModuleType(
            "verl.single_controller.base"
        ),
        "verl.single_controller.base.worker": worker,
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.device": device,
        "verl.utils.ray_utils": ray_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")

    install_sparse_cuda_mapping()
    instance = _Worker()
    instance._setup_env_cuda_visible_devices()

    assert selected == [3]
    assert os.environ["LOCAL_RANK"] == "3"
    assert (
        getattr(
            _Worker._setup_env_cuda_visible_devices,
            "_shopping_bpo_marker",
            None,
        )
        == _SPARSE_CUDA_MAPPING_MARKER
    )


def test_optimizer_audit_requires_real_gradient_and_parameter_delta(
    monkeypatch, capsys
):
    import torch

    class _FSDPEngine:
        def optimizer_step(self):
            self.optimizer.step()
            return 0.0

    transformer = types.ModuleType("verl.workers.engine.fsdp.transformer_impl")
    transformer.FSDPEngine = _FSDPEngine
    monkeypatch.setitem(sys.modules, transformer.__name__, transformer)
    monkeypatch.setenv("SHOPPING_BPO_REQUIRE_PARAMETER_UPDATE", "1")

    install_optimizer_update_audit()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([2.0])
    engine = _FSDPEngine()
    engine.module = torch.nn.Module()
    engine.module.register_parameter("weight", parameter)
    engine.optimizer = torch.optim.SGD([parameter], lr=0.1)

    assert engine.optimizer_step() == 0.0
    assert parameter.item() == pytest.approx(0.8)
    output = capsys.readouterr().out
    assert "BPO optimizer update audit" in output
    assert '"changed_parameter_tensors": 1' in output
    assert (
        getattr(_FSDPEngine.optimizer_step, "_shopping_bpo_marker", None)
        == _OPTIMIZER_AUDIT_MARKER
    )


def test_optimizer_audit_rejects_zero_gradient(monkeypatch):
    import torch

    class _FSDPEngine:
        def optimizer_step(self):
            self.optimizer.step()
            return 0.0

    transformer = types.ModuleType("verl.workers.engine.fsdp.transformer_impl")
    transformer.FSDPEngine = _FSDPEngine
    monkeypatch.setitem(sys.modules, transformer.__name__, transformer)
    monkeypatch.setenv("SHOPPING_BPO_REQUIRE_PARAMETER_UPDATE", "1")

    install_optimizer_update_audit()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.zeros(1)
    engine = _FSDPEngine()
    engine.module = torch.nn.Module()
    engine.module.register_parameter("weight", parameter)
    engine.optimizer = torch.optim.SGD([parameter], lr=0.1)

    with pytest.raises(RuntimeError, match="no non-zero gradients"):
        engine.optimizer_step()


def test_scheduler_contract_decouples_curve_from_formal_stop(monkeypatch, capsys):
    calls = []

    class _FSDPEngine:
        def _build_lr_scheduler(self, optimizer):
            calls.append((dict(vars(self.optimizer_config)), optimizer))
            return "scheduler"

    transformer = types.ModuleType("verl.workers.engine.fsdp.transformer_impl")
    transformer.FSDPEngine = _FSDPEngine
    monkeypatch.setitem(sys.modules, transformer.__name__, transformer)
    monkeypatch.setenv("SHOPPING_BPO_SCHEDULER_HORIZON", "500")
    monkeypatch.setenv("SHOPPING_BPO_WARMUP_STEPS", "10")
    monkeypatch.setenv("SHOPPING_BPO_MIN_LR_RATIO", "0.1")

    install_scheduler_contract()
    engine = _FSDPEngine()
    engine.optimizer_config = SimpleNamespace(total_training_steps=100)
    optimizer = object()
    assert engine._build_lr_scheduler(optimizer) == "scheduler"
    assert calls == [(
        {
            "total_training_steps": 500,
            "lr_warmup_steps": 10,
            "lr_scheduler_type": "cosine",
            "min_lr_ratio": 0.1,
        },
        optimizer,
    )]
    assert "BPO scheduler contract" in capsys.readouterr().out
    assert (
        getattr(_FSDPEngine._build_lr_scheduler, "_shopping_bpo_marker", None)
        == _SCHEDULER_CONTRACT_MARKER
    )


def test_scheduler_contract_forwards_keyword_arguments(monkeypatch):
    calls = []

    class _FSDPEngine:
        def _build_lr_scheduler(self, optimizer):
            calls.append(optimizer)
            return "scheduler"

    transformer = types.ModuleType("verl.workers.engine.fsdp.transformer_impl")
    transformer.FSDPEngine = _FSDPEngine
    monkeypatch.setitem(sys.modules, transformer.__name__, transformer)
    monkeypatch.setenv("SHOPPING_BPO_SCHEDULER_HORIZON", "500")
    monkeypatch.setenv("SHOPPING_BPO_WARMUP_STEPS", "10")
    monkeypatch.setenv("SHOPPING_BPO_MIN_LR_RATIO", "0.1")

    install_scheduler_contract()
    engine = _FSDPEngine()
    engine.optimizer_config = SimpleNamespace(total_training_steps=200)
    optimizer = object()
    assert engine._build_lr_scheduler(optimizer=optimizer) == "scheduler"
    assert calls == [optimizer]
