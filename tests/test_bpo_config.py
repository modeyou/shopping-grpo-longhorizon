import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.check_bpo_runtime import validate_visible_gpu_headroom
from shopping_grpo.training.bpo.runtime import (
    _SPARSE_CUDA_MAPPING_MARKER,
    cuda_logical_ordinal,
    install_sparse_cuda_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


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
    assert config["trainer"]["project_name"] == "shopping-bpo"


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
