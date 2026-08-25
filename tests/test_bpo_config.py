from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.check_bpo_runtime import validate_visible_gpu_headroom


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
    assert model["use_fused_kernels"] is False
    assert model["use_liger"] is True
    assert model["use_remove_padding"] is True
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
