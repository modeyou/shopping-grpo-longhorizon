from pathlib import Path

import yaml


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
    assert model["use_fused_kernels"] is False
    assert model["use_liger"] is True
    assert model["use_remove_padding"] is True
    assert config["trainer"]["n_gpus_per_node"] == 4
    assert config["trainer"]["project_name"] == "shopping-bpo"
