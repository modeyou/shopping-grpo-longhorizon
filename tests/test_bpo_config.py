from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_formal_bpo_config_is_independent_and_frozen():
    path = ROOT / "configs/bpo.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert path != ROOT / "configs/grpo.yaml"
    assert config["algorithm"]["adv_estimator"] == "bpo"
    assert config["algorithm"]["bpo"] == {
        "sibling_count": 4,
        "branch_count": 1,
        "upstream_lambda": 0.95,
    }
    assert config["actor_rollout_ref"]["rollout"]["n"] == 4
    assert config["actor_rollout_ref"]["rollout"]["agent"]["num_workers"] == 2
    assert config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["max_logprobs"] == -1
    model = config["actor_rollout_ref"]["model"]
    assert model["use_fused_kernels"] is True
    assert model["use_liger"] is True
    assert config["trainer"]["total_training_steps"] == 10
    assert config["trainer"]["n_gpus_per_node"] == 4
