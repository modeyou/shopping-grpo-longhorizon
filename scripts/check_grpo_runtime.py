#!/usr/bin/env python3
"""在加载模型前拒绝污染或版本不匹配的 GRPO 环境。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import py_compile
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path


EXPECTED_VERSIONS = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "transformers": "5.15.0.dev0",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
    "swanlab": "0.9.1",
}
EXPECTED_TRANSFORMERS_REVISION = "7ea2320c76117e6742364808a666ef6f2fb40a67"
PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V4"
MAX_SAFE_RESPONSE_LENGTH = 20480
MAX_SAFE_SEQUENCE_LENGTH = 24576
COMMON_RUNTIME_FILES = {
    "observation.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py",
    "pack_api.py": "environments/ShopSimulator/shop_env/shop_env/pack_api.py",
    "slot_lease_pool.py": "environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py",
    "web_agent_text_env.py": "environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py",
}
REWARD_RUNTIME_FILES = {
    "shopsimulator-reward-v3": {
        "reward.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py",
    },
    "shopsimulator-reward-v4": {
        "price_constraints.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/price_constraints.py",
        "reward_features_v2.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward_features_v2.py",
        "reward_registry.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward_registry.py",
        "reward_v4.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward_v4.py",
    },
}


def validate_reward_runtime_files(manifest, root):
    if manifest.get("lease_contract") != "explicit-client-release-v1":
        raise SystemExit(
            "Environment v2.1 manifest must select explicit-client-release-v1"
        )
    reward_version = (manifest.get("reward") or {}).get("version")
    runtime_files = {
        **COMMON_RUNTIME_FILES,
        **REWARD_RUNTIME_FILES.get(reward_version, {}),
    }
    expected = manifest.get("runtime_files_sha256")
    if not isinstance(expected, dict) or set(expected) != set(runtime_files):
        raise SystemExit(
            "Environment v2.1 manifest runtime_files_sha256 is missing or incomplete"
        )
    from shopping_grpo.environment.manifest import sha256_file

    mismatches = {}
    for name, relative_path in runtime_files.items():
        actual = sha256_file(Path(root) / relative_path)
        if actual != expected[name]:
            mismatches[name] = {"expected": expected[name], "actual": actual}
    if mismatches:
        raise SystemExit(
            "Environment v2.1 runtime file hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_environment_contract():
    required_version = os.environ.get(
        "SHOPPING_ENVIRONMENT_VERSION",
        "shopsimulator-environment-v2.1",
    )
    if required_version != "shopsimulator-environment-v2.1":
        raise SystemExit(
            "this repository supports only shopsimulator-environment-v2.1"
        )
    manifest_path = os.environ.get("SHOPPING_ENV_MANIFEST")
    if not manifest_path or not Path(manifest_path).is_file():
        raise SystemExit(
            f"{required_version} requires SHOPPING_ENV_MANIFEST pointing to a frozen manifest"
        )
    try:
        from shopping_grpo.environment.manifest import validate_manifest

        manifest = validate_manifest(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {required_version} manifest: {exc}") from exc
    actual_environment_version = manifest.get(
        "environment_version",
        "shopsimulator-environment-v2.1",
    )
    if actual_environment_version != required_version:
        raise SystemExit(
            "environment manifest version mismatch: "
            f"expected {required_version}, got {actual_environment_version}"
        )
    tools_path = Path(
        os.environ.get(
            "SHOPPING_TOOL_CONFIG",
            Path(__file__).resolve().parents[1]
            / "configs/tools.json",
        )
    )
    tools = json.loads(tools_path.read_text(encoding="utf-8")).get("tools", [])
    tool_names = {
        item.get("tool_schema", {}).get("function", {}).get("name")
        for item in tools
    }
    if "finish_without_purchase" not in tool_names:
        raise SystemExit("Environment v2 tool config is missing finish_without_purchase")
    if "ask_shopper" not in tool_names:
        raise SystemExit("multi-turn GRPO tool config is missing ask_shopper")
    for name in ("SHOPPER_MODEL", "SHOPPER_BASE_URL", "SHOPPER_API_KEY"):
        if not os.environ.get(name):
            raise SystemExit(f"multi-turn GRPO requires {name}")
    if int(manifest["max_steps"]) != 35:
        raise SystemExit("Environment v2 GRPO contract requires max_steps=35")
    validate_reward_runtime_files(
        manifest,
        Path(__file__).resolve().parents[1],
    )
    print(
        f"{required_version} manifest preflight passed: "
        + json.dumps(
            {
                "manifest": str(Path(manifest_path).resolve()),
                "shopsimulator_commit": manifest["shopsimulator_commit"],
                "observation_version": manifest["observation_version"],
                "reward_version": manifest["reward"]["version"],
                "search_version": manifest["search"]["version"],
                "lease_contract": manifest.get("lease_contract"),
                "runtime_file_count": len(manifest.get("runtime_files_sha256") or {}),
            },
            sort_keys=True,
        )
    )


def compose_runtime_config(overrides):
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError as exc:
        raise SystemExit(f"cannot parse GRPO config before preflight: {exc}") from exc

    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_name = os.environ.get("GRPO_CONFIG_NAME", "grpo")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


def validate_transformers_revision():
    """The Qwen3.5 runtime uses one pinned upstream Transformers revision."""
    dist = distribution("transformers")
    direct_url = Path(dist.locate_file("transformers-5.15.0.dev0.dist-info/direct_url.json"))
    if not direct_url.is_file():
        raise SystemExit(
            "cannot verify pinned Transformers revision: direct_url.json is missing"
        )
    try:
        metadata = json.loads(direct_url.read_text(encoding="utf-8"))
        revision = metadata["vcs_info"]["commit_id"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid Transformers direct_url.json: {exc}") from exc
    if revision != EXPECTED_TRANSFORMERS_REVISION:
        raise SystemExit(
            "incompatible Transformers revision: expected "
            f"{EXPECTED_TRANSFORMERS_REVISION}, got {revision}"
        )
    print(f"pinned Transformers revision preflight passed: {revision}")


def validate_dynamic_sampling(config, verl_source: Path, installed):
    dynamic_config = config.get("shopping_dynamic_sampling", {})
    if not bool(dynamic_config.get("enable", False)):
        return

    if installed.get("verl") != "0.8.0":
        raise SystemExit(
            f"shopping dynamic sampling requires verl==0.8.0, got {installed.get('verl')}"
        )
    ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not ray_trainer.is_file():
        raise SystemExit(f"cannot locate installed RayPPOTrainer source: {ray_trainer}")
    try:
        from scripts.apply_verl_dynamic_sampling_patch import EXPECTED_PATCHED_SHA256
        from scripts.apply_verl_scheduler_horizon_patch import (
            PATCH_MARKER as SCHEDULER_PATCH_MARKER,
            verify_patched as verify_scheduler_patch,
        )
    except ImportError:  # Direct execution from the scripts directory.
        from apply_verl_dynamic_sampling_patch import EXPECTED_PATCHED_SHA256
        from apply_verl_scheduler_horizon_patch import (
            PATCH_MARKER as SCHEDULER_PATCH_MARKER,
            verify_patched as verify_scheduler_patch,
        )

    actual_patch_sha256 = hashlib.sha256(ray_trainer.read_bytes()).hexdigest()
    ray_trainer_source = ray_trainer.read_text(encoding="utf-8")
    scheduler_patch_enabled = SCHEDULER_PATCH_MARKER in ray_trainer_source
    if scheduler_patch_enabled:
        try:
            verify_scheduler_patch(ray_trainer)
        except (OSError, RuntimeError, py_compile.PyCompileError) as exc:
            raise SystemExit(f"invalid scheduler-horizon patch: {exc}") from exc
    elif actual_patch_sha256 != EXPECTED_PATCHED_SHA256:
        raise SystemExit(
            "shopping dynamic-sampling patch hash mismatch: "
            f"expected {EXPECTED_PATCHED_SHA256}, got {actual_patch_sha256}; "
            "run scripts/apply_verl_dynamic_sampling_patch.py first"
        )
    if PATCH_MARKER not in ray_trainer_source:
        raise SystemExit(
            "shopping dynamic sampling is enabled but the pinned veRL patch marker is missing; "
            "run scripts/apply_verl_dynamic_sampling_patch.py first"
        )

    try:
        from shopping_grpo.training.grpo.dynamic_sampling import (
            extract_shopping_group_signals,
            select_reward_varying_groups,
        )
    except ImportError as exc:
        raise SystemExit(f"shopping dynamic sampling helper is unavailable: {exc}") from exc
    utility, success, invalid, reasons = extract_shopping_group_signals(
        [
            {
                "infrastructure_invalid": False,
                "reward": {
                    "terminal_utility": reward,
                    "purchase_success": reward > 0,
                    "sampling_invalid": False,
                },
            }
            for reward in (0.0, 1.0, 0.0, 0.0)
        ]
    )
    indices, _ = select_reward_varying_groups(
        ["preflight"] * 4,
        [0.0, 1.0, 0.0, 0.0],
        terminal_utilities=utility,
        purchase_success=success,
        sampling_invalid=invalid,
        sampling_invalid_reasons=reasons,
    )
    if indices != [0, 1, 2, 3]:
        raise SystemExit("shopping dynamic sampling helper failed its import-time sanity check")

    if dynamic_config.get("metric") != "seq_reward":
        raise SystemExit("shopping_dynamic_sampling.metric must be seq_reward")
    if int(dynamic_config.get("max_num_gen_batches", 0)) <= 0:
        raise SystemExit("shopping_dynamic_sampling.max_num_gen_batches must be positive")
    if int(dynamic_config.get("max_consecutive_skipped_updates", 0)) <= 0:
        raise SystemExit(
            "shopping_dynamic_sampling.max_consecutive_skipped_updates must be positive"
        )
    reward_tolerance = float(dynamic_config.get("reward_tolerance", -1))
    if reward_tolerance < 0 or not math.isfinite(reward_tolerance):
        raise SystemExit("shopping_dynamic_sampling.reward_tolerance must be finite and non-negative")
    if not bool(config.algorithm.rollout_correction.get("bypass_mode", False)):
        raise SystemExit("shopping dynamic sampling requires rollout_correction.bypass_mode=true")
    if not bool(config.actor_rollout_ref.rollout.get("calculate_log_probs", False)):
        raise SystemExit("shopping dynamic sampling requires rollout.calculate_log_probs=true")

    print(
        "shopping dynamic sampling preflight passed: "
        + json.dumps(
            {
                "enable": True,
                "metric": str(dynamic_config.metric),
                "max_num_gen_batches": int(dynamic_config.max_num_gen_batches),
                "max_consecutive_skipped_updates": int(
                    dynamic_config.max_consecutive_skipped_updates
                ),
                "reward_tolerance": reward_tolerance,
                "ray_trainer": str(ray_trainer),
                "marker": PATCH_MARKER,
                "sha256": actual_patch_sha256,
                "scheduler_horizon_patch": scheduler_patch_enabled,
            },
            sort_keys=True,
        )
    )


def validate_scheduler_horizon(config, verl_source: Path):
    """Bind a staged stopping point to one reproducible long-horizon schedule."""
    scheduler = config.get("shopping_scheduler", {})
    horizon = int(scheduler.get("total_training_steps", 0))
    stage_end = int(config.trainer.total_training_steps)
    optim = config.actor_rollout_ref.actor.optim
    model = config.actor_rollout_ref.model
    ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    try:
        from scripts.apply_verl_scheduler_horizon_patch import PATCH_MARKER as marker
    except ImportError:  # Direct execution from the scripts directory.
        from apply_verl_scheduler_horizon_patch import PATCH_MARKER as marker

    if not ray_trainer.is_file() or marker not in ray_trainer.read_text(encoding="utf-8"):
        raise SystemExit(
            "formal GRPO scheduler horizon requires "
            "scripts/apply_verl_scheduler_horizon_patch.py"
        )

    if horizon <= 0 or stage_end <= 0 or stage_end > horizon:
        raise SystemExit(
            "formal GRPO requires 0 < trainer.total_training_steps <= "
            "shopping_scheduler.total_training_steps"
        )
    expected = {
        "lr": 1.0e-6,
        "lr_warmup_steps": 10,
        "lr_warmup_steps_ratio": 0.0,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "num_cycles": 0.5,
    }
    actual = {
        "lr": float(optim.lr),
        "lr_warmup_steps": int(optim.lr_warmup_steps),
        "lr_warmup_steps_ratio": float(optim.lr_warmup_steps_ratio),
        "lr_scheduler_type": str(optim.lr_scheduler_type),
        "min_lr_ratio": float(optim.min_lr_ratio),
        "num_cycles": float(optim.num_cycles),
    }
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            matches = math.isclose(actual_value, expected_value, rel_tol=0, abs_tol=1e-12)
        else:
            matches = actual_value == expected_value
        if not matches:
            raise SystemExit(
                f"formal GRPO scheduler mismatch: {key} must be "
                f"{expected_value!r}, got {actual_value!r}"
            )
    if not bool(model.use_remove_padding):
        raise SystemExit("formal GRPO requires model.use_remove_padding=true")
    if not bool(model.use_liger) or not bool(model.use_fused_kernels):
        raise SystemExit(
            "formal GRPO requires model.use_liger=true and use_fused_kernels=true"
        )
    if int(config.data.dataloader_num_workers) != 0:
        raise SystemExit("formal GRPO requires data.dataloader_num_workers=0")
    print(
        "GRPO scheduler horizon preflight passed: "
        + json.dumps(
            {
                **actual,
                "stage_total_training_steps": stage_end,
                "scheduler_total_training_steps": horizon,
                "use_liger": True,
                "use_fused_kernels": True,
                "use_remove_padding": True,
                "dataloader_num_workers": 0,
            },
            sort_keys=True,
        )
    )


def validate_swanlab_tracking(config):
    """Validate SwanLab only when the user explicitly enables it."""
    logger_backends = list(config.trainer.get("logger", []))
    if "swanlab" not in logger_backends:
        return
    forbidden = {"wandb", "tracking", "vemlp_wandb"} & set(logger_backends)
    if forbidden:
        raise SystemExit(
            "Reward v4 GRPO forbids W&B logger backends: "
            + ", ".join(sorted(forbidden))
        )
    if os.environ.get("SWANLAB_MODE") != "online":
        raise SystemExit("Reward v4 GRPO requires SWANLAB_MODE=online")
    if not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit(
            "Reward v4 GRPO requires SWANLAB_API_KEY in the launching environment"
        )
    log_dir = os.environ.get("SWANLAB_LOG_DIR")
    if not log_dir:
        raise SystemExit("Reward v4 GRPO requires SWANLAB_LOG_DIR")
    resolved_log_dir = Path(log_dir).resolve()
    if str(config.trainer.get("project_name")) != "shopping-multiturn-agentic":
        raise SystemExit(
            "Reward v4 GRPO SwanLab project must be shopping-multiturn-agentic"
        )
    print(
        "SwanLab online preflight passed: "
        + json.dumps(
            {
                "api_key": "present",
                "logger": logger_backends,
                "log_dir": str(resolved_log_dir),
                "mode": "online",
                "project": str(config.trainer.project_name),
                "run_name": str(config.trainer.experiment_name),
            },
            sort_keys=True,
        )
    )


def validate_reward_shaping_profile():
    """Reject ambiguous or unsupported training-only reward profiles."""
    from shopping_grpo.training.grpo.adapter.runtime import reward_shaping_config

    profile = os.environ.get("SHOPPING_REWARD_SHAPING_PROFILE", "none")
    try:
        config = reward_shaping_config(profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    legacy_length_enabled = (
        os.environ.get("SHOPPING_LENGTH_SHAPING_ENABLE", "false").lower()
        == "true"
    )
    if profile != "none" and legacy_length_enabled:
        raise SystemExit(
            "bounded reward shaping and legacy length shaping cannot be combined"
        )
    print(
        "GRPO reward profile preflight passed: "
        + json.dumps(config, sort_keys=True)
    )


def validate_visible_gpu_headroom(torch, expected_devices: int, minimum_free_gib=20.0):
    """Require the formal run to see only the intended clean CUDA devices."""
    from shopping_grpo.training.grpo.compat import (
        cuda_logical_ordinal,
        parse_visible_cuda_devices,
    )

    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw_visible_devices:
        raise SystemExit(
            "formal GRPO requires an explicit CUDA_VISIBLE_DEVICES mask; "
            "for the current server use CUDA_VISIBLE_DEVICES=0,2,3,4"
        )
    try:
        physical_devices = parse_visible_cuda_devices(raw_visible_devices)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(physical_devices) != expected_devices:
        raise SystemExit(
            f"formal GRPO requires exactly {expected_devices} physical GPU ids in "
            f"CUDA_VISIBLE_DEVICES, got {physical_devices}"
        )
    if os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit(
            "formal GRPO requires "
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1"
        )
    logical_mapping = {
        physical: cuda_logical_ordinal(physical, raw_visible_devices)
        for physical in physical_devices
    }
    if list(logical_mapping.values()) != list(range(expected_devices)):
        raise SystemExit(f"invalid sparse CUDA logical mapping: {logical_mapping}")
    device_count = int(torch.cuda.device_count())
    if device_count != expected_devices:
        raise SystemExit(
            f"formal GRPO requires exactly {expected_devices} visible CUDA devices; "
            "set CUDA_VISIBLE_DEVICES to the intended clean GPUs"
        )
    free_gib = []
    for index in range(device_count):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        available = free_bytes / (1024 ** 3)
        free_gib.append(round(available, 2))
        if available < minimum_free_gib:
            raise SystemExit(
                f"visible CUDA device {index} has only {available:.2f} GiB free; "
                f"formal GRPO requires at least {minimum_free_gib:.2f} GiB per GPU"
            )
    print(
        "GRPO visible-GPU headroom preflight passed: "
        + json.dumps(
            {
                "device_count": device_count,
                "physical_devices": physical_devices,
                "physical_to_logical": logical_mapping,
                "free_gib": free_gib,
                "minimum_free_gib": minimum_free_gib,
            },
            sort_keys=True,
        )
    )
    return {
        "physical_devices": physical_devices,
        "physical_to_logical": logical_mapping,
        "free_gib": free_gib,
    }


def ppo_gradient_accumulation_steps(mini_batch_size: int, micro_batch_size: int) -> int:
    mini = int(mini_batch_size)
    micro = int(micro_batch_size)
    if mini <= 0 or micro <= 0:
        raise ValueError("PPO mini and micro batch sizes must be positive")
    if mini % micro:
        raise ValueError("PPO mini batch size must be divisible by micro batch size")
    return mini // micro


def validate_grpo_seeds(config):
    data_seed = config.data.seed
    actor_seed = config.actor_rollout_ref.actor.data_loader_seed
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (data_seed, actor_seed)
    ):
        raise SystemExit(
            "data.seed and actor.data_loader_seed must resolve to integers, "
            f"got {type(data_seed).__name__} and {type(actor_seed).__name__}"
        )
    environment_seed = int(os.environ.get("GRPO_SEED", data_seed))
    if data_seed < 0 or actor_seed < 0:
        raise SystemExit("GRPO seeds must be non-negative")
    if len({data_seed, actor_seed, environment_seed}) != 1:
        raise SystemExit(
            "data.seed, actor.data_loader_seed and GRPO_SEED must match"
        )
    print(f"GRPO seed preflight passed: {data_seed}")


def validate_training_memory_budget(config):
    prompt_length = int(config.data.max_prompt_length)
    response_length = int(config.data.max_response_length)
    total_length = prompt_length + response_length
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    reference = config.actor_rollout_ref.ref

    if response_length > MAX_SAFE_RESPONSE_LENGTH:
        raise SystemExit(
            "unsafe GRPO response budget: "
            f"max_response_length={response_length} exceeds {MAX_SAFE_RESPONSE_LENGTH}"
        )
    if total_length > MAX_SAFE_SEQUENCE_LENGTH:
        raise SystemExit(
            "unsafe GRPO sequence budget: "
            f"max_prompt_length + max_response_length = {total_length}, "
            f"limit is {MAX_SAFE_SEQUENCE_LENGTH}"
        )
    for name, value in (
        ("rollout.max_model_len", int(rollout.max_model_len)),
        ("rollout.max_num_batched_tokens", int(rollout.max_num_batched_tokens)),
        (
            "rollout.log_prob_max_token_len_per_gpu",
            int(rollout.log_prob_max_token_len_per_gpu),
        ),
        ("actor.ppo_max_token_len_per_gpu", int(actor.ppo_max_token_len_per_gpu)),
        ("ref.log_prob_max_token_len_per_gpu", int(reference.log_prob_max_token_len_per_gpu)),
    ):
        if value != MAX_SAFE_SEQUENCE_LENGTH:
            raise SystemExit(
                f"unsafe or inconsistent GRPO memory budget: {name} must equal "
                f"{MAX_SAFE_SEQUENCE_LENGTH}, got {value}"
            )
    if bool(actor.use_dynamic_bsz):
        raise SystemExit(
            "actor.use_dynamic_bsz must be false so configured PPO micro batches are enforced"
        )
    actor_micro_batch_size = int(actor.ppo_micro_batch_size_per_gpu)
    actor_mini_batch_size = int(actor.ppo_mini_batch_size)
    try:
        gradient_accumulation_steps = ppo_gradient_accumulation_steps(
            actor_mini_batch_size,
            actor_micro_batch_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if bool(rollout.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "rollout.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(rollout.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("rollout.log_prob_micro_batch_size_per_gpu must equal 1")
    if bool(reference.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "ref.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(reference.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("ref.log_prob_micro_batch_size_per_gpu must equal 1")

    print(
        "GRPO training memory budget preflight passed: "
        + json.dumps(
            {
                "max_prompt_length": prompt_length,
                "max_response_length": response_length,
                "max_sequence_length": total_length,
                "actor_mini_batch_size": actor_mini_batch_size,
                "actor_micro_batch_size_per_gpu": actor_micro_batch_size,
                "actor_gradient_accumulation_steps": gradient_accumulation_steps,
                "actor_dynamic_batch": False,
                "rollout_log_prob_micro_batch_size_per_gpu": 1,
                "rollout_log_prob_dynamic_batch": False,
                "reference_micro_batch_size_per_gpu": 1,
                "reference_dynamic_batch": False,
            },
            sort_keys=True,
        )
    )


def main():
    config = compose_runtime_config(sys.argv[1:])
    validate_environment_contract()
    validate_reward_shaping_profile()
    validate_grpo_seeds(config)
    required_paths = {
        "GRPO_TRAIN_FILE": os.environ.get("GRPO_TRAIN_FILE"),
        "GRPO_VAL_FILE": os.environ.get("GRPO_VAL_FILE"),
    }
    missing = [name for name, value in required_paths.items() if not value or not Path(value).is_file()]
    if missing:
        raise SystemExit("missing GRPO parquet file(s): " + ", ".join(missing))
    validate_training_memory_budget(config)
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"incompatible Python: expected 3.12, got {sys.version.split()[0]}")

    installed = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise SystemExit(f"missing GRPO dependency: {package}=={expected}") from exc
        if installed[package].split("+", 1)[0] != expected:
            raise SystemExit(
                f"incompatible GRPO dependency: expected {package}=={expected}, got {installed[package]}"
            )
    validate_transformers_revision()

    try:
        import torch
        import verl
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
        from shopping_grpo.training.grpo.compat import (
            SPARSE_CUDA_MAPPING_MARKER,
            install_sparse_cuda_mapping,
            install_torch_padding_fallback,
        )
        from verl.single_controller.base.worker import Worker
        from verl.tools.base_tool import BaseTool
        from verl.utils.tracking import Tracking
    except ImportError as exc:
        raise SystemExit(
            "incompatible veRL 0.8 install: required AgentLoop/Tool APIs are unavailable; "
            f"original error: {exc}"
        ) from exc

    verl_source = Path(verl.__file__).resolve()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the GRPO environment")
    gpu_audit = validate_visible_gpu_headroom(
        torch,
        expected_devices=int(config.trainer.n_gpus_per_node),
    )
    if (
        not issubclass(ShoppingToolAgentLoop, ToolAgentLoop)
        or not issubclass(ShopSimulatorTool, BaseTool)
        or AgentState.TERMINATED.value != "terminated"
        or not hasattr(ToolAgentLoop, "_handle_processing_tools_state")
    ):
        raise SystemExit("incompatible veRL ToolAgentLoop lifecycle API")
    if "qwen3_coder" not in ToolParser._registry:
        raise SystemExit("veRL 0.8 built-in qwen3_coder parser is unavailable")
    if "swanlab" not in Tracking.supported_backend:
        raise SystemExit("veRL 0.8 SwanLab tracking backend is unavailable")
    validate_dynamic_sampling(config, verl_source, installed)
    validate_scheduler_horizon(config, verl_source)
    validate_swanlab_tracking(config)
    install_torch_padding_fallback()
    install_sparse_cuda_mapping()
    if (
        getattr(
            Worker._setup_env_cuda_visible_devices,
            "_shopping_grpo_marker",
            None,
        )
        != SPARSE_CUDA_MAPPING_MARKER
    ):
        raise SystemExit("GRPO sparse CUDA worker hook was not installed")
    print(
        "GRPO sparse-CUDA mapping preflight passed: "
        + json.dumps(gpu_audit, sort_keys=True)
    )
    print(
        "GRPO runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in installed.items())
        + f", source={verl_source}"
    )


if __name__ == "__main__":
    main()
