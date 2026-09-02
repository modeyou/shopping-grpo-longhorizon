#!/usr/bin/env python3
"""在加载模型前拒绝污染或版本不匹配的 GRPO 环境。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import py_compile
import shutil
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
    except ImportError:  # Direct execution from the scripts directory.
        from apply_verl_dynamic_sampling_patch import EXPECTED_PATCHED_SHA256

    actual_patch_sha256 = hashlib.sha256(ray_trainer.read_bytes()).hexdigest()
    ray_trainer_source = ray_trainer.read_text(encoding="utf-8")
    if actual_patch_sha256 != EXPECTED_PATCHED_SHA256:
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
        from shopping_grpo.training.grpo.optimizer_eligibility import (
            SHOPPER_REJECTION_EXCLUSION_REASON,
            SHOPPER_REJECTION_EXCLUSION_THRESHOLD,
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

    pathology_infos = [
        {
            "shopper_rejections": 3 if index == 1 else 0,
            "infrastructure_invalid": False,
            "reward": {
                "terminal_utility": float(index % 2),
                "purchase_success": bool(index % 2),
                "sampling_invalid": False,
            },
        }
        for index in range(8)
    ]
    utility, success, invalid, reasons = extract_shopping_group_signals(
        pathology_infos
    )
    indices, pathology_stats = select_reward_varying_groups(
        ["pathology"] * 4 + ["eligible"] * 4,
        utility,
        terminal_utilities=utility,
        purchase_success=success,
        sampling_invalid=invalid,
        sampling_invalid_reasons=reasons,
    )
    if indices != [4, 5, 6, 7]:
        raise SystemExit(
            "shopping rejection pathology preflight did not drop the whole uid group"
        )
    if (
        pathology_stats["sampling_invalid_reason_counts"].get(
            SHOPPER_REJECTION_EXCLUSION_REASON
        )
        != 1
        or pathology_stats["infrastructure_invalid_group_count"] != 0
    ):
        raise SystemExit(
            "shopping rejection pathology preflight corrupted exclusion diagnostics"
        )

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
                "shopper_rejection_exclusion_threshold": (
                    SHOPPER_REJECTION_EXCLUSION_THRESHOLD
                ),
                "shopper_rejection_exclusion_reason": (
                    SHOPPER_REJECTION_EXCLUSION_REASON
                ),
                "ray_trainer": str(ray_trainer),
                "marker": PATCH_MARKER,
                "sha256": actual_patch_sha256,
            },
            sort_keys=True,
        )
    )


def validate_formal_training_contract(config):
    """Freeze the one supported 500-update native GRPO recipe."""
    optim = config.actor_rollout_ref.actor.optim
    model = config.actor_rollout_ref.model
    checks = {
        "total_training_steps": int(config.trainer.total_training_steps) == 500,
        "save_freq": int(config.trainer.save_freq) == 25,
        "test_freq": int(config.trainer.test_freq) == 50,
        "val_before_train": bool(config.trainer.val_before_train),
        "lr": math.isclose(float(optim.lr), 1.0e-6, rel_tol=0, abs_tol=1e-12),
        "warmup_steps_delegated": int(optim.lr_warmup_steps) == -1,
        "warmup_ratio": math.isclose(float(optim.lr_warmup_steps_ratio), 0.03),
        "scheduler": str(optim.lr_scheduler_type) == "constant",
        "use_remove_padding": bool(model.use_remove_padding),
        "use_liger": bool(model.use_liger),
        "use_fused_kernels": bool(model.use_fused_kernels),
        "fused_impl_backend": str(model.fused_kernel_options.impl_backend) == "torch",
        "dataloader_num_workers": int(config.data.dataloader_num_workers) == 0,
        "train_batch_size": int(config.data.train_batch_size) == 2,
        "validation_rows_per_batch": int(config.data.val_batch_size) == 2,
        "rollout_n": int(config.actor_rollout_ref.rollout.n) == 4,
        "native_reward": not bool(config.algorithm.use_kl_in_reward),
    }
    failed = sorted(name for name, accepted in checks.items() if not accepted)
    if failed:
        raise SystemExit("formal GRPO contract mismatch: " + ", ".join(failed))
    print("formal 500-update GRPO contract preflight passed: " + json.dumps(checks, sort_keys=True))


def validate_environment_concurrency(config):
    """Keep whole-batch validation below the ShopSimulator lease capacity."""
    workers = int(config.actor_rollout_ref.rollout.agent.num_workers)
    per_worker = int(os.environ.get("SHOPPING_ENV_CONCURRENCY_PER_WORKER", "0"))
    slots = int(os.environ.get("SHOPPING_EXPECTED_SHOPSIM_SLOTS", "0"))
    maximum = workers * per_worker
    if workers != 8 or per_worker != 2 or slots != 20 or maximum > slots:
        raise SystemExit(
            "formal GRPO environment concurrency mismatch: "
            f"workers={workers}, per_worker={per_worker}, slots={slots}"
        )
    print(
        "GRPO environment-concurrency preflight passed: "
        + json.dumps(
            {
                "agent_loop_workers": workers,
                "per_worker": per_worker,
                "maximum_concurrent_leases": maximum,
                "expected_shopsim_slots": slots,
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
    if str(config.trainer.get("project_name")) != "shopping-multiturn-grpo-sft200":
        raise SystemExit(
            "SFT-200 Reward v4 GRPO SwanLab project must be "
            "shopping-multiturn-grpo-sft200"
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
    if profile != "none":
        raise SystemExit("formal GRPO supports only native Reward v4 profile=none")
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


def validate_fused_ppo_gradient_patch(torch, verl_source: Path):
    """Verify the backport and reproduce the former non-contiguous grad drop."""
    try:
        from scripts.apply_verl_fused_ppo_grad_patch import expected_patched_sha256
    except ImportError:  # Direct execution from the scripts directory.
        from apply_verl_fused_ppo_grad_patch import expected_patched_sha256
    from shopping_grpo.training.grpo.fused_ppo_grad_patch import PATCH_MARKER as marker

    target = verl_source.parent / "utils/experimental/torch_functional.py"
    if not target.is_file():
        raise SystemExit(f"GRPO fused-PPO source is missing: {target}")
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        expected = expected_patched_sha256(target)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if actual != expected or target.read_text(encoding="utf-8").count(marker) != 1:
        raise SystemExit(
            "GRPO fused-PPO gradient patch mismatch; "
            "run scripts/apply_verl_fused_ppo_grad_patch.py first"
        )
    from verl.utils.experimental import torch_functional as fused

    base = torch.randn(2, 3, 5, dtype=torch.float32, requires_grad=True)
    hidden_states = base.transpose(0, 1)
    if hidden_states.is_contiguous():
        raise SystemExit("fused-PPO gradient probe must be non-contiguous")
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
        raise SystemExit("fused-PPO gradient probe produced no finite input gradient")
    gradient_abs_sum = float(base.grad.abs().sum().item())
    if gradient_abs_sum <= 0:
        raise SystemExit("fused-PPO gradient probe produced an all-zero gradient")
    print(
        "GRPO fused-PPO input-gradient preflight passed: "
        + json.dumps({"gradient_abs_sum": gradient_abs_sum, "sha256": actual}, sort_keys=True)
    )


def validate_liger_integration(config, verl_source: Path):
    """Require veRL's supported Liger-plus-fused-output-head integration."""
    try:
        installed = version("liger_kernel")
    except PackageNotFoundError as exc:
        raise SystemExit("formal GRPO requires liger-kernel>=0.8.2") from exc
    numeric = tuple(int(part) for part in installed.split("+", 1)[0].split(".")[:3])
    if numeric < (0, 8, 2):
        raise SystemExit(f"formal GRPO requires liger-kernel>=0.8.2, got {installed}")
    candidates = (
        verl_source.parent / "workers/fsdp_workers.py",
        verl_source.parent / "workers/engine/fsdp/fsdp_workers.py",
        verl_source.parent / "workers/engine/fsdp/transformer_impl.py",
    )
    sources = [path.read_text(encoding="utf-8") for path in candidates if path.is_file()]
    if not sources or not any(
        "use_liger" in source
        and "fused_linear_cross_entropy=False" in source
        and "apply_monkey_patch" in source
        for source in sources
    ):
        raise SystemExit(
            "installed veRL does not separate Liger kernels from its fused PPO output head"
        )
    model = config.actor_rollout_ref.model
    if not (
        bool(model.use_liger)
        and bool(model.use_fused_kernels)
        and bool(model.use_remove_padding)
        and str(model.fused_kernel_options.impl_backend) == "torch"
    ):
        raise SystemExit("formal GRPO fused/Liger/remove-padding contract is incomplete")
    print(
        "GRPO Liger/fused/remove-padding integration preflight passed: "
        + json.dumps({"liger_kernel": installed, "fused_impl_backend": "torch"}, sort_keys=True)
    )


def validate_independent_grpo_environment(verl_source: Path):
    """Reject a runtime carrying BPO-only environment state or patch markers."""
    if os.environ.get("SHOPPING_GRPO_ENV_ROLE") != "formal-grpo-v1":
        raise SystemExit("formal GRPO requires SHOPPING_GRPO_ENV_ROLE=formal-grpo-v1")
    forbidden_environment = sorted(
        name for name in os.environ if name.startswith("SHOPPING_BPO_")
    )
    if forbidden_environment:
        raise SystemExit("formal GRPO environment contains BPO variables: " + ", ".join(forbidden_environment))
    contaminated = []
    for target in verl_source.parent.rglob("*.py"):
        source = target.read_text(encoding="utf-8")
        if "SHOPPING_BPO_" in source or "CARL_BPO" in source:
            contaminated.append(str(target))
    if contaminated:
        raise SystemExit(
            "BPO-only patch marker found in GRPO environment: "
            + ", ".join(contaminated)
        )
    print(
        "independent GRPO environment preflight passed: "
        + json.dumps({"python": sys.executable, "prefix": sys.prefix}, sort_keys=True)
    )


def validate_checkpoint_disk_budget():
    """Reserve a conservative 20-checkpoint budget before loading the model."""
    model_root = Path(os.environ["GRPO_MODEL_PATH"])
    output = Path(os.environ["GRPO_OUTPUT_DIR"])
    output.parent.mkdir(parents=True, exist_ok=True)
    model_bytes = sum(path.stat().st_size for path in model_root.rglob("*") if path.is_file())
    checkpoint_count = 20
    required_bytes = model_bytes * 3 * checkpoint_count + 20 * 1024**3
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_bytes:
        raise SystemExit(
            f"insufficient disk for 20 retained checkpoints: required={required_bytes}, free={free_bytes}"
        )
    print(
        "GRPO checkpoint disk-budget preflight passed: "
        + json.dumps(
            {"checkpoint_count": checkpoint_count, "free_bytes": free_bytes, "required_bytes": required_bytes},
            sort_keys=True,
        )
    )


def validate_no_external_grpo_ray_processes():
    """Reject live stale clusters while ignoring this preflight's parent launcher."""
    import psutil

    allowed = {os.getpid()}
    parent = psutil.Process(os.getpid()).parent()
    while parent is not None:
        allowed.add(parent.pid)
        parent = parent.parent()
    markers = (
        "verl.trainer.main_ppo", "raylet", "gcs_server",
        "ray::TaskRunner", "ray::WorkerDict",
    )
    active = []
    for process in psutil.process_iter(("pid", "name", "cmdline", "status")):
        if process.info["pid"] in allowed or process.info["status"] == psutil.STATUS_ZOMBIE:
            continue
        command = " ".join(process.info.get("cmdline") or ())
        identity = f"{process.info.get('name') or ''} {command}"
        if any(marker in identity for marker in markers):
            active.append({"pid": process.info["pid"], "command": command})
    if active:
        raise SystemExit("active external GRPO/Ray processes found: " + json.dumps(active, sort_keys=True))
    print("no external GRPO/Ray process preflight passed")


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
    validate_independent_grpo_environment(verl_source)
    validate_no_external_grpo_ray_processes()
    validate_checkpoint_disk_budget()
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
    validate_formal_training_contract(config)
    validate_environment_concurrency(config)
    validate_swanlab_tracking(config)
    validate_fused_ppo_gradient_patch(torch, verl_source)
    validate_liger_integration(config, verl_source)
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
