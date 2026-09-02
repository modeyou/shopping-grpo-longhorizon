#!/usr/bin/env python3
"""Run the repository's single supported Shopping Agent GRPO recipe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

from shopping_grpo.training.grpo.data_manifest import validate_grpo_data_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/environment-v4.json"
DEFAULT_MODEL = ROOT / "outputs/models/sft-merged"
DEFAULT_TRAIN_DATA = ROOT / "data/grpo/formal-v2/multiturn-train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/grpo/formal-v2/multiturn-validation.parquet"


DEFAULT_DATA_MANIFEST = ROOT / "data/grpo/formal-v2/manifest.json"


def validate_launcher_owned_ray(environ=None):
    """Reject attachment to a Ray cluster not created by this launch."""
    environment = os.environ if environ is None else environ
    address = str(environment.get("RAY_ADDRESS", "")).strip()
    if address:
        raise SystemExit(
            "formal GRPO requires a launcher-owned local Ray runtime; "
            "stop the manually started Ray head and unset RAY_ADDRESS"
        )


def _model_has_weights(path: Path) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--shopper-model", default="deepseek-v4-flash-0731")
    parser.add_argument("--shopper-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="resume optimizer/scheduler/model state from a checkpoint inside --output",
    )
    parser.add_argument(
        "--logger",
        choices=("console", "swanlab"),
        default="swanlab",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("GRPO_SEED", "20260823")),
    )
    parser.add_argument(
        "--reward-profile",
        choices=("none",),
        default="none",
        help="native Reward v4; shaping profiles are not supported",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=DEFAULT_DATA_MANIFEST,
        help="accepted formal Reward-v4 GRPO data manifest",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved veRL command without running preflight or training",
    )
    execution.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the full GRPO runtime preflight without loading the model or training",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args()


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def _hydra_overrides(args: argparse.Namespace) -> list[str]:
    logger_override = (
        "trainer.logger=[console,swanlab]"
        if args.logger == "swanlab"
        else "trainer.logger=[console]"
    )
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    resume_overrides = []
    if args.resume_from_checkpoint is not None:
        resume_overrides = [
            "trainer.resume_mode=resume_path",
            f"trainer.resume_from_path={args.resume_from_checkpoint.expanduser().resolve()}",
        ]
    return [
        logger_override,
        f"trainer.experiment_name={args.experiment_name}",
        *resume_overrides,
        *extra,
    ]


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    validate_launcher_owned_ray()
    if args.require_clean_git and not args.dry_run:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
        )
        if status:
            raise SystemExit("formal GRPO requires a clean Git worktree")
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(
            "model directory has no supported weight file or sharded index: "
            f"{model}"
        )
    train_data = _validated_path(args.train_data, "train parquet")
    val_data = _validated_path(args.val_data, "validation parquet")
    config = _validated_path(args.config, "GRPO example config")
    data_manifest = args.data_manifest.expanduser().resolve()
    if not data_manifest.is_file():
        raise SystemExit(f"GRPO data manifest is missing: {data_manifest}")
    try:
        validate_grpo_data_manifest(
            data_manifest,
            train_data=train_data,
            validation_data=val_data,
            environment_manifest=DEFAULT_MANIFEST,
            root=ROOT,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid GRPO data manifest: {exc}") from exc
    output = args.output.expanduser().resolve()
    resume_checkpoint = None
    if args.resume_from_checkpoint is not None:
        resume_checkpoint = _validated_path(
            args.resume_from_checkpoint,
            "resume checkpoint",
        )
        if not resume_checkpoint.is_dir():
            raise SystemExit(f"resume checkpoint must be a directory: {resume_checkpoint}")
        if not output.exists() or not output.is_dir():
            raise SystemExit("resume requires an existing --output directory")
        if not resume_checkpoint.is_relative_to(output):
            raise SystemExit(
                "resume checkpoint must be inside the same --output directory: "
                f"{resume_checkpoint}"
            )
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        if any(output.iterdir()) and resume_checkpoint is None:
            raise SystemExit(f"output directory must be new or empty: {output}")
    if (
        args.logger == "swanlab"
        and not os.environ.get("SWANLAB_API_KEY")
        and not args.dry_run
    ):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")
    shopper_api_key = os.environ.get("SHOPPER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if (not args.shopper_base_url or not shopper_api_key) and not args.dry_run:
        raise SystemExit(
            "multi-turn GRPO requires --shopper-base-url and SHOPPER_API_KEY "
            "(OPENAI_API_KEY is accepted as a fallback)"
        )

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            # veRL's colocated FSDP workers select devices by local rank. Ray's
            # default per-actor CUDA_VISIBLE_DEVICES rewrite makes every rank see
            # a one-device namespace and can map all ranks onto cuda:0.
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "SHOP_REWARD_VERSION": "shopsimulator-reward-v4",
            "GRPO_SEED": str(args.seed),
            "SHOPPING_REWARD_SHAPING_PROFILE": str(args.reward_profile),
            "SHOPPER_MODEL": str(args.shopper_model),
            "SHOPPER_BASE_URL": str(args.shopper_base_url or ""),
            "SHOPPER_API_KEY": shopper_api_key or "",
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "GRPO_OUTPUT_DIR": str(output),
            "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(
                output / "training_diagnostics.jsonl"
            ),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "GRPO_CONFIG_NAME": config.stem,
            "GRPO_RESUME_FROM_CHECKPOINT": str(resume_checkpoint or ""),
            "SHOPPING_GRPO_ENV_ROLE": "formal-grpo-v1",
            "SHOPPING_ENV_CONCURRENCY_PER_WORKER": "2",
            "SHOPPING_EXPECTED_SHOPSIM_SLOTS": "20",
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    environment["SHOPPING_GRPO_DATA_MANIFEST"] = str(data_manifest)
    overrides = _hydra_overrides(args)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *overrides,
    ]
    return command, environment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_artifact_paths(model: Path) -> list[Path]:
    """Return every file needed to identify the exact merged model weights."""
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        index = model / "pytorch_model.bin.index.json"
    if index.is_file():
        metadata = json.loads(index.read_text(encoding="utf-8"))
        shard_names = sorted(set((metadata.get("weight_map") or {}).values()))
        if not shard_names:
            raise ValueError(f"model weight index has no shards: {index}")
        paths = [index, *(model / name for name in shard_names)]
    else:
        paths = [
            path
            for name in ("model.safetensors", "pytorch_model.bin")
            if (path := model / name).is_file()
        ]
    missing = [path for path in paths if not path.is_file()]
    if not paths or missing:
        raise ValueError(f"model artifact inventory is incomplete: {missing or model}")
    merge_manifest = model / "merge_manifest.json"
    if merge_manifest.is_file():
        paths.append(merge_manifest)
    return paths


def _runtime_inventory(environment: dict[str, str]) -> dict:
    packages = {}
    for name in (
        'torch', 'transformers', 'verl', 'vllm', 'ray',
        'liger_kernel', 'swanlab', 'numpy', 'tensordict',
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        driver = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        driver_versions = sorted(set(driver.stdout.split())) if driver.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        driver_versions = []
    return {
        'python': sys.version,
        'python_executable': sys.executable,
        'python_prefix': sys.prefix,
        'packages': packages,
        'nvidia_driver_versions': driver_versions,
    }


def _patch_inventory() -> list[dict]:
    import verl

    verl_root = Path(verl.__file__).resolve().parent
    specifications = (
        (
            'dynamic-sampling-v4',
            verl_root / 'trainer/ppo/ray_trainer.py',
            'SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V4',
            '.shopping-grpo-dynamic-sampling.orig',
        ),
        (
            'fused-ppo-input-gradient-v1',
            verl_root / 'utils/experimental/torch_functional.py',
            'SHOPPING_GRPO_FUSED_PPO_NEEDS_INPUT_GRAD_PATCH_V1',
            '.shopping-grpo-fused-grad.orig',
        ),
    )
    inventory = []
    for name, target, marker, backup_suffix in specifications:
        backup = Path(str(target) + backup_suffix)
        source = target.read_text(encoding='utf-8')
        inventory.append(
            {
                'name': name,
                'target': str(target),
                'patched_sha256': _sha256_file(target),
                'marker': marker,
                'marker_count': source.count(marker),
                'original_backup': str(backup),
                'original_sha256': _sha256_file(backup) if backup.is_file() else None,
            }
        )
    return inventory


def write_run_contract(audit: dict, environment: dict[str, str]) -> Path:
    """Persist a secret-free, machine-verifiable GRPO launch contract."""
    from shopping_grpo.training.grpo.compat import parse_visible_cuda_devices

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
    )
    git_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
    )
    input_paths = {
        "train_data": Path(environment["GRPO_TRAIN_FILE"]),
        "validation_data": Path(environment["GRPO_VAL_FILE"]),
        "grpo_config": Path(audit["config"]),
        "agent_loop_config": Path(environment["SHOPPING_AGENT_LOOP_CONFIG"]),
        "tool_config": Path(environment["SHOPPING_TOOL_CONFIG"]),
        "environment_manifest": Path(environment["SHOPPING_ENV_MANIFEST"]),
        "model_config": Path(environment["GRPO_MODEL_PATH"]) / "config.json",
    }
    input_paths["data_manifest"] = Path(
        environment["SHOPPING_GRPO_DATA_MANIFEST"]
    )
    model_artifacts = _model_artifact_paths(Path(environment["GRPO_MODEL_PATH"]))
    cuda_visible_devices = environment.get("CUDA_VISIBLE_DEVICES", "")
    cuda_physical_to_logical = {}
    if cuda_visible_devices:
        cuda_physical_to_logical = {
            physical: logical
            for logical, physical in enumerate(
                parse_visible_cuda_devices(cuda_visible_devices)
            )
        }
    contract = {
        "schema_version": "shopping-grpo-run-contract-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_commit,
            "dirty": bool(git_status),
            "diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        },
        "runtime_inventory": _runtime_inventory(environment),
        "patches": _patch_inventory(),
        "launch": audit,
        "runtime_contract": {
            "environment_version": environment["SHOPPING_ENVIRONMENT_VERSION"],
            "reward_version": environment["SHOP_REWARD_VERSION"],
            "reward_profile": environment["SHOPPING_REWARD_SHAPING_PROFILE"],
            "seed": int(environment["GRPO_SEED"]),
            "shopper_model": environment["SHOPPER_MODEL"],
            "shopper_base_url": environment["SHOPPER_BASE_URL"],
            "cuda_visible_devices": cuda_visible_devices,
            "cuda_physical_to_logical": cuda_physical_to_logical,
            "ray_preserves_cuda_visible_devices": environment.get(
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            )
            == "1",
            "ray_launcher_owned": not bool(environment.get("RAY_ADDRESS", "").strip()),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "environment_role": environment.get("SHOPPING_GRPO_ENV_ROLE"),
            "agent_loop_workers": 8,
            "environment_concurrency_per_worker": int(
                environment["SHOPPING_ENV_CONCURRENCY_PER_WORKER"]
            ),
            "expected_shopsim_slots": int(
                environment["SHOPPING_EXPECTED_SHOPSIM_SLOTS"]
            ),
            "swanlab_mode": environment.get("SWANLAB_MODE"),
            "swanlab_log_dir": environment.get("SWANLAB_LOG_DIR"),
            "gpu_telemetry": str(
                (Path(environment["GRPO_OUTPUT_DIR"]) / "gpu_telemetry.csv").resolve()
            ),
            "planned_checkpoint_steps": list(range(25, 501, 25)),
            "validation_steps": [0, *range(50, 501, 50)],
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for name, path in input_paths.items()
        },
    }
    contract["inputs"]["model_artifacts"] = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
        for path in model_artifacts
    ]
    resume_checkpoint = environment.get("GRPO_RESUME_FROM_CHECKPOINT")
    filename = (
        f"run_contract.resume-{Path(resume_checkpoint).name}.json"
        if resume_checkpoint
        else "run_contract.json"
    )
    destination = Path(environment["GRPO_OUTPUT_DIR"]) / filename
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite run contract: {destination}")
    destination.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def refresh_latest_checkpoint(output: Path) -> dict | None:
    '''Publish a pointer only when veRL's tracker and checkpoint agree.'''
    tracker = output / 'latest_checkpointed_iteration.txt'
    if not tracker.is_file():
        return None
    raw_step = tracker.read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'[0-9]+', raw_step):
        return None
    step = int(raw_step)
    checkpoint = output / f'global_step_{step}'
    actor = checkpoint / 'actor'
    if (
        not checkpoint.is_dir()
        or not actor.is_dir()
        or not any(path.is_file() for path in actor.rglob('*'))
    ):
        return None
    payload = {
        'schema_version': 'shopping-grpo-latest-checkpoint-v1',
        'step': step,
        'path': str(checkpoint.resolve()),
        'tracker': str(tracker.resolve()),
        'observed_at': datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / 'latest_checkpoint.json', payload)
    return payload


def sample_gpu_telemetry(output: Path, environment: dict[str, str]) -> None:
    '''Append one physical-GPU sample without logging secrets.'''
    query = (
        'index,uuid,memory.used,memory.total,utilization.gpu,'
        'temperature.gpu,power.draw'
    )
    result = subprocess.run(
        [
            'nvidia-smi',
            f'--query-gpu={query}',
            '--format=csv,noheader,nounits',
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f'nvidia-smi telemetry failed: {result.stderr.strip()}')
    path = output / 'gpu_telemetry.csv'
    if not path.exists():
        path.write_text(
            'observed_at,index,uuid,memory_used_mib,memory_total_mib,'
            'utilization_gpu_pct,temperature_c,power_draw_w\n',
            encoding='utf-8',
        )
    observed_at = datetime.now(timezone.utc).isoformat()
    with path.open('a', encoding='utf-8') as handle:
        for row in result.stdout.splitlines():
            if row.strip():
                handle.write(f'{observed_at},{row}\n')


def _tee_process_output(process, log_path: Path) -> None:
    with log_path.open('a', encoding='utf-8') as log:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()


def _first_failure_marker(log_path: Path) -> str | None:
    markers = (
        'Traceback (most recent call last)', 'CUDA out of memory',
        'RuntimeError:', 'Error executing job', 'RayTaskError',
    )
    if not log_path.is_file():
        return None
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if any(marker in line for marker in markers):
            return line.strip()
    return None


def _swanlab_urls(log_path: Path) -> list[str]:
    if not log_path.is_file():
        return []
    source = log_path.read_text(encoding='utf-8', errors='replace')
    return sorted(set(re.findall(r'https://swanlab\.cn/[^\s\]\)]+', source)))


def _gpu_peak_summary(path: Path) -> dict:
    peaks = {}
    if not path.is_file():
        return peaks
    lines = path.read_text(encoding='utf-8').splitlines()[1:]
    for line in lines:
        columns = [value.strip() for value in line.split(',')]
        if len(columns) != 8:
            continue
        index = columns[1]
        try:
            numeric = [float(value) for value in columns[3:]]
        except ValueError:
            continue
        current = peaks.setdefault(
            index,
            {'memory_used_mib': 0.0, 'memory_total_mib': numeric[1],
             'utilization_gpu_pct': 0.0, 'temperature_c': 0.0, 'power_draw_w': 0.0},
        )
        for key, value in zip(
            ('memory_used_mib', 'memory_total_mib', 'utilization_gpu_pct', 'temperature_c', 'power_draw_w'),
            numeric,
        ):
            current[key] = max(current[key], value)
    return peaks


def _diagnostic_summary(path: Path) -> dict:
    summary = {'events': {}, 'optimizer_updates': 0, 'max_global_step': 0}
    if not path.is_file():
        return summary
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = str(record.get('event', 'unknown'))
        summary['events'][event] = summary['events'].get(event, 0) + 1
        summary['max_global_step'] = max(
            summary['max_global_step'], int(record.get('global_step', 0))
        )
        if event == 'optimizer_step' and int(
            (record.get('metrics') or {}).get('training/optimizer_updated', 0)
        ) == 1:
            summary['optimizer_updates'] += 1
    return summary


def _complete_checkpoint_steps(output: Path) -> list[int]:
    steps = []
    for checkpoint in output.glob('global_step_*'):
        match = re.fullmatch(r'global_step_([0-9]+)', checkpoint.name)
        actor = checkpoint / 'actor'
        if (
            match
            and actor.is_dir()
            and any(path.is_file() for path in actor.rglob('*'))
        ):
            steps.append(int(match.group(1)))
    return sorted(steps)


def _raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def run_supervised(command, environment, output: Path, interval_seconds=30) -> int:
    '''Run veRL while retaining GPU telemetry and the latest complete checkpoint.'''
    training_log = output / 'training.log'
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_thread = threading.Thread(
        target=_tee_process_output,
        args=(process, training_log),
        daemon=True,
    )
    output_thread.start()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    started_at = datetime.now(timezone.utc)
    telemetry_error = None
    interrupted = False
    try:
        while process.poll() is None:
            try:
                sample_gpu_telemetry(output, environment)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                telemetry_error = f'{type(exc).__name__}: {exc}'
            refresh_latest_checkpoint(output)
            try:
                process.wait(timeout=interval_seconds)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        interrupted = True
        process.terminate()
        process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    exit_code = 130 if interrupted else int(process.returncode)
    output_thread.join(timeout=30)
    latest = refresh_latest_checkpoint(output)
    telemetry_path = output / 'gpu_telemetry.csv'
    diagnostics_path = output / 'training_diagnostics.jsonl'
    diagnostics = _diagnostic_summary(diagnostics_path)
    checkpoint_steps = _complete_checkpoint_steps(output)
    expected_checkpoint_steps = list(range(25, 501, 25))
    completion_valid = (
        exit_code == 0
        and diagnostics['optimizer_updates'] == 500
        and checkpoint_steps == expected_checkpoint_steps
    )
    if exit_code == 0 and not completion_valid:
        exit_code = 4
    summary = {
        'schema_version': 'shopping-grpo-run-summary-v1',
        'started_at': started_at.isoformat(),
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'exit_code': exit_code,
        'status': 'completed' if completion_valid else ('interrupted' if interrupted else 'failed'),
        'completion_valid': completion_valid,
        'diagnostics': diagnostics,
        'complete_checkpoint_steps': checkpoint_steps,
        'expected_checkpoint_steps': expected_checkpoint_steps,
        'latest_complete_checkpoint': latest,
        'gpu_telemetry': str(telemetry_path.resolve()),
        'gpu_peaks': _gpu_peak_summary(telemetry_path),
        'training_log': str(training_log.resolve()),
        'swanlab_urls': _swanlab_urls(training_log),
        'first_failure_marker': _first_failure_marker(training_log),
        'telemetry_error': telemetry_error,
    }
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    _atomic_json(output / f'run_summary.{suffix}.json', summary)
    return exit_code


def main() -> None:
    args = parse_args()
    command, environment = build_command(args)
    audit = {
        "command": command,
        "model": environment["GRPO_MODEL_PATH"],
        "train_data": environment["GRPO_TRAIN_FILE"],
        "val_data": environment["GRPO_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["GRPO_OUTPUT_DIR"],
        "logger": args.logger,
        "reward_profile": environment["SHOPPING_REWARD_SHAPING_PROFILE"],
        "seed": int(environment["GRPO_SEED"]),
        "shopper_model": environment["SHOPPER_MODEL"],
        "shopper_base_url": environment["SHOPPER_BASE_URL"],
        "config": str(args.config.resolve()),
        "resume_from_checkpoint": environment["GRPO_RESUME_FROM_CHECKPOINT"] or None,
    }
    audit["data_manifest"] = environment["SHOPPING_GRPO_DATA_MANIFEST"]
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    Path(environment["GRPO_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    overrides = _hydra_overrides(args)
    preflight = [
        sys.executable,
        str(ROOT / "scripts/check_grpo_runtime.py"),
        *overrides,
    ]
    preflight_status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if preflight_status:
        raise SystemExit(preflight_status)
    if args.preflight_only:
        print("GRPO runtime preflight-only passed")
        return
    contract_path = write_run_contract(audit, environment)
    print(f"GRPO run contract written: {contract_path}")
    raise SystemExit(
        run_supervised(command, environment, Path(environment['GRPO_OUTPUT_DIR']))
    )


if __name__ == "__main__":
    main()
