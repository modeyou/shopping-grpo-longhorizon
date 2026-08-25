"""veRL 0.8 的窄范围运行时兼容。"""

import os


SPARSE_CUDA_MAPPING_MARKER = "SHOPPING_GRPO_SPARSE_CUDA_MAPPING_V1"


def install_torch_padding_fallback():
    """用 veRL 自带的纯 PyTorch 实现替代 FlashAttention padding 工具。"""
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions


def parse_visible_cuda_devices(value):
    """校验 CUDA 可见设备掩码，同时保留用户指定的物理卡顺序。"""
    devices = [item.strip() for item in str(value).split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain unique non-empty device identifiers"
        )
    return devices


def cuda_logical_ordinal(accelerator_id, visible_devices):
    """将 Ray 物理 GPU ID 映射到 CUDA 掩码内的逻辑 ordinal。"""
    devices = parse_visible_cuda_devices(visible_devices)
    accelerator_id = str(accelerator_id).strip()
    if accelerator_id in devices:
        return devices.index(accelerator_id)
    try:
        logical = int(accelerator_id)
    except ValueError as exc:
        raise ValueError(
            f"Ray accelerator id {accelerator_id!r} is absent from "
            "CUDA_VISIBLE_DEVICES"
        ) from exc
    if 0 <= logical < len(devices):
        return logical
    raise ValueError(
        f"Ray accelerator id {accelerator_id!r} is outside the logical CUDA namespace"
    )


def install_sparse_cuda_mapping():
    """让 veRL colocated worker 支持非连续的物理 GPU ID。"""
    import ray
    from verl.single_controller.base.worker import Worker
    from verl.utils.device import get_torch_device, get_visible_devices_keyword
    from verl.utils.ray_utils import ray_noset_visible_devices

    current = Worker._setup_env_cuda_visible_devices
    if getattr(current, "_shopping_grpo_marker", None) == SPARSE_CUDA_MAPPING_MARKER:
        return

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

    setup_env_cuda_visible_devices._shopping_grpo_marker = (
        SPARSE_CUDA_MAPPING_MARKER
    )
    Worker._setup_env_cuda_visible_devices = setup_env_cuda_visible_devices


def worker_process_setup_hook():
    """在每个 Ray worker 内安装全部通用 GRPO 兼容 hook。"""
    install_torch_padding_fallback()
    install_sparse_cuda_mapping()
