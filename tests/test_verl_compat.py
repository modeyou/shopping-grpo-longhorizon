"""veRL 不应为了纯 padding 操作强制依赖 FlashAttention。"""

import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class VerlCompatTest(unittest.TestCase):
    def test_sparse_physical_cuda_ids_map_to_masked_logical_ordinals(self):
        from shopping_grpo.training.grpo.compat import (
            cuda_logical_ordinal,
            parse_visible_cuda_devices,
        )

        visible = "0,2,3,4"
        self.assertEqual(parse_visible_cuda_devices(visible), ["0", "2", "3", "4"])
        self.assertEqual(
            [cuda_logical_ordinal(value, visible) for value in ("0", "2", "3", "4")],
            [0, 1, 2, 3],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_visible_cuda_devices("0,2,2,4")

    def test_sparse_cuda_worker_hook_selects_masked_logical_device(self):
        from shopping_grpo.training.grpo.compat import (
            SPARSE_CUDA_MAPPING_MARKER,
            install_sparse_cuda_mapping,
        )

        selected = []

        class Worker:
            def _setup_env_cuda_visible_devices(self):
                raise AssertionError("pinned veRL fallback must not run")

        ray = ModuleType("ray")
        ray.get_runtime_context = lambda: SimpleNamespace(
            get_accelerator_ids=lambda: {"GPU": ["4"]}
        )
        worker_module = ModuleType("verl.single_controller.base.worker")
        worker_module.Worker = Worker
        device_module = ModuleType("verl.utils.device")
        device_module.get_torch_device = lambda: SimpleNamespace(
            set_device=lambda value: selected.append(value)
        )
        device_module.get_visible_devices_keyword = lambda: "CUDA_VISIBLE_DEVICES"
        ray_utils = ModuleType("verl.utils.ray_utils")
        ray_utils.ray_noset_visible_devices = lambda: True
        modules = {
            "ray": ray,
            "verl": ModuleType("verl"),
            "verl.single_controller": ModuleType("verl.single_controller"),
            "verl.single_controller.base": ModuleType("verl.single_controller.base"),
            "verl.single_controller.base.worker": worker_module,
            "verl.utils": ModuleType("verl.utils"),
            "verl.utils.device": device_module,
            "verl.utils.ray_utils": ray_utils,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2,3,4"}),
        ):
            install_sparse_cuda_mapping()
            Worker()._setup_env_cuda_visible_devices()
            self.assertEqual(os.environ["LOCAL_RANK"], "3")

        self.assertEqual(selected, [3])
        self.assertEqual(
            getattr(
                Worker._setup_env_cuda_visible_devices,
                "_shopping_grpo_marker",
                None,
            ),
            SPARSE_CUDA_MAPPING_MARKER,
        )

    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils

        with patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
            },
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            install_torch_padding_fallback()

        self.assertEqual(attention._get_attention_functions(), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
