# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)


def test_select_nvfp4_backend_override_preserves_config_backend():
    config = make_dummy_moe_config()
    config.moe_backend = "flashinfer_cutlass"
    expert_cls = MagicMock()
    expert_cls.is_supported_config.return_value = (True, None)

    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.nvfp4.map_nvfp4_backend",
            return_value=NvFp4MoeBackend.MARLIN,
        ) as map_backend,
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.nvfp4.backend_to_kernel_cls",
            return_value=[expert_cls],
        ),
    ):
        backend, selected_expert_cls = select_nvfp4_moe_backend(
            config,
            weight_key=None,
            activation_key=None,
            backend_override="marlin",
        )

    assert backend is NvFp4MoeBackend.MARLIN
    assert selected_expert_cls is expert_cls
    map_backend.assert_called_once_with("marlin")
    assert config.moe_backend == "flashinfer_cutlass"
