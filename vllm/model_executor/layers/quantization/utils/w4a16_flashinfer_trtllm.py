# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert ModelOpt NVFP4 group-16 weights to MXFP4 group-32."""

from dataclasses import dataclass

import torch

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class RequantizationStats:
    elements: int
    source_nonzero: int
    changed_codes: int
    saturated_values: int
    absolute_error_sum: float
    source_absolute_sum: float
    max_absolute_error: float

    @property
    def changed_code_fraction(self) -> float:
        return self.changed_codes / self.elements

    @property
    def relative_l1_error(self) -> float:
        if self.source_absolute_sum == 0:
            return 0.0
        return self.absolute_error_sum / self.source_absolute_sum


def _e8m0_values(encoded: torch.Tensor) -> torch.Tensor:
    return (encoded.to(torch.int32) << 23).view(torch.float32)


def _decode_fp4(codes: torch.Tensor) -> torch.Tensor:
    magnitudes = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=codes.device)
    decoded = magnitudes[(codes & 0x7).to(torch.int64)]
    return torch.where((codes & 0x8) != 0, -decoded, decoded)


def _encode_fp4(values: torch.Tensor) -> torch.Tensor:
    absolute = values.abs()
    magnitudes = torch.zeros_like(absolute, dtype=torch.uint8)
    magnitudes = torch.where(absolute > 0.25, 1, magnitudes)
    magnitudes = torch.where(absolute >= 0.75, 2, magnitudes)
    magnitudes = torch.where(absolute > 1.25, 3, magnitudes)
    magnitudes = torch.where(absolute >= 1.75, 4, magnitudes)
    magnitudes = torch.where(absolute > 2.5, 5, magnitudes)
    magnitudes = torch.where(absolute >= 3.5, 6, magnitudes)
    magnitudes = torch.where(absolute > 5.0, 7, magnitudes)
    negative = torch.signbit(values) & (magnitudes != 0)
    return magnitudes | (negative.to(torch.uint8) << 3)


def requantize_nvfp4_group16_to_mxfp4_group32(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    max_chunk_elements: int = 8 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor, RequantizationStats]:
    """Requantize a packed expert matrix for an MXFP4 kernel.

    Args:
        weight: Packed E2M1 weights with shape ``[experts, rows, K / 2]``.
        block_scale: E4M3 block scales with shape
            ``[experts, rows, K / 16]``.
        global_scale: One scale per expert, or separate W1 and W3 scales.
        max_chunk_elements: Maximum decoded FP32 values converted at once.

    Returns:
        Packed E2M1 weights, E8M0 group-32 scales, and conversion statistics.

    Raises:
        ValueError: If tensor types, shapes, or scales are invalid.
    """
    if weight.dtype != torch.uint8 or weight.ndim != 3:
        raise ValueError(
            f"weight must be 3D packed uint8, found {weight.dtype} "
            f"{tuple(weight.shape)}"
        )
    if block_scale.dtype != torch.float8_e4m3fn or block_scale.ndim != 3:
        raise ValueError(
            "block_scale must be 3D float8_e4m3fn, found "
            f"{block_scale.dtype} {tuple(block_scale.shape)}"
        )
    if weight.shape[:2] != block_scale.shape[:2]:
        raise ValueError("weight and block_scale expert/row dimensions differ")
    if max_chunk_elements <= 0:
        raise ValueError(
            f"max_chunk_elements must be positive, found {max_chunk_elements}"
        )

    k = weight.shape[-1] * 2
    if k % 32 != 0 or block_scale.shape[-1] != k // 16:
        raise ValueError(
            f"expected packed K divisible by 32 and {k // 16} group-16 "
            f"scales; found K={k}, scales={block_scale.shape[-1]}"
        )

    experts, rows = weight.shape[:2]
    global_scale = global_scale.to(torch.float32)
    if global_scale.shape in ((experts,), (experts, 1)):
        per_row_global = global_scale.reshape(experts, 1).expand(experts, rows)
    elif global_scale.shape == (experts, 2):
        if rows % 2 != 0:
            raise ValueError(
                f"two W13 global scales require an even row count, found {rows}"
            )
        half = rows // 2
        per_row_global = torch.cat(
            (
                global_scale[:, 0, None].expand(experts, half),
                global_scale[:, 1, None].expand(experts, half),
            ),
            dim=1,
        )
    else:
        raise ValueError(
            "expected one global scale per expert or separate W1/W3 scales "
            f"with shape ({experts}, 2); found {tuple(global_scale.shape)}"
        )
    valid_global_scale = torch.isfinite(per_row_global) & (per_row_global > 0)
    if not bool(valid_global_scale.all().item()):
        raise ValueError("global scales must be finite and positive")

    groups32 = k // 32
    flat_weight = weight.reshape(experts * rows, k // 2)
    flat_scale = block_scale.reshape(experts * rows, k // 16)
    flat_global = per_row_global.reshape(experts * rows)
    out_weight = torch.empty_like(flat_weight)
    out_scale = torch.empty(
        experts * rows,
        groups32,
        dtype=torch.uint8,
        device=weight.device,
    )

    chunk_rows = max(1, max_chunk_elements // k)
    stats_accumulator = torch.zeros(6, dtype=torch.float64, device=weight.device)

    for start in range(0, flat_weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, flat_weight.shape[0])
        packed = flat_weight[start:end].reshape(-1, groups32, 16)
        codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
            -1, groups32, 32
        )
        source_values = _decode_fp4(codes).reshape(-1, groups32, 2, 16)
        scales16 = (
            flat_scale[start:end].to(torch.float32).reshape(-1, groups32, 2)
            * flat_global[start:end, None, None]
        )
        source_values.mul_(scales16.unsqueeze(-1))
        source_values = source_values.reshape(-1, groups32, 32)

        absolute_source = source_values.abs()
        required_scale = absolute_source.amax(dim=-1) / 6.0
        stats_accumulator[4].add_(absolute_source.sum(dtype=torch.float64))
        del absolute_source
        exponent = torch.ceil(
            torch.log2(required_scale.clamp_min(torch.finfo(torch.float32).tiny))
        )
        scale_code = (exponent.to(torch.int32) + 127).clamp_(1, 254).to(torch.uint8)
        scale_value = _e8m0_values(scale_code)
        normalized = source_values / scale_value.unsqueeze(-1)
        target_codes = _encode_fp4(normalized)
        target_values = _decode_fp4(target_codes)
        target_values.mul_(scale_value.unsqueeze(-1))

        packed_target = (
            target_codes[..., 0::2] | (target_codes[..., 1::2] << 4)
        ).reshape(end - start, k // 2)
        out_weight[start:end].copy_(packed_target)
        out_scale[start:end].copy_(scale_code)

        stats_accumulator[0].add_(torch.count_nonzero(source_values))
        stats_accumulator[1].add_(torch.count_nonzero(target_codes != codes))
        stats_accumulator[2].add_(torch.count_nonzero(normalized.abs() > 6.0))
        target_values.sub_(source_values).abs_()
        stats_accumulator[3].add_(target_values.sum(dtype=torch.float64))
        stats_accumulator[5].copy_(
            torch.maximum(stats_accumulator[5], target_values.max().to(torch.float64))
        )

    (
        source_nonzero,
        changed_codes,
        saturated_values,
        absolute_error_sum,
        source_absolute_sum,
        max_absolute_error,
    ) = stats_accumulator.cpu().tolist()

    stats = RequantizationStats(
        elements=weight.numel() * 2,
        source_nonzero=int(source_nonzero),
        changed_codes=int(changed_codes),
        saturated_values=int(saturated_values),
        absolute_error_sum=absolute_error_sum,
        source_absolute_sum=source_absolute_sum,
        max_absolute_error=max_absolute_error,
    )
    return (
        out_weight.reshape_as(weight),
        out_scale.reshape(experts, rows, groups32),
        stats,
    )
