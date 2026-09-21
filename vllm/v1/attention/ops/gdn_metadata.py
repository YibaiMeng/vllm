# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused metadata preparation for GatedDeltaNet attention."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_full_graph_align_pure_spec_metadata_kernel(
    block_table,
    seq_lens,
    query_start_loc,
    accepted_in,
    state_out,
    sequence_mask_out,
    token_indices_out,
    query_start_out,
    accepted_out,
    block_table_row_stride: tl.constexpr,
    block_table_col_stride: tl.constexpr,
    state_row_stride: tl.constexpr,
    state_col_stride: tl.constexpr,
    num_spec_decodes,
    batch_size,
    num_spec_tokens,
    mamba_block_size: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid_rows = rows < batch_size
    spec_rows = rows < num_spec_decodes

    state_slots = tl.arange(0, STATE_BLOCK)
    valid_state_slots = state_slots < NUM_STATE_SLOTS
    seq_lens_i32 = tl.load(seq_lens + rows, mask=spec_rows, other=0).to(tl.int32)
    state_start = tl.maximum(seq_lens_i32 - 1, 0) // mamba_block_size
    state_indices = tl.load(
        block_table
        + rows[:, None] * block_table_row_stride
        + (state_start[:, None] + state_slots[None, :]) * block_table_col_stride,
        mask=spec_rows[:, None] & valid_state_slots[None, :],
        other=0,
    )
    tl.store(
        state_out
        + rows[:, None] * state_row_stride
        + state_slots[None, :] * state_col_stride,
        state_indices,
        mask=valid_rows[:, None] & valid_state_slots[None, :],
    )

    tl.store(sequence_mask_out + rows, spec_rows, mask=valid_rows)
    accepted = tl.load(accepted_in + rows, mask=spec_rows, other=1)
    tl.store(accepted_out + rows, accepted, mask=valid_rows)

    valid_query_starts = rows <= batch_size
    input_query_starts = tl.load(
        query_start_loc + rows,
        mask=valid_query_starts,
        other=0,
    )
    spec_query_end = tl.load(query_start_loc + num_spec_decodes)
    query_starts = tl.where(
        rows <= num_spec_decodes, input_query_starts, spec_query_end
    )
    tl.store(query_start_out + rows, query_starts, mask=valid_query_starts)

    tl.store(token_indices_out + rows, rows, mask=rows < num_spec_tokens)


def prepare_full_graph_align_pure_spec_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_in: torch.Tensor,
    state_out: torch.Tensor,
    sequence_mask_out: torch.Tensor,
    token_indices_out: torch.Tensor,
    query_start_out: torch.Tensor,
    accepted_out: torch.Tensor,
    num_spec_decodes: int,
    batch_size: int,
    num_spec_tokens: int,
    mamba_block_size: int,
) -> None:
    """Populate persistent GDN metadata for a pure speculative decode batch."""
    block = 64
    grid = (triton.cdiv(max(batch_size + 1, num_spec_tokens), block),)
    _prepare_full_graph_align_pure_spec_metadata_kernel[grid](
        block_table,
        seq_lens,
        query_start_loc,
        accepted_in,
        state_out,
        sequence_mask_out,
        token_indices_out,
        query_start_out,
        accepted_out,
        block_table.stride(0),
        block_table.stride(1),
        state_out.stride(0),
        state_out.stride(1),
        num_spec_decodes,
        batch_size,
        num_spec_tokens,
        mamba_block_size,
        NUM_STATE_SLOTS=state_out.shape[1],
        STATE_BLOCK=triton.next_power_of_2(state_out.shape[1]),
        BLOCK=block,
    )
