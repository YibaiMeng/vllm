# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass, fields
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def _create_full_graph_align_builder(
    num_speculative_tokens: int,
) -> GDNAttentionMetadataBuilder:
    """Create a CUDA builder for aligned MTP decode metadata."""
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            max_cudagraph_capture_size=None,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        ),
        additional_config={"gdn_prefill_backend": "triton"},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=128)
        ),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=num_speculative_tokens,
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=CUDA_DEVICE,
    )


def _make_padded_pure_spec_common(
    num_reqs: int,
    padded_reqs: int,
    num_speculative_tokens: int,
    padded_tokens: int,
    seq_len_offset: int = 0,
    optimistic_cpu_seq_lens: bool = False,
):
    """Build a padded MTP decode batch with distinct rows for every request."""
    assert num_reqs <= padded_reqs
    query_len = num_speculative_tokens + 1
    seq_lens = [2112 + seq_len_offset + i for i in range(num_reqs)]
    seq_lens.extend([0] * (padded_reqs - num_reqs))
    query_lens = [query_len] * num_reqs + [0] * (padded_reqs - num_reqs)
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, CUDA_DEVICE, arange_block_indices=True
    )

    block_width = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_width += num_speculative_tokens
    block_table = torch.arange(
        padded_reqs * block_width, dtype=torch.int32, device=CUDA_DEVICE
    ).reshape(padded_reqs, block_width)
    common = common.replace(
        block_table_tensor=block_table.add_(1),
        num_actual_tokens=padded_tokens,
    )
    if optimistic_cpu_seq_lens:
        common = common.replace(
            seq_lens_cpu_upper_bound=(common.seq_lens.cpu() + 1),
        )

    accepted = torch.tensor(
        [1 + (i % query_len) for i in range(num_reqs)] + [1] * (padded_reqs - num_reqs),
        dtype=torch.int32,
        device=CUDA_DEVICE,
    )
    drafts = torch.tensor(
        [num_speculative_tokens] * num_reqs + [-1] * (padded_reqs - num_reqs),
        dtype=torch.int32,
    )
    return common, accepted, drafts


def _build_fast(
    builder: GDNAttentionMetadataBuilder,
    common,
    accepted: torch.Tensor,
    drafts: torch.Tensor,
) -> GDNAttentionMetadata:
    """Build metadata and prove the aligned pure-spec fast path was selected."""
    assert (
        builder._try_build_full_graph_align_pure_spec_metadata(common, accepted, drafts)
        is not None
    )
    return builder.build(0, common, accepted, drafts)


def _assert_metadata_matches(
    actual: GDNAttentionMetadata, expected: GDNAttentionMetadata
) -> None:
    """Compare every GDN metadata field, including persistent padded tails."""
    for field in fields(GDNAttentionMetadata):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(actual_value, torch.Tensor):
            assert isinstance(expected_value, torch.Tensor), field.name
            torch.testing.assert_close(actual_value, expected_value, msg=field.name)
        elif isinstance(actual_value, dict):
            assert actual_value.keys() == expected_value.keys(), field.name
            for key, value in actual_value.items():
                expected_item = expected_value[key]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        value, expected_item, msg=f"{field.name}[{key}]"
                    )
                else:
                    assert value == expected_item
        else:
            assert actual_value == expected_value, field.name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("num_reqs", "padded_reqs", "num_speculative_tokens", "padded_tokens"),
    [
        pytest.param(1, 1, 1, 2, id="one-request-mtp1"),
        pytest.param(21, 32, 1, 64, id="trace-shape-mtp1"),
        pytest.param(21, 32, 2, 64, id="trace-shape-mtp2"),
        pytest.param(32, 32, 3, 128, id="full-batch-mtp3"),
    ],
)
@torch.inference_mode()
def test_full_graph_align_pure_spec_fast_metadata_matches_legacy(
    monkeypatch: pytest.MonkeyPatch,
    num_reqs: int,
    padded_reqs: int,
    num_speculative_tokens: int,
    padded_tokens: int,
) -> None:
    """Fused aligned MTP metadata matches legacy metadata at graph shapes."""
    common, accepted, drafts = _make_padded_pure_spec_common(
        num_reqs,
        padded_reqs,
        num_speculative_tokens,
        padded_tokens,
        optimistic_cpu_seq_lens=True,
    )
    fast_builder = _create_full_graph_align_builder(num_speculative_tokens)
    legacy_builder = _create_full_graph_align_builder(num_speculative_tokens)

    actual = _build_fast(fast_builder, common, accepted, drafts)
    monkeypatch.setattr(
        legacy_builder,
        "_try_build_full_graph_align_pure_spec_metadata",
        lambda *_args, **_kwargs: None,
    )
    expected = legacy_builder.build(0, common, accepted, drafts)

    _assert_metadata_matches(actual, expected)
    assert actual.spec_state_indices_tensor is not None
    assert (
        actual.spec_state_indices_tensor[0, 0].item()
        == common.block_table_tensor[0, (2112 - 1) // BLOCK_SIZE].item()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_full_graph_align_fast_metadata_clears_dirty_padded_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A smaller replay batch cannot retain GDN state from an earlier batch."""
    builder = _create_full_graph_align_builder(num_speculative_tokens=1)
    legacy_builder = _create_full_graph_align_builder(num_speculative_tokens=1)
    monkeypatch.setattr(
        legacy_builder,
        "_try_build_full_graph_align_pure_spec_metadata",
        lambda *_args, **_kwargs: None,
    )

    large = _make_padded_pure_spec_common(32, 32, 1, 64, seq_len_offset=8)
    _build_fast(builder, *large)
    common, accepted, drafts = _make_padded_pure_spec_common(21, 32, 1, 64)
    actual = _build_fast(builder, common, accepted, drafts)
    expected = legacy_builder.build(0, common, accepted, drafts)

    _assert_metadata_matches(actual, expected)
    assert actual.spec_state_indices_tensor is not None
    assert torch.all(actual.spec_state_indices_tensor[21:] == NULL_BLOCK_ID)
    assert actual.spec_sequence_masks is not None
    assert not actual.spec_sequence_masks[21:].any()
    assert actual.spec_query_start_loc is not None
    assert torch.all(actual.spec_query_start_loc[22:] == 42)
    assert actual.num_accepted_tokens is not None
    assert torch.all(actual.num_accepted_tokens[21:] == 1)

    permutation = torch.arange(20, -1, -1, device=CUDA_DEVICE)
    reordered_block_table = common.block_table_tensor.clone()
    reordered_block_table[:21] = common.block_table_tensor[permutation]
    reordered_seq_lens = common.seq_lens.clone()
    reordered_seq_lens[:21] = common.seq_lens[permutation]
    reordered = common.replace(
        block_table_tensor=reordered_block_table,
        seq_lens=reordered_seq_lens,
    )
    actual = _build_fast(builder, reordered, accepted, drafts)
    expected = legacy_builder.build(0, reordered, accepted, drafts)
    _assert_metadata_matches(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_full_graph_align_fast_metadata_keeps_cache_groups_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three GDN cache groups share request layout, never state indices."""
    common, accepted, drafts = _make_padded_pure_spec_common(21, 32, 1, 64)
    group_metadata: list[GDNAttentionMetadata] = []
    for group in range(3):
        group_common = common.replace(
            block_table_tensor=common.block_table_tensor + (group + 1) * 50_000
        )
        fast_builder = _create_full_graph_align_builder(num_speculative_tokens=1)
        legacy_builder = _create_full_graph_align_builder(num_speculative_tokens=1)
        monkeypatch.setattr(
            legacy_builder,
            "_try_build_full_graph_align_pure_spec_metadata",
            lambda *_args, **_kwargs: None,
        )
        actual = _build_fast(fast_builder, group_common, accepted, drafts)
        expected = legacy_builder.build(0, group_common, accepted, drafts)
        _assert_metadata_matches(actual, expected)
        group_metadata.append(actual)

    first_states = []
    for metadata in group_metadata:
        assert metadata.spec_state_indices_tensor is not None
        first_states.append(metadata.spec_state_indices_tensor[0, 0])
    assert len({int(state.item()) for state in first_states}) == 3
    first_query_start = group_metadata[0].spec_query_start_loc
    assert first_query_start is not None
    for metadata in group_metadata[1:]:
        assert metadata.spec_query_start_loc is not None
        torch.testing.assert_close(metadata.spec_query_start_loc, first_query_start)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_full_graph_align_fast_metadata_capture_replays_changed_inputs() -> None:
    """The captured metadata builder reads updated aligned state inputs."""
    common, accepted, drafts = _make_padded_pure_spec_common(21, 32, 1, 64)
    builder = _create_full_graph_align_builder(num_speculative_tokens=1)
    warmup = _build_fast(builder, common, accepted, drafts)
    assert warmup.spec_state_indices_tensor is not None
    state_pointer = warmup.spec_state_indices_tensor.data_ptr()
    torch.accelerator.synchronize()

    observed = torch.empty((32, 2), dtype=torch.int32, device=CUDA_DEVICE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = builder.build(0, common, accepted, drafts)
        assert metadata.spec_state_indices_tensor is not None
        observed.copy_(metadata.spec_state_indices_tensor)

    common.block_table_tensor.add_(10_000)
    accepted.add_(1)
    graph.replay()
    torch.accelerator.synchronize()
    assert builder.spec_state_indices_tensor.data_ptr() == state_pointer

    expected_builder = _create_full_graph_align_builder(num_speculative_tokens=1)
    expected = expected_builder.build(0, common, accepted, drafts)
    assert expected.spec_state_indices_tensor is not None
    torch.testing.assert_close(observed, expected.spec_state_indices_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_full_graph_align_fast_metadata_falls_back_for_non_mtp_rows() -> None:
    """Zero-draft, mixed, and irregular rows retain the general builder."""
    builder = _create_full_graph_align_builder(num_speculative_tokens=1)
    common, accepted, _ = _make_padded_pure_spec_common(1, 1, 1, 2)
    regular = common.replace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=CUDA_DEVICE),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        num_actual_tokens=1,
    )
    assert (
        builder._try_build_full_graph_align_pure_spec_metadata(
            regular, accepted, torch.tensor([0], dtype=torch.int32)
        )
        is None
    )
    metadata = builder.build(
        0,
        regular,
        accepted,
        torch.tensor([0], dtype=torch.int32),
    )
    assert metadata.num_spec_decodes == 0
    assert metadata.spec_state_indices_tensor is None
    assert metadata.non_spec_state_indices_tensor is not None

    mixed, mixed_accepted, _ = _make_padded_pure_spec_common(2, 2, 1, 4)
    # Production mixed scheduling marks ordinary rows with -1. The spec row has
    # one target token plus one draft token, so its query length is two. Since
    # spec rows are not a leading prefix, it must retain the general builder.
    mixed = mixed.replace(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32, device=CUDA_DEVICE),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        num_actual_tokens=3,
    )
    mixed_drafts = torch.tensor([-1, 1], dtype=torch.int32)
    assert (
        builder._try_build_full_graph_align_pure_spec_metadata(
            mixed, mixed_accepted, mixed_drafts
        )
        is None
    )
    mixed_metadata = builder.build(
        0,
        mixed,
        mixed_accepted,
        mixed_drafts,
    )
    assert mixed_metadata.num_spec_decodes == 1
    assert mixed_metadata.num_prefills == 1

    builder.vllm_config.cache_config.mamba_cache_mode = "none"
    assert (
        builder._try_build_full_graph_align_pure_spec_metadata(
            common, accepted, torch.tensor([1], dtype=torch.int32)
        )
        is None
    )
