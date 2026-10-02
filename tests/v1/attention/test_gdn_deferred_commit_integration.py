# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN spec-decode deferred commit: the builder's D decision and request rows
(plain and fused decode-metadata paths), the pending set / materialize trigger,
header invalidation, and a multi-step model path (builder -> layer core) whose
outputs and live states must equal the snapshot kernel's bit for bit.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(80)):
    pytest.skip("CUDA compute capability 8.0+ required", allow_module_level=True)

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.config.compilation import CUDAGraphMode  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import (  # noqa: E402
    gdn_deferred_commit,
    qwen_gdn_linear_attn,
)
from vllm.model_executor.layers.mamba.gdn.gdn_deferred_commit import (  # noqa: E402
    GdnDeferredCommit,
    GdnDeferredStep,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops import gdn_mtp_cuda  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: E402
    rmsnorm_fn,
)
from vllm.utils.torch_utils import _encode_layer_name  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
    GDNDecodeMetadataFusion,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

DEVICE = torch.device("cuda")
NUM_SPEC = 4
W = NUM_SPEC + 1
H = 2
HV = 8
K = 128
V = 128
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
PREFIX = "model.layers.0.linear_attn"
EPS = 1e-6
MAX_REQS = 16
MIN_REQS = 4


def _vllm_config(mamba_cache_mode: str = "none", full_cuda_graph: bool = False):
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=16,
        hf_config_override={"linear_key_head_dim": K},
    )
    config.additional_config = {"gdn_prefill_backend": "cutedsl"}
    config.cache_config.mamba_cache_mode = mamba_cache_mode
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    if full_cuda_graph:
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    return config


def _builder(vllm_config, block_size: int = 16) -> GDNAttentionMetadataBuilder:
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=block_size,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _manager(layers, num_accepted=None, align=False) -> GdnDeferredCommit:
    if num_accepted is None:
        num_accepted = torch.ones(MAX_REQS, dtype=torch.int32, device=DEVICE)
    manager = GdnDeferredCommit(layers, MAX_REQS, num_accepted, align, MIN_REQS, DEVICE)
    manager.bind_num_computed(torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE))
    return manager


def _fake_layer():
    return types.SimpleNamespace(
        num_v_heads=HV, tp_size=1, kv_cache=(None, torch.empty(1, device=DEVICE))
    )


def _batch(query_lens: list[int], seq_lens: list[int] | None = None):
    if seq_lens is None:
        seq_lens = [q + 100 if q else 0 for q in query_lens]
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(batch, 16, DEVICE, arange_block_indices=True)
    common.block_table_tensor.add_(1)
    return common


def _draft(query_lens: list[int], spec: list[bool] | None = None) -> torch.Tensor:
    if spec is None:
        spec = [q == W for q in query_lens]
    return torch.tensor(
        [q - 1 if s else -1 for q, s in zip(query_lens, spec)], dtype=torch.int32
    )


def _step(num_rows: int, seed: int = 0) -> GdnDeferredStep:
    rng = np.random.default_rng(seed)
    idx_np = rng.permutation(MAX_REQS)[:num_rows]
    return GdnDeferredStep(torch.from_numpy(idx_np).to(DEVICE), idx_np)


def _build(builder, common, draft, step, **kwargs):
    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.ones(
            common.num_reqs, dtype=torch.int32, device=DEVICE
        ),
        num_decode_draft_tokens_cpu=draft,
        deferred_step=step,
        **kwargs,
    )


@pytest.fixture
def ready():
    with patch.object(gdn_mtp_cuda, "ready", return_value=True):
        yield


@pytest.fixture
def materialize():
    with patch.object(gdn_mtp_cuda, "gdn_mtp_materialize") as mock:
        yield mock


# --------------------------------------------------------------------------
# Builder: decision, request rows, pending / materialize.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_spec", [MIN_REQS - 1, MIN_REQS, MIN_REQS + 3])
def test_eager_decision_min_reqs(num_spec, ready, materialize):
    builder = _builder(_vllm_config())
    manager = _manager({PREFIX: _fake_layer()})
    manager.bind_builders([builder])
    query_lens = [W] * num_spec
    step = _step(num_spec)
    meta = _build(builder, _batch(query_lens), _draft(query_lens), step)
    assert meta.spec_deferred == (num_spec >= MIN_REQS)
    if meta.spec_deferred:
        assert meta.spec_req_rows is not None
        assert meta.spec_req_rows.dtype == torch.int32
        assert meta.spec_req_rows.tolist() == step.idx_mapping_np.tolist()
    else:
        assert meta.spec_req_rows is None


def test_decision_needs_ready_kernel(materialize):
    builder = _builder(_vllm_config())
    _manager({PREFIX: _fake_layer()}).bind_builders([builder])
    query_lens = [W] * MIN_REQS
    with patch.object(gdn_mtp_cuda, "ready", return_value=False):
        meta = _build(builder, _batch(query_lens), _draft(query_lens), _step(4))
    assert not meta.spec_deferred


@pytest.mark.parametrize(
    "num_real,num_padded,expected",
    [
        pytest.param(MIN_REQS - 1, MIN_REQS, True, id="padding-reaches-min"),
        pytest.param(MIN_REQS - 2, MIN_REQS - 1, False, id="padded-below-min"),
        pytest.param(MIN_REQS + 1, MIN_REQS + 3, True, id="above-min"),
    ],
)
def test_full_graph_decision_counts_padded_rows(
    num_real, num_padded, expected, ready, materialize
):
    """FULL decode buffers: n_eff is the padded batch (capture and replay of
    one graph agree); spec_req_rows is the persistent buffer, -1 past the real
    spec rows.
    """
    builder = _builder(_vllm_config(full_cuda_graph=True))
    _manager({PREFIX: _fake_layer()}).bind_builders([builder])
    assert builder.spec_req_rows is not None
    builder.spec_req_rows.fill_(123)  # stale content of an earlier step
    query_lens = [W] * num_real + [0] * (num_padded - num_real)
    step = _step(num_real, seed=3)
    meta = _build(builder, _batch(query_lens), _draft(query_lens), step)
    assert builder._uses_spec_decode_buffers(
        builder._split_batch(_batch(query_lens), _draft(query_lens))
    )
    assert meta.spec_deferred == expected
    if expected:
        assert meta.spec_req_rows is not None
        assert meta.spec_req_rows.data_ptr() == builder.spec_req_rows.data_ptr()
        rows = meta.spec_req_rows.tolist()
        assert rows == step.idx_mapping_np.tolist() + [-1] * (num_padded - num_real)


@pytest.mark.parametrize(
    "query_lens,spec,expected",
    [
        pytest.param([W] * 4 + [37], None, True, id="spec-first-prefill"),
        pytest.param([37] + [W] * 4, None, True, id="prefill-first"),
        pytest.param([W, W, 37, W, W], None, False, id="interleaved"),
        pytest.param([W] * 4 + [1], [True] * 4 + [False], True, id="decode-reclass"),
    ],
)
def test_mixed_decision_needs_contiguous_spec_block(
    query_lens, spec, expected, ready, materialize
):
    builder = _builder(_vllm_config())
    _manager({PREFIX: _fake_layer()}).bind_builders([builder])
    step = _step(len(query_lens), seed=5)
    draft = _draft(query_lens, spec)
    meta = _build(builder, _batch(query_lens), draft, step)
    assert meta.num_prefills > 0
    assert meta.spec_deferred == expected
    if expected:
        spec_rows = np.flatnonzero(draft.numpy() >= 0)
        assert meta.spec_req_rows is not None
        assert meta.spec_req_rows.tolist() == step.idx_mapping_np[spec_rows].tolist()


def test_non_spec_decode_batch_never_defers(ready, materialize):
    builder = _builder(_vllm_config())
    _manager({PREFIX: _fake_layer()}).bind_builders([builder])
    query_lens = [1] * 8
    meta = _build(builder, _batch(query_lens), torch.full((8,), -1), _step(8))
    assert meta.num_decodes == 8 and not meta.spec_deferred


def test_capture_build_keeps_snapshot_rows(ready, materialize):
    """Capture: the flag follows n_eff (the replayed graph runs the deferred
    kernel), the persistent rows are reset to -1 (snapshot mode for the
    capture/dummy replays), and no pending state is marked or materialized.
    """
    builder = _builder(_vllm_config(full_cuda_graph=True))
    manager = _manager({PREFIX: _fake_layer()})
    manager.bind_builders([builder])
    group = builder.deferred_commit
    assert group is not None
    group.mark_pending(np.array([3, 4]))
    assert builder.spec_req_rows is not None
    builder.spec_req_rows.fill_(9)
    for num_reqs, expected in ((MIN_REQS - 1, False), (MIN_REQS + 2, True)):
        meta = builder.build_for_cudagraph_capture(_batch([W] * num_reqs))
        assert meta.spec_deferred == expected
        if expected:
            assert meta.spec_req_rows is not None
            assert meta.spec_req_rows.tolist() == [-1] * num_reqs
    materialize.assert_not_called()
    assert group.has_pending and group.pending[[3, 4]].all()


def test_pending_rows_materialize_on_first_non_deferred_step(ready, materialize):
    layers = {PREFIX: _fake_layer()}
    num_accepted = torch.arange(MAX_REQS, dtype=torch.int32, device=DEVICE)
    manager = _manager(layers, num_accepted)
    builder = _builder(_vllm_config())
    manager.bind_builders([builder])
    group = builder.deferred_commit
    assert group is not None

    marked: set[int] = set()
    for seed in range(2):
        step = _step(MIN_REQS, seed=seed)
        meta = _build(builder, _batch([W] * MIN_REQS), _draft([W] * MIN_REQS), step)
        assert meta.spec_deferred
        marked |= set(step.idx_mapping_np.tolist())
    materialize.assert_not_called()
    assert set(np.flatnonzero(group.pending).tolist()) == marked

    # D -> non-D: the batch drops below MIN_REQS.
    small = [W] * (MIN_REQS - 1)
    meta = _build(builder, _batch(small), _draft(small), _step(MIN_REQS - 1, seed=9))
    assert not meta.spec_deferred
    materialize.assert_called_once()
    state, hdr, rows, accepted = materialize.call_args.args
    layer = layers[PREFIX]
    assert state is layer.kv_cache[1]
    assert hdr.data_ptr() == manager.hdr[0].data_ptr() and hdr.shape == (
        MAX_REQS,
        HV,
        gdn_mtp_cuda.GDN_DEFERRED_HDR_INTS,
    )
    assert rows.dtype == torch.int32 and rows.device.type == "cuda"
    assert rows.tolist() == sorted(marked)
    assert accepted is num_accepted
    assert not group.has_pending and not group.pending.any()

    # Nothing pending: the next non-D steps launch nothing.
    _build(builder, _batch(small), _draft(small), _step(MIN_REQS - 1, seed=9))
    _build(builder, _batch([1, 1]), torch.tensor([-1, -1]), _step(2))
    assert materialize.call_count == 1


def test_deferred_step_materializes_pending_non_spec_rows(ready, materialize):
    """A deferred mixed step whose non-spec rows (a reclassified 1-token row,
    a prefill) include pending request states materializes exactly those
    before its forward and leaves the other pending states pending.
    """
    layers = {PREFIX: _fake_layer()}
    manager = _manager(layers)
    builder = _builder(_vllm_config())
    manager.bind_builders([builder])
    group = builder.deferred_commit
    idx_np = np.array([3, 7, 1, 9, 4, 11], dtype=np.int64)
    step = GdnDeferredStep(torch.from_numpy(idx_np).to(DEVICE), idx_np)
    query_lens = [W] * 6
    assert _build(builder, _batch(query_lens), _draft(query_lens), step).spec_deferred
    materialize.assert_not_called()

    # Request states 1 and 9 are now non-spec rows; 12 is a new prefill.
    idx_np = np.array([3, 7, 4, 11, 1, 9, 12], dtype=np.int64)
    step = GdnDeferredStep(torch.from_numpy(idx_np).to(DEVICE), idx_np)
    query_lens = [W] * 4 + [1, 1, 37]
    meta = _build(
        builder, _batch(query_lens), _draft(query_lens, [True] * 4 + [False] * 3), step
    )
    assert meta.spec_deferred and meta.num_prefills == 3
    materialize.assert_called_once()
    assert materialize.call_args.args[2].tolist() == [1, 9]
    assert sorted(np.flatnonzero(group.pending).tolist()) == [3, 4, 7, 11]

    # All non-spec rows already materialized: no launch.
    _build(
        builder, _batch(query_lens), _draft(query_lens, [True] * 4 + [False] * 3), step
    )
    assert materialize.call_count == 1


def test_fused_decode_metadata_writes_req_rows(ready, materialize):
    """The fused decode-metadata kernel writes the request rows of a deferred
    step into every builder's persistent buffer, as build() does, and the
    metadata of both paths agree.
    """
    config = _vllm_config(full_cuda_graph=True)
    fused_builders = [_builder(config), _builder(config)]
    plain = _builder(config)
    manager = _manager({PREFIX: _fake_layer()})
    manager.bind_builders(fused_builders + [plain])
    fusion = GDNDecodeMetadataFusion.create(list(enumerate(fused_builders)), DEVICE)
    assert fusion is not None and fusion.writes_req_rows
    assert int(fusion.out_ptrs[0, 7]) == fused_builders[0].spec_req_rows.data_ptr()

    num_real, num_padded = MIN_REQS + 1, MIN_REQS + 3
    query_lens = [W] * num_real + [0] * (num_padded - num_real)
    step = _step(num_real, seed=11)
    num_accepted_by_req = torch.randint(
        1, W + 1, (MAX_REQS,), dtype=torch.int32, device=DEVICE
    )
    for builder in fused_builders:
        builder.spec_req_rows.fill_(77)
    common = _batch(query_lens)
    fused_step = fusion.begin_step(
        [common.block_table_tensor] * 2,
        step.idx_mapping,
        num_accepted_by_req,
        num_real,
        None,
    )
    expected_rows = step.idx_mapping_np.tolist() + [-1] * (num_padded - num_real)
    metas = []
    for builder in fused_builders:
        meta = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_decode_draft_tokens_cpu=_draft(query_lens),
            fused_decode=fused_step,
            deferred_step=step,
        )
        assert meta.spec_deferred
        assert meta.spec_req_rows is not None
        assert meta.spec_req_rows.data_ptr() == builder.spec_req_rows.data_ptr()
        assert meta.spec_req_rows.tolist() == expected_rows
        metas.append(meta)
    plain_meta = plain.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=num_accepted_by_req[step.idx_mapping][:num_real].clone(),
        num_decode_draft_tokens_cpu=_draft(query_lens),
        deferred_step=step,
    )
    assert plain_meta.spec_deferred
    assert plain_meta.spec_req_rows.tolist() == expected_rows
    for builder in fused_builders + [plain]:
        group = builder.deferred_commit
        assert set(np.flatnonzero(group.pending).tolist()) == set(
            step.idx_mapping_np.tolist()
        )

    # Below MIN_REQS: no rows written, and the pending rows of every group are
    # materialized by its own build.
    small = [W] * 2
    common = _batch(small)
    small_step = _step(2, seed=12)
    fused_step = fusion.begin_step(
        [common.block_table_tensor] * 2,
        small_step.idx_mapping,
        num_accepted_by_req,
        2,
        None,
    )
    for builder in fused_builders:
        meta = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_decode_draft_tokens_cpu=_draft(small),
            fused_decode=fused_step,
            deferred_step=small_step,
        )
        assert not meta.spec_deferred and meta.spec_req_rows is None
        assert builder.spec_req_rows[:num_padded].tolist() == expected_rows
    assert materialize.call_count == 2


def test_deferred_off_is_a_no_op(materialize):
    """Unbound builders: no buffers, no flag, no rows, an all-zero 8th pointer
    column; the manager is not created with the env unset.
    """
    with patch.object(gdn_deferred_commit, "GDN_DEFERRED_COMMIT", False):
        assert (
            GdnDeferredCommit.maybe_create(
                _vllm_config(), MAX_REQS, torch.ones(MAX_REQS), DEVICE
            )
            is None
        )
    config = _vllm_config(full_cuda_graph=True)
    builder = _builder(config)
    assert builder.deferred_commit is None and builder.spec_req_rows is None
    fusion = GDNDecodeMetadataFusion.create([(0, builder)], DEVICE)
    assert fusion is not None and not fusion.writes_req_rows
    assert int(fusion.out_ptrs[0, 7]) == 0
    query_lens = [W] * (MIN_REQS + 2)
    with patch.object(gdn_mtp_cuda, "ready", return_value=True):
        meta = _build(builder, _batch(query_lens), _draft(query_lens), None)
        common = _batch(query_lens)
        fused = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_decode_draft_tokens_cpu=_draft(query_lens),
            fused_decode=fusion.begin_step(
                [common.block_table_tensor],
                torch.arange(len(query_lens), device=DEVICE),
                torch.ones(MAX_REQS, dtype=torch.int32, device=DEVICE),
                len(query_lens),
                None,
            ),
        )
    for m in (meta, fused):
        assert not m.spec_deferred and m.spec_req_rows is None
    materialize.assert_not_called()


def test_bind_is_all_or_nothing():
    manager = _manager({PREFIX: _fake_layer()})
    config = _vllm_config()
    other = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=16,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=["model.layers.1.linear_attn"],
        vllm_config=config,
        device=DEVICE,
    )
    builder = _builder(config)
    manager.bind_builders([builder, other])
    assert builder.deferred_commit is None and other.deferred_commit is None

    # Align: the window is the builders' Mamba block size (not the attention
    # block size); builders that disagree on it disable deferral.
    manager = _manager({PREFIX: _fake_layer()}, align=True)
    builder = _builder(_vllm_config("align"), block_size=64)
    manager.bind_builders([builder])
    assert builder.deferred_commit is not None and manager.block_size == 64
    manager = _manager({PREFIX: _fake_layer(), "x": _fake_layer()}, align=True)
    b1 = _builder(_vllm_config("align"), block_size=64)
    b2 = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=32,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=["x"],
        vllm_config=_vllm_config("align"),
        device=DEVICE,
    )
    manager.bind_builders([b1, b2])
    assert b1.deferred_commit is None and b2.deferred_commit is None
    assert manager.block_size == 0


def test_invalidation_flush():
    manager = _manager({PREFIX: _fake_layer(), "x": _fake_layer()})
    builder = _builder(_vllm_config())
    manager.bind_builders([builder])
    group = builder.deferred_commit
    group.mark_pending(np.array([1, 2, 5]))
    manager.hdr.fill_(7)
    before = manager.hdr.clone()
    manager.flush_invalidations()  # nothing queued
    assert torch.equal(manager.hdr, before)
    for row in (2, 9, 2):
        manager.invalidate(row)
    assert group.pending[[1, 5]].all() and not group.pending[[2, 9]].any()
    assert torch.equal(manager.hdr, before)  # deferred to the flush
    manager.flush_invalidations()
    expected = before.clone()
    expected[:, [2, 9]] = 0
    assert torch.equal(manager.hdr, expected)
    assert not manager._invalidate_rows


def test_model_state_prepare_attn(ready, materialize, monkeypatch):
    """MambaHybridModelState: add/remove invalidate headers in one flush before
    the builds; the step's request rows reach every GDN builder through both
    the plain and the fused decode-metadata paths, which agree on the flag,
    the rows and the materialize launches.
    """
    from tests.v1.worker import test_gdn_decode_metadata_fusion as fusion_test
    from vllm.v1.worker.gpu.model_states import mamba_hybrid

    def make_state():
        state, groups = fusion_test._make_state(3, "none", seed=0)
        layers = {
            name: _fake_layer() for group in groups[1:] for name in group[0].layer_names
        }
        state.gdn_deferred_commit = _manager(layers)
        state._gdn_dc_req_index = {}
        state.rope_state = None
        state.prompt_embeds_state = None
        return state, groups

    states = {"fused": make_state(), "plain": make_state()}
    for state, _ in states.values():
        for i in range(4):
            state.add_request(
                i, types.SimpleNamespace(req_id=f"r{i}", num_computed_tokens=9)
            )
        state.gdn_deferred_commit.hdr.fill_(5)
        state.remove_request("r1")
        state.remove_request("unknown")
        state.add_request(7, types.SimpleNamespace(req_id="r7", num_computed_tokens=0))

    rng = np.random.default_rng(0)
    seen_deferred = seen_other = materialize_steps = 0
    for step_idx, step in enumerate(fusion_test.SPEC_STEPS):
        input_batch, block_tables = fusion_test._make_inputs(step, rng, DEVICE)
        input_batch.idx_mapping_np = input_batch.idx_mapping.cpu().numpy()
        metas = {}
        calls = {}
        for name, (state, groups) in states.items():
            materialize.reset_mock()
            with monkeypatch.context() as m:
                if name == "plain":
                    m.setattr(
                        mamba_hybrid.GDNDecodeMetadataFusion, "create", lambda *_: None
                    )
                metas[name] = fusion_test._prepare_attn(
                    state, groups, step, input_batch, block_tables
                )
            calls[name] = [c.args[2].tolist() for c in materialize.call_args_list]
            hdr = state.gdn_deferred_commit.hdr
            if step_idx == 0:
                # Rows 0..3 and 7 were added, 1 removed: all zeroed, rest kept.
                assert not hdr[:, [0, 1, 2, 3, 7]].any()
                assert bool((hdr[:, [4, 5, 6]] == 5).all())
        assert calls["fused"] == calls["plain"], step_idx
        materialize_steps += bool(calls["fused"])
        spec = np.array([r[1] > 0 and not r[3] for r in step.reqs])
        for group in states["fused"][1][1:]:
            layer = group[0].layer_names[0]
            fused, plain = metas["fused"][layer], metas["plain"][layer]
            assert fused.spec_deferred == plain.spec_deferred, step_idx
            if not fused.spec_deferred:
                assert fused.spec_req_rows is None and plain.spec_req_rows is None
                continue
            rows = input_batch.idx_mapping_np[np.flatnonzero(spec)].tolist()
            for meta in (fused, plain):
                got = meta.spec_req_rows.tolist()
                assert got[: len(rows)] == rows
                assert got[len(rows) :] == [-1] * (len(got) - len(rows))
        deferred = metas["fused"][states["fused"][1][1][0].layer_names[0]]
        if deferred.spec_deferred:
            seen_deferred += 1
        else:
            seen_other += 1
    assert seen_deferred and seen_other and materialize_steps


# --------------------------------------------------------------------------
# Model path: several steps through builder -> layer core, deferred on vs off.
# --------------------------------------------------------------------------


class _GatedNorm:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.bias = None
        self.eps = EPS
        self.group_size = None
        self.norm_before_gate = True
        self.activation = "silu"

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fn(
            x, self.weight, None, z=z, eps=EPS, norm_before_gate=True, activation="silu"
        )


def _layer(vllm_config, conv_state, ssm_state, params):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        disable_tp_for_ba_proj=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=H * K,
        value_dim=HV * V,
        activation="silu",
        A_log=params["a_log"],
        dt_bias=params["dt_bias"],
        conv1d=types.SimpleNamespace(weight=params["conv_weight"], bias=None),
        kv_cache=(conv_state, ssm_state),
        norm=_GatedNorm(params["norm_weight"]),
        layer_norm_epsilon=EPS,
        gdn_decode_kernel="cuda",
        _fused_decode_counters=None,
        _gdn_dc_hdr=None,
        _gdn_dc=None,
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_spec_post_conv_fused_norm",
        "_forward_core_decode_spec_fused_norm",
        "_can_use_fused_gdn_mtp_decode",
        "_rms_norm_gated_cuda",
        "_rms_norm_gated_strided_gate_cuda",
        "_forward_core_fused_norm",
        "_forward_core_fused_norm_packed",
        "_forward_core_decode_spec_one_launch",
        "_gated_norm_mxfp8",
        "split_ba",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


# Per step: active request states; whether one new prefill request joins
# (mixed batch, spec rows first: a deferred mixed step); the active states that
# run as 1-token non-spec rows (no drafts: reclassified as prefills, after the
# spec rows, still a deferred step), with 1 accepted token the step before (the
# prefill path reads column 0, as with the snapshot kernel). 3 rows: deferred
# only when FULL padding lifts the batch to MIN_REQS; 2 rows: never deferred.
_ALL = [0, 1, 2, 3, 4, 5]
_SCHEDULE: list[tuple[list[int], bool, list[int]]] = [
    (_ALL, False, []),
    (_ALL, False, []),
    (_ALL, False, [2]),
    (_ALL, False, []),
    (_ALL, True, []),
    ([0, 2, 4], False, []),
    (_ALL, False, []),
    (_ALL, True, [0, 5]),
    (_ALL, False, []),
    ([1, 5], False, []),
    (_ALL, False, []),
    (_ALL, False, []),
    ([3, 4], False, []),
]
PREFILL_REQ = 12
PREFILL_LEN = 37


@pytest.mark.parametrize(
    "mode,full_graph,state_dtype",
    [
        pytest.param("none", False, torch.float32, id="none-eager"),
        pytest.param("none", True, torch.float32, id="none-full"),
        pytest.param("none", False, torch.bfloat16, id="none-eager-bf16"),
        pytest.param("align", False, torch.float32, id="align-eager"),
        pytest.param("align", True, torch.float32, id="align-full"),
    ],
)
@torch.inference_mode()
def test_model_path_deferred_matches_snapshot(mode, full_graph, state_dtype):
    """Multi-step spec decode through the GDN builder and layer core with the
    deferred commit on vs off (the snapshot JIT CUDA kernel), accepted counts
    cycling through 1..W, D <-> non-D transitions (the batch drops below
    MIN_REQS; FULL padding lifting a 3-row batch to MIN_REQS), mixed D steps
    (a prefill; pending request states as 1-token non-spec rows, then spec rows
    again) and, in align mode, block-boundary crossings. Outputs, conv states
    and every request's live state (its next source slot, after
    materialization) must match bit for bit.

    Align mode is simulated explicitly (not through MambaHybridModelState): the
    running block column R = (P + T - 1) // B per step, the pre-copy of
    bt[R_old + n - 1] -> bt[R] with num_accepted reset to 1 when R moves, and
    the post-step checkpoint copy bt[R + bias] -> bt[aligned // B - 1] (with
    the reset when it lands on R), the reads the danger window protects.
    """
    if not gdn_mtp_cuda.enable() or not hasattr(gdn_mtp_cuda, "gdn_mtp_cuda_deferred"):
        pytest.skip("GDN MTP CUDA kernel (deferred mode) did not build")
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        pytest.skip("fused GDN decode MTP op is not built")
    torch.manual_seed(7)
    align = mode == "align"
    mamba_block = 32 if align else 16
    vllm_config = _vllm_config(mode, full_cuda_graph=full_graph)
    # Columns per request: the window, plus the blocks the align run crosses.
    num_cols = 12 if align else W
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
    )
    num_slots = 1 + MAX_REQS * num_cols
    block_table = (
        1 + torch.arange(MAX_REQS * num_cols, dtype=torch.int32).view(MAX_REQS, -1)
    ).to(DEVICE)
    conv_seed = 0.05 * torch.randn(
        num_slots, *conv_shape, dtype=torch.bfloat16, device=DEVICE
    )
    ssm_seed = (
        0.05 * torch.randn(num_slots, *ssm_shape, dtype=torch.float32, device=DEVICE)
    ).to(state_dtype)
    params = {
        "a_log": 0.1 * torch.randn(HV, dtype=torch.float32, device=DEVICE),
        "dt_bias": 0.1 * torch.randn(HV, dtype=torch.float32, device=DEVICE),
        "conv_weight": 0.1
        * torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=DEVICE),
        "norm_weight": torch.randn(V, dtype=torch.float32, device=DEVICE),
    }

    class _Run:
        def __init__(self, deferred: bool):
            self.layer = _layer(
                vllm_config, conv_seed.clone(), ssm_seed.clone(), params
            )
            self.builder = _builder(vllm_config, mamba_block)
            self.num_accepted = torch.ones(MAX_REQS, dtype=torch.int32, device=DEVICE)
            self.manager = None
            if deferred:
                self.manager = _manager(
                    {PREFIX: self.layer},
                    self.num_accepted,
                    align=align,
                )
                self.manager.bind_builders([self.builder])
                assert self.builder.deferred_commit is not None

    on, off = _Run(True), _Run(False)
    assert on.manager is not None and on.manager.num_computed is not None
    num_computed = on.manager.num_computed
    # Per request state: tokens before the step, running column, accepted.
    start = [5, 20, 27, 40, 58, 63] if align else [100, 7, 64, 33, 2, 90]
    P = {rho: start[rho] for rho in _ALL}
    R = {rho: (P[rho] - 1) // mamba_block if align else 0 for rho in _ALL}
    n = {rho: 1 for rho in _ALL}
    deferred_steps = 0
    nondeferred_after_deferred = 0
    boundary_moves = 0
    deferred_rows = 0
    snapshot_rows = 0
    decode_rows_checked = 0

    def copy_slot(src: int, dst: int) -> None:
        for run in (on, off):
            for state in run.layer.kv_cache:
                state[dst].copy_(state[src])

    def live_slot(rho: int) -> int:
        return int(block_table[rho, R[rho] + n[rho] - 1])

    for step_idx, (active, with_prefill, decode_rows) in enumerate(_SCHEDULE):
        if with_prefill and align:
            with_prefill = False
        # Batch row r -> request state rho: the spec rows (a permutation of
        # the active set), then the 1-token non-spec rows.
        rows = [rho for rho in active if rho not in decode_rows]
        rng = np.random.default_rng(step_idx)
        rng.shuffle(rows)
        query_len = {rho: W for rho in rows} | {rho: 1 for rho in decode_rows}
        # Align pre-copy (MambaHybridModelState.preprocess_state).
        if align:
            for rho in rows + decode_rows:
                r_new = (P[rho] + query_len[rho] - 1) // mamba_block
                if r_new != R[rho]:
                    copy_slot(
                        int(block_table[rho, R[rho] + n[rho] - 1]),
                        int(block_table[rho, r_new]),
                    )
                    R[rho] = r_new
                    n[rho] = 1
                    boundary_moves += 1
        assert all(n[rho] == 1 for rho in decode_rows)
        for rho in _ALL:
            on.num_accepted[rho] = n[rho]
            off.num_accepted[rho] = n[rho]
            num_computed[rho] = P[rho]
        query_lens = [query_len[rho] for rho in rows + decode_rows]
        seq_lens = [P[rho] + query_len[rho] for rho in rows + decode_rows]
        bt_rows = [block_table[rho] for rho in rows + decode_rows]
        idx = rows + decode_rows
        if with_prefill:
            query_lens.append(PREFILL_LEN)
            seq_lens.append(PREFILL_LEN)
            bt_rows.append(block_table[PREFILL_REQ])
            idx.append(PREFILL_REQ)
        mixed = with_prefill or bool(decode_rows)
        num_real = len(query_lens)
        num_padded = num_real
        if full_graph and not mixed and num_real == 3:
            num_padded = MIN_REQS
        while len(query_lens) < num_padded:
            query_lens.append(0)
            seq_lens.append(0)
            bt_rows.append(torch.zeros_like(block_table[0]))
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=seq_lens, query_lens=query_lens), 16, DEVICE
        )
        common.block_table_tensor = torch.stack(bt_rows)
        draft = torch.tensor(
            [W - 1] * len(rows) + [-1] * (num_padded - len(rows)), dtype=torch.int32
        )
        accepted_rows = torch.tensor(
            [n[rho] for rho in rows] + [1] * (num_padded - len(rows)),
            dtype=torch.int32,
            device=DEVICE,
        )
        idx_np = np.array(idx, dtype=np.int64)
        num_tokens = sum(query_lens)
        mixed_qkv = 0.1 * torch.randn(
            num_tokens, CONV_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        b = 0.5 * torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=DEVICE)
        a = 0.5 * torch.randn_like(b)
        gate = 0.1 * torch.randn(num_tokens, HV, V, dtype=torch.bfloat16, device=DEVICE)
        mixed_qkvz = torch.cat((mixed_qkv, gate.flatten(1)), dim=-1)
        ba = torch.cat((b, a), dim=-1)
        # The spec rows' mid-window slots (columns R+1..R+W-2), before the step.
        mid_slots = torch.stack(
            [block_table[rho, R[rho] + 1 : R[rho] + W - 1] for rho in rows]
        ).flatten()
        mid_before = on.layer.kv_cache[1][mid_slots].clone()

        outs = {}
        metas = {}
        for name, run in (("on", on), ("off", off)):
            step = GdnDeferredStep(torch.from_numpy(idx_np).to(DEVICE), idx_np)
            with set_current_vllm_config(vllm_config):
                metas[name] = run.builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common,
                    num_accepted_tokens=accepted_rows,
                    num_decode_draft_tokens_cpu=draft,
                    deferred_step=step,
                )
            context = types.SimpleNamespace(
                attn_metadata={PREFIX: metas[name]},
                no_compile_layers={PREFIX: run.layer},
            )
            out = torch.full_like(gate, 3.0)
            with (
                patch.object(
                    qwen_gdn_linear_attn, "get_forward_context", return_value=context
                ),
                patch.object(qwen_gdn_linear_attn, "GDN_MTP_TRITON_MAX_REQUESTS", 0),
            ):
                torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                    mixed_qkvz.clone(), ba, out, layer_name=_encode_layer_name(PREFIX)
                )
            outs[name] = out
        expect_deferred = (len(rows) if mixed else num_padded) >= MIN_REQS
        assert metas["on"].spec_deferred == expect_deferred, step_idx
        assert not metas["off"].spec_deferred
        if expect_deferred:
            deferred_steps += 1
            # Rows outside the align danger window ran deferred: valid header,
            # no mid-window snapshot writes (the snapshot run wrote them).
            valid = on.manager.hdr[0, rows, :, 0].bool()
            assert bool((valid.all(1) | ~valid.any(1)).all())
            row_deferred = valid.all(1).cpu()
            deferred_rows += int(row_deferred.sum())
            snapshot_rows += int((~row_deferred).sum())
            if not align:
                assert bool(row_deferred.all())
            mid_after = on.layer.kv_cache[1][mid_slots].view(len(rows), W - 2, -1)
            mid_ref = mid_before.view(len(rows), W - 2, -1)
            for j in range(len(rows)):
                unchanged = torch.equal(_bits(mid_after[j]), _bits(mid_ref[j]))
                assert unchanged == bool(row_deferred[j]), (step_idx, rows[j])
        torch.testing.assert_close(outs["on"], outs["off"], atol=0, rtol=0)
        assert torch.equal(on.layer.kv_cache[0], off.layer.kv_cache[0]), step_idx

        # Sampling + post-step state (post_update, align postprocess). The
        # states of the next step's non-spec rows accept 1 token here.
        next_decode = (
            _SCHEDULE[step_idx + 1][2] if step_idx + 1 < len(_SCHEDULE) else []
        )
        for j, rho in enumerate(rows + decode_rows):
            n_new = 1 + (3 * step_idx + 2 * rho + j) % W
            if rho in decode_rows or rho in next_decode:
                n_new = 1
            p_new = P[rho] + n_new
            if align:
                aligned = (p_new // mamba_block) * mamba_block
                if aligned >= P[rho] + 1:
                    bias = aligned - (P[rho] + 1)
                    dest = aligned // mamba_block - 1
                    if dest == R[rho]:
                        n_new = 1
                    if not (dest == R[rho] and bias == 0):
                        copy_slot(
                            int(block_table[rho, R[rho] + bias]),
                            int(block_table[rho, dest]),
                        )
            P[rho] = p_new
            n[rho] = n_new
        # The non-spec rows of pending request states were materialized and
        # their headers invalidated before the prefill path read them.
        for rho in decode_rows:
            slot = live_slot(rho)
            assert torch.equal(
                _bits(on.layer.kv_cache[1][slot]), _bits(off.layer.kv_cache[1][slot])
            ), (step_idx, rho)
            assert not bool(on.manager.hdr[0, rho, :, 0].any()), (step_idx, rho)
            decode_rows_checked += 1

        if not expect_deferred:
            if step_idx and _SCHEDULE[step_idx - 1][0] == _ALL:
                nondeferred_after_deferred += 1
            # Every pending request state was materialized before this step:
            # all live states agree with the snapshot run.
            hdr_valid = on.manager.hdr[0, :, :, 0]
            for rho in _ALL:
                slot = live_slot(rho)
                assert torch.equal(
                    _bits(on.layer.kv_cache[1][slot]),
                    _bits(off.layer.kv_cache[1][slot]),
                ), (step_idx, rho)
                assert not bool(hdr_valid[rho].any()), (step_idx, rho)
        if with_prefill:
            slot = int(block_table[PREFILL_REQ, 0])
            assert torch.equal(
                _bits(on.layer.kv_cache[1][slot]), _bits(off.layer.kv_cache[1][slot])
            )

    assert _SCHEDULE[-1][0] != _ALL
    assert deferred_steps >= 7
    assert nondeferred_after_deferred >= 2
    assert deferred_rows > 0 and decode_rows_checked == 3
    if align:
        # Block crossings ran, and the danger window put rows of deferred
        # steps into snapshot mode.
        assert boundary_moves > 0 and snapshot_rows > 0
