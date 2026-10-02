# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused router GEMV + routing kernel vs FlashInfer 0.6.18.post1 routing semantics.

The reference routing mirrors FlashInfer's routingCustom (NoOp preprocess + Softmax
postprocess, as built with -use_fast_math): packed-key top-K (lower expert index wins
ties), ex2.approx / div.approx softmax over the top-K with the warp butterfly sum, and
the block-kernel metadata (token order within an expert). On SM10x the kernel is also
compared with FlashInfer itself (tinygemm2_sm100 logits, production routing runner).
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe import fused_router_routing as frr
from vllm.triton_utils import tl, triton

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
    pytest.skip("SM90+ CUDA device required", allow_module_level=True)

E = frr.NUM_EXPERTS
K = frr.HIDDEN_SIZE
TILES = [8, 16, 32, 64, 128, 256]  # FlashInfer MXFP8 tile_N set
_LOG2E = torch.tensor(1.4426950216293335, dtype=torch.float32)  # 0x3FB8AA3B


@triton.jit
def _ptx_kernel(a_ptr, b_ptr, o_ptr, n, DIV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0)
    if DIV:
        o = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        o = tl.inline_asm_elementwise(
            "ex2.approx.ftz.f32 $0, $1;",
            "=r,r",
            [a],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(o_ptr + offs, o, mask=mask)


def _ptx(a: torch.Tensor, b: torch.Tensor, div: bool) -> torch.Tensor:
    a = a.contiguous()
    b = b.contiguous()
    o = torch.empty_like(a)
    n = a.numel()
    _ptx_kernel[(triton.cdiv(n, 256),)](a, b, o, n, DIV=div, BLOCK=256)
    return o


def _topk_ids(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """FlashInfer's packed (twiddled bf16 << 16 | 65535 - expert) max, descending."""
    bits = logits.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    twiddled = torch.where(bits & 0x8000 != 0, (~bits) & 0xFFFF, bits | 0x8000)
    idx = torch.arange(logits.shape[1], device=logits.device, dtype=torch.int64)
    keys = (twiddled << 16) | (65535 - idx)
    vals = torch.topk(keys, top_k, dim=1, sorted=True).values
    return (65535 - (vals & 0xFFFF)).to(torch.int32)


def _softmax_weights(scores: torch.Tensor) -> torch.Tensor:
    """calcSoftmax(warp, score, laneIdx, topK) as compiled with -use_fast_math."""
    m, top_k = scores.shape
    d = scores - scores[:, :1]
    p = _ptx(d * _LOG2E.to(scores.device), d, div=False)
    lanes = torch.zeros(m, 32, dtype=torch.float32, device=scores.device)
    lanes[:, :top_k] = p
    for mask in (16, 8, 4, 2, 1):
        partner = torch.arange(32, device=scores.device) ^ mask
        lanes = lanes + lanes[:, partner]
    total = lanes[:, :1].expand(m, top_k)
    return _ptx(p, total, div=True).to(torch.bfloat16)


def _reference(logits: torch.Tensor, top_k: int, tile_n: int) -> dict:
    m = logits.shape[0]
    ids = _topk_ids(logits, top_k)
    scores = torch.gather(logits, 1, ids.long()).float()
    weights = _softmax_weights(scores)
    ids_cpu = ids.cpu()
    counts = torch.bincount(ids_cpu.flatten().long(), minlength=E)
    n_cta = (counts + tile_n - 1) // tile_n
    cta_off = torch.cumsum(n_cta, 0) - n_cta
    total = int(n_cta.sum())
    batch, mn = [], []
    for e in range(E):
        for c in range(int(n_cta[e])):
            o = int(cta_off[e])
            batch.append(e)
            mn.append(min((o + c + 1) * tile_n, o * tile_n + int(counts[e])))
    exp_to_perm = torch.empty(m * top_k, dtype=torch.int32)
    fill = torch.zeros(E, dtype=torch.int64)
    for t in range(m):  # rows of an expert in token order
        for k in range(top_k):
            e = int(ids_cpu[t, k])
            exp_to_perm[t * top_k + k] = int(cta_off[e]) * tile_n + int(fill[e])
            fill[e] += 1
    return dict(
        ids=ids_cpu,
        weights=weights,
        counts=counts.to(torch.int32),
        total=total,
        batch=torch.tensor(batch, dtype=torch.int32),
        mn=torch.tensor(mn, dtype=torch.int32),
        exp_to_perm=exp_to_perm,
    )


def _bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    def ordered(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((ordered(a) - ordered(b)).abs().max())


def _assert_logits_close(logits: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    """fp32 accumulation + one bf16 rounding: within half a bf16 step (2^-9 relative)
    of the exact result, plus the fp32 accumulation error (absolute).
    """
    exact = x.double() @ w.double().t()
    err = (logits.double() - exact).abs()
    assert bool((err <= exact.abs() * 2**-8 + 2**-12).all())


def _inputs(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, K, device="cuda", dtype=torch.float32, generator=g)
    w = torch.randn(E, K, device="cuda", dtype=torch.float32, generator=g) * 0.02
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


def _run(x, w, top_k, tile_n, method=4, use_pdl=False, out=None, logits=True):
    m = x.shape[0]
    if out is None:
        out = frr.FusedRoutingOutputs.allocate(frr.MAX_TOKENS, top_k, E, tile_n, "cuda")
    view = out.for_tokens(m)
    lg = torch.empty(m, E, device="cuda", dtype=torch.bfloat16) if logits else None
    frr.fused_router_routing(
        x,
        w,
        view,
        top_k=top_k,
        tile_n=tile_n,
        renormalize_mode=method,
        use_pdl=use_pdl,
        logits_out=lg,
    )
    return view, lg


def _check_against_reference(view, logits, top_k, tile_n):
    m = logits.shape[0]
    ref = _reference(logits, top_k, tile_n)
    assert torch.equal(view.expert_weights.cpu(), ref["weights"].cpu())
    assert int(view.num_non_exiting_ctas[0]) == ref["total"]
    assert int(view.total_num_padded_tokens[0]) == ref["total"] * tile_n
    assert torch.equal(view.num_tokens_per_expert.cpu(), ref["counts"])
    total = ref["total"]
    assert torch.equal(view.cta_idx_xy_to_batch_idx[:total].cpu(), ref["batch"])
    assert torch.equal(view.cta_idx_xy_to_mn_limit[:total].cpu(), ref["mn"])
    e2p = view.expanded_idx_to_permuted_idx.cpu()
    assert torch.equal(e2p, ref["exp_to_perm"])
    p2t = view.permuted_idx_to_token_idx.cpu()
    tokens = torch.arange(m * top_k) // top_k
    assert torch.equal(p2t[e2p.long()], tokens.to(torch.int32))
    return ref


@pytest.mark.parametrize(
    "m", [1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 31, 32, 33, 48, 63, 64]
)
@pytest.mark.parametrize("tile_n", [8, 32, 256])
def test_matches_flashinfer_semantics(m: int, tile_n: int):
    x, w = _inputs(m, seed=m)
    out = frr.FusedRoutingOutputs.allocate(frr.MAX_TOKENS, 8, E, tile_n, "cuda")
    out.permuted_idx_to_token_idx.fill_(-7)
    out.expert_count_histogram.fill_(-9)
    view, logits = _run(x, w, 8, tile_n, out=out)
    _assert_logits_close(logits, x, w)
    _check_against_reference(view, logits, 8, tile_n)
    # Left untouched, as by FlashInfer: histogram scratch and padding rows.
    assert bool((out.expert_count_histogram == -9).all())
    used = torch.zeros(view.permuted_idx_to_token_idx.numel(), dtype=torch.bool)
    used[view.expanded_idx_to_permuted_idx.cpu().long()] = True
    assert bool((view.permuted_idx_to_token_idx.cpu()[~used] == -7).all())
    assert int(out.workspace[0]) == 0  # arrival counter reset


@pytest.mark.parametrize("tile_n", TILES)
@pytest.mark.parametrize("method", [1, 4])
@pytest.mark.parametrize("top_k", [4, 8])
def test_tiles_methods_topk(tile_n: int, method: int, top_k: int):
    x, w = _inputs(20, seed=tile_n + method + top_k)
    view, logits = _run(x, w, top_k, tile_n, method=method)
    _check_against_reference(view, logits, top_k, tile_n)


def test_ties_lower_expert_wins():
    m = 16
    x, w = _inputs(m, seed=3)
    # Groups of identical router rows give exactly equal logits.
    w = w[torch.arange(E, device="cuda") // 4].contiguous()
    x[5].zero_()  # all logits +0: experts 0..7, uniform weights
    view, logits = _run(x, w, 8, 8)
    assert torch.equal(logits[:, 0::4], logits[:, 3::4])
    ref = _check_against_reference(view, logits, 8, 8)
    assert ref["ids"][5].tolist() == list(range(8))
    assert torch.equal(view.expert_weights[5].float().cpu(), torch.full((8,), 0.125))
    for t in range(m):  # tied groups are taken lowest index first
        ids = ref["ids"][t].tolist()
        for e in ids:
            assert all(g in ids for g in range(e - e % 4, e))


def test_logits_scratch_matches_logits_out():
    x, w = _inputs(24, seed=11)
    a, _ = _run(x, w, 8, 16)
    b, _ = _run(x, w, 8, 16, logits=False)
    rows = a.expanded_idx_to_permuted_idx.long()
    ctas = int(a.num_non_exiting_ctas[0])
    for i, (ta, tb) in enumerate(zip(a.tensors(), b.tensors())):
        if i == 2:  # permuted->token: padding rows are never written
            assert torch.equal(ta[rows], tb[rows])
        elif i in (6, 7):  # CTA tables: live entries only
            assert torch.equal(ta[:ctas], tb[:ctas])
        elif i != 4:  # histogram scratch is not an output
            assert torch.equal(ta, tb)


@pytest.mark.parametrize("use_pdl", [False, True])
def test_cuda_graph_replay(use_pdl: bool):
    """Three back-to-back calls (distinct router weights, one shared output object)
    captured once and replayed with new inputs: the arrival counter self-resets.
    """
    m, top_k, tile_n = 12, 8, 8
    weights = [_inputs(1, seed=100 + i)[1] for i in range(3)]
    x = torch.zeros(m, K, device="cuda", dtype=torch.bfloat16)
    out = frr.FusedRoutingOutputs.allocate(frr.MAX_TOKENS, top_k, E, tile_n, "cuda")
    view = out.for_tokens(m)
    logits = [torch.empty(m, E, device="cuda", dtype=torch.bfloat16) for _ in weights]
    snaps: list[list[torch.Tensor]] = []

    def body():
        snaps.clear()
        for w, lg in zip(weights, logits):
            frr.fused_router_routing(
                x,
                w,
                view,
                top_k=top_k,
                tile_n=tile_n,
                renormalize_mode=4,
                use_pdl=use_pdl,
                logits_out=lg,
            )
            snaps.append([t.clone() for t in view.tensors()])

    body()  # warm up / build outside the capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for rep in range(4):
        x.copy_(_inputs(m, seed=500 + rep)[0])
        graph.replay()
        torch.accelerator.synchronize()
        assert int(out.workspace[0]) == 0
        for w, lg, snap in zip(weights, logits, snaps):
            _assert_logits_close(lg, x, w)
            ref = _reference(lg, top_k, tile_n)
            assert torch.equal(snap[3].cpu(), ref["weights"].cpu())
            assert torch.equal(snap[1].cpu(), ref["exp_to_perm"])
            assert int(snap[8][0]) == ref["total"]


def test_rejects_unsupported():
    x, w = _inputs(4, seed=1)
    out = frr.FusedRoutingOutputs.allocate(frr.MAX_TOKENS, 8, E, 8, "cuda")
    with pytest.raises(ValueError):
        frr.fused_router_routing(
            x, w, out, top_k=8, tile_n=8, renormalize_mode=0, use_pdl=False
        )
    with pytest.raises(ValueError):
        x65, _ = _inputs(65, seed=1)
        frr.fused_router_routing(
            x65, w, out, top_k=8, tile_n=8, renormalize_mode=4, use_pdl=False
        )


# ---- SM10x: against FlashInfer itself ----------------------------------------------


def _requires_sm10x():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FlashInfer tinygemm2_sm100 / trtllm-gen routing are SM10x-only")
    pytest.importorskip("flashinfer")


@pytest.mark.parametrize("m", [1, 4, 7, 8, 12, 16, 20, 24, 32, 48, 64])
def test_logits_bitwise_vs_tinygemm(m: int):
    _requires_sm10x()
    from flashinfer.gemm.routergemm import tinygemm_bf16

    x, w = _inputs(m, seed=1000 + m)
    _, logits = _run(x, w, 8, 8)
    ref = torch.empty(m, E, device="cuda", dtype=torch.bfloat16)
    zero_bias = torch.zeros(E, device="cuda", dtype=torch.bfloat16)
    tinygemm_bf16(x, w, ref, bias=zero_bias)
    assert torch.equal(logits, ref), f"max ulp {_bf16_ulp_diff(logits, ref)}"


def _expert_segments(e2p, p2t, batch, m, top_k, tile_n):
    """Expert -> sorted token list (within-expert order canonicalized), after checking
    that every expanded entry's permuted row holds its token.
    """
    e2p = e2p.cpu().long()
    tokens = torch.arange(m * top_k) // top_k
    assert torch.equal(p2t.cpu()[e2p], tokens.to(torch.int32))
    experts = batch.cpu()[e2p // tile_n].tolist()
    seg: dict[int, list[int]] = {}
    for e, t in zip(experts, tokens.tolist()):
        seg.setdefault(e, []).append(t)
    return sorted(e2p.tolist()), {e: sorted(v) for e, v in seg.items()}


@pytest.mark.parametrize("m", [1, 3, 4, 5, 8, 12, 16, 17, 24, 32, 48, 64])
@pytest.mark.parametrize("tile_n", [8, 16, 32, 64, 128, 256])
def test_routing_vs_flashinfer_runner(m: int, tile_n: int):
    _requires_sm10x()
    from flashinfer.fused_moe import core

    top_k = 8
    x, w = _inputs(m, seed=2000 + m)
    view, logits = _run(x, w, top_k, tile_n)
    canon = core.allocate_trtllm_moe_canonical_routing(
        logits, top_k=top_k, tile_n=tile_n
    )
    core.canonicalize_trtllm_moe_routing_(
        canon,
        logits,
        None,
        x,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=4,
        use_routing_scales_on_input=False,
        use_deep_seek_fp8=False,
        norm_topk_prob=True,
        enable_pdl=False,
    )
    torch.accelerator.synchronize()
    # canon.scratch: num_tokens_per_expert (not written by the from-logits runner),
    # padded size, expanded->permuted, permuted->token, histogram, CTA tables, CTAs.
    (_, padded, e2p, p2t, _, batch, mn, nctas) = canon.scratch
    assert torch.equal(view.expert_weights, canon.expert_weights)
    assert torch.equal(
        canon.routing_replay_ids.to(torch.int32).cpu(), _topk_ids(logits, top_k).cpu()
    )
    assert int(view.num_non_exiting_ctas[0]) == int(nctas[0])
    assert int(view.total_num_padded_tokens[0]) == int(padded[0])
    total = int(nctas[0])
    assert torch.equal(view.cta_idx_xy_to_batch_idx[:total], batch[:total])
    assert torch.equal(view.cta_idx_xy_to_mn_limit[:total], mn[:total])
    # Extents equal FlashInfer's own allocation for this M.
    assert view.permuted_idx_to_token_idx.numel() == p2t.numel()
    assert view.cta_idx_xy_to_batch_idx.numel() == batch.numel()
    # Same permuted rows, same tokens per expert (within-expert order canonicalized).
    assert _expert_segments(
        view.expanded_idx_to_permuted_idx,
        view.permuted_idx_to_token_idx,
        view.cta_idx_xy_to_batch_idx,
        m,
        top_k,
        tile_n,
    ) == _expert_segments(e2p, p2t, batch, m, top_k, tile_n)
    if m <= 16:  # block / dynamic-block kernels: token order, fully identical
        assert torch.equal(view.expanded_idx_to_permuted_idx, e2p)
