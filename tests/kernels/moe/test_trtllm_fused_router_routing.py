# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE router + routing (VLLM_MOE_FUSED_ROUTING_MAX_TOKENS) feeding
FlashInfer's MXFP8 prepared-metadata body, against the router GEMM + stock
``trtllm_fp8_block_scale_moe`` on the same input and tactic.

The GPU tests need SM107 and the GS2_ROUTE patched FlashInfer module
(``GS2_ROUTE=1``, optionally ``GS2_ROUTE_PREBUILT``) in the environment.
"""

import types

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts import trtllm_fused_routing
from vllm.model_executor.layers.fused_moe.experts.trtllm_fused_routing import (
    FusedRouterInput,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.platforms import current_platform

E, TOP_K, H, INTER = 256, 8, 2048, 512
FUSE_MAX = 64


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_FUSED_ROUTING_MAX_TOKENS", str(FUSE_MAX))
    monkeypatch.setenv("VLLM_FI_SM107_MOE_PDL_MAX_TOKENS", "1024")


@pytest.mark.parametrize(
    ("num_tokens", "dtype", "contiguous", "expected"),
    [
        (0, torch.bfloat16, True, False),
        (1, torch.bfloat16, True, True),
        (FUSE_MAX, torch.bfloat16, True, True),
        (FUSE_MAX + 1, torch.bfloat16, True, False),
        (8, torch.float16, True, False),
        (8, torch.bfloat16, False, False),
    ],
)
def test_runner_skips_router_gemm_only_within_the_knob(
    num_tokens, dtype, contiguous, expected
):
    runner = types.SimpleNamespace(_fused_router_routing_layer=True)
    x = torch.empty(num_tokens, 2 * H if not contiguous else H, dtype=dtype)
    if not contiguous:
        x = x[:, :H]
    assert MoERunner._fused_router_routing_applies(runner, x) is expected


def test_runner_knob_off_never_skips(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_FUSED_ROUTING_MAX_TOKENS", "0")
    runner = types.SimpleNamespace(_fused_router_routing_layer=True)
    x = torch.empty(4, H, dtype=torch.bfloat16)
    assert not MoERunner._fused_router_routing_applies(runner, x)


def _gpu_ready() -> str | None:
    if not current_platform.is_cuda() or not current_platform.is_device_capability(107):
        return "needs SM107"
    from vllm.model_executor.layers.fused_moe import flashinfer_exact_routing

    flashinfer_exact_routing.maybe_install()
    if not flashinfer_exact_routing.installed():
        return "needs the GS2_ROUTE patched FlashInfer module (GS2_ROUTE=1)"
    return None


_SKIP_REASON = _gpu_ready()
gpu = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


class _Gate(torch.nn.Module):
    """Router linear as the Qwen3.6 MoE block builds it (low-M tinygemm)."""

    def __init__(self, weight: torch.Tensor):
        from vllm.model_executor.kernels.linear.lowm_bf16_gemm import (
            maybe_use_lowm_bf16_gemm,
        )
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()
        assert maybe_use_lowm_bf16_gemm(self)

    def forward(self, x):
        return self.quant_method.apply(self, x), None


def _make_experts(
    device, routing=RoutingMethodType.RenormalizeNaive, deferred=True, seed=0
):
    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
        TrtLlmFp8ExpertsMonolithic,
    )

    g = torch.Generator(device=device).manual_seed(seed)

    def fp8(*shape):
        return (torch.randn(*shape, device=device, generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )

    def scales(*shape):  # e8m0 bytes around 2^0
        return torch.randint(
            124, 129, shape, device=device, dtype=torch.uint8, generator=g
        )

    moe_config = FusedMoEConfig(
        num_experts=E,
        experts_per_token=TOP_K,
        hidden_dim=H,
        intermediate_size=INTER,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=routing,
        max_num_tokens=256,
    )
    moe_config.defer_moe_finalize_local = deferred
    quant_config = FusedMoEQuantConfig.make(
        "mxfp8",
        w1_scale=scales(E, 2 * INTER, H // 32),
        w2_scale=scales(E, H, INTER // 32),
        block_shape=[1, 32],
        is_scale_swizzled=False,
    )
    experts = TrtLlmFp8ExpertsMonolithic(moe_config, quant_config)
    w1, w2 = fp8(E, 2 * INTER, H), fp8(E, H, INTER)
    gate = _Gate(
        (torch.randn(E, H, device=device, generator=g) * 0.02).to(torch.bfloat16)
    )
    return experts, w1, w2, gate


def _inputs(num_tokens, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, H, device=device, generator=g).to(torch.bfloat16)
    xq = (torch.randn(num_tokens, H, device=device, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    xs = torch.randint(
        124, 129, (num_tokens, H // 32), device=device, dtype=torch.uint8, generator=g
    )
    return x, xq, xs


def _apply(experts, w1, w2, xq, xs, router_logits):
    out = experts.apply(
        hidden_states=xq,
        w1=w1,
        w2=w2,
        router_logits=router_logits,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        a1q_scale=xs,
        apply_router_weight_on_input=False,
    )
    if isinstance(out, UnfinalizedMoEOutput):
        # DLC-8's consumer reduction, on the rows the deferred op returns.
        m = out.expert_weights.shape[0]
        reduced = torch.ops.vllm.moe_finalize(
            out.gemm2_permuted[: m * TOP_K],
            out.expert_weights,
            out.expanded_idx_to_permuted_idx,
        )
        return reduced, out.expert_weights.clone()
    return out.clone(), None


def _tactics(experts, num_tokens, device):
    """One valid FlashInfer tactic per tile_n for this token count."""
    import flashinfer.fused_moe.core as fi_core
    from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

    runtime = fi_core.get_trtllm_moe_sm100_module()
    runner = runtime.MoERunner(
        top_k=TOP_K,
        num_local_experts=E,
        dtype_act=fi_core.DtypeTrtllmGen.MxE4m3,
        dtype_weights=fi_core.DtypeTrtllmGen.MxE4m3,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        hidden_size=H,
        intermediate_size=INTER,
        activation_type=3,
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        num_experts=E,
    )
    hs = torch.empty(num_tokens, H, device=device, dtype=torch.float8_e4m3fn)
    inputs = fi_core.MoeRunnerInputs(
        output=torch.empty(num_tokens, 0, device=device, dtype=torch.bfloat16),
        routing_logits=torch.empty(num_tokens, E, device=device, dtype=torch.bfloat16),
        topk_ids=None,
        expert_weights=None,
        hidden_states=hs,
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    ).to_list()
    by_tile: dict[int, list[int]] = {}
    for tile_n, config in runner.get_valid_tactics(inputs, None):
        by_tile.setdefault(int(tile_n), [int(tile_n), int(config)])
    return list(by_tile.values())


@pytest.fixture
def forced_tactic(monkeypatch):
    """Make the autotuner return one fixed tactic for the FP8 block-scale MoE,
    so the stock op and the fused path run the same body.
    """
    from flashinfer.autotuner import AutoTuner

    holder: dict[str, list[int]] = {}
    orig = AutoTuner.choose_one

    def choose_one(self, custom_op, runners, *args, **kwargs):
        if custom_op == "flashinfer::trtllm_fp8_block_scale_moe" and holder:
            return runners[0], list(holder["tactic"])
        return orig(self, custom_op, runners, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "choose_one", choose_one)
    return holder


@pytest.fixture
def fused_launches(monkeypatch):
    """Token counts of the fused router + routing kernel launches."""
    from vllm.model_executor.layers.fused_moe import fused_router_routing as frr

    calls: list[int] = []
    orig = frr.fused_router_routing

    def spy(x, *args, **kwargs):
        calls.append(x.shape[0])
        return orig(x, *args, **kwargs)

    monkeypatch.setattr(frr, "fused_router_routing", spy)
    return calls


@gpu
@pytest.mark.parametrize("deferred", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 13, 16, 17, 24, 32, 48, 64])
@pytest.mark.parametrize(
    "routing", [RoutingMethodType.RenormalizeNaive, RoutingMethodType.Renormalize]
)
def test_fused_matches_stock(
    num_tokens, deferred, routing, forced_tactic, fused_launches
):
    device = torch.device("cuda")
    experts, w1, w2, gate = _make_experts(device, routing, deferred)
    assert experts.supports_fused_router_routing(gate)
    x, xq, xs = _inputs(num_tokens, device, seed=num_tokens)
    for tactic in _tactics(experts, num_tokens, device):
        forced_tactic["tactic"] = tactic
        ref = _apply(experts, w1, w2, xq, xs, gate(x)[0])
        fused_launches.clear()
        got = _apply(experts, w1, w2, xq, xs, FusedRouterInput(x, gate))
        assert fused_launches == [num_tokens], tactic
        torch.accelerator.synchronize()
        assert torch.equal(got[0], ref[0]), tactic
        if deferred:
            assert torch.equal(got[1], ref[1]), tactic


@gpu
def test_cuda_graph_capture_and_replay(forced_tactic, fused_launches):
    """Graphs of fused calls at several sizes (captured after an eager call of
    each size; the shared workspace grows between captures) replay to the
    stock result on new inputs.
    """
    device = torch.device("cuda")
    experts, w1, w2, gate = _make_experts(device, seed=1)
    sizes = [4, 32, 16, 64]
    graphs = []
    for m in sizes:
        forced_tactic["tactic"] = _tactics(experts, m, device)[0]
        x, xq, xs = _inputs(m, device, seed=100 + m)
        _apply(experts, w1, w2, xq, xs, FusedRouterInput(x, gate))  # prepares
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        fused_launches.clear()
        with torch.cuda.graph(graph, stream=stream):
            out = _apply(experts, w1, w2, xq, xs, FusedRouterInput(x, gate))
        assert fused_launches == [m]
        graphs.append((m, forced_tactic["tactic"], graph, (x, xq, xs), out))
    for _ in range(2):  # replay each graph after all others were captured/run
        for m, tactic, graph, (x, xq, xs), out in graphs:
            forced_tactic["tactic"] = tactic
            nx, nxq, nxs = _inputs(m, device, seed=1000 + m)
            x.copy_(nx)
            xq.copy_(nxq)
            xs.copy_(nxs)
            graph.replay()
            ref = _apply(experts, w1, w2, xq, xs, gate(x)[0])
            torch.accelerator.synchronize()
            assert torch.equal(out[0], ref[0]), m
            assert torch.equal(out[1], ref[1]), m


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn()
    return graph, out


@gpu
def test_unprepared_capture_uses_stock_path(forced_tactic, fused_launches):
    """No buffer is created during capture: a capture before any eager call
    or preparation records the stock call, with the stock result.
    """
    device = torch.device("cuda")
    experts, w1, w2, gate = _make_experts(device, seed=2)
    m = 12
    forced_tactic["tactic"] = _tactics(experts, m, device)[0]
    trtllm_fused_routing._STATES.clear()
    x, xq, xs = _inputs(m, device, seed=7)
    _apply(experts, w1, w2, xq, xs, gate(x)[0])  # stock warmup only
    graph, out = _capture(
        lambda: _apply(experts, w1, w2, xq, xs, FusedRouterInput(x, gate))
    )
    assert not trtllm_fused_routing._STATES and not fused_launches
    graph.replay()
    ref = _apply(experts, w1, w2, xq, xs, gate(x)[0])
    torch.accelerator.synchronize()
    assert torch.equal(out[0], ref[0])


@gpu
def test_prepared_capture_without_eager_call(
    monkeypatch, forced_tactic, fused_launches
):
    """VLLM captures decode graphs without eager warmup: after
    prepare_fused_router_routing (kernel_warmup) the first call of each size
    is captured on the fused path, and replays to the stock result.
    """
    device = torch.device("cuda")
    monkeypatch.setenv("VLLM_MOE_FUSED_ROUTING_MAX_TOKENS", "16")
    experts, w1, w2, gate = _make_experts(device, seed=5)
    forced_tactic["tactic"] = _tactics(experts, 16, device)[0]
    trtllm_fused_routing._STATES.clear()
    experts.prepare_fused_router_routing(w1, w2)
    for m in (1, 7, 16):
        x, xq, xs = _inputs(m, device, seed=20 + m)
        fused_launches.clear()
        graph, out = _capture(
            lambda x=x, xq=xq, xs=xs: _apply(
                experts, w1, w2, xq, xs, FusedRouterInput(x, gate)
            )
        )
        assert fused_launches == [m]
        graph.replay()
        ref = _apply(experts, w1, w2, xq, xs, gate(x)[0])
        torch.accelerator.synchronize()
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]), m


@gpu
def test_calls_outside_the_fused_range_use_stock_path(
    monkeypatch, forced_tactic, fused_launches
):
    device = torch.device("cuda")
    experts, w1, w2, gate = _make_experts(device, seed=3)
    monkeypatch.setenv("VLLM_MOE_FUSED_ROUTING_MAX_TOKENS", "8")
    m = 16
    forced_tactic["tactic"] = _tactics(experts, m, device)[0]
    x, xq, xs = _inputs(m, device, seed=11)
    got = _apply(experts, w1, w2, xq, xs, FusedRouterInput(x, gate))
    assert not fused_launches
    ref = _apply(experts, w1, w2, xq, xs, gate(x)[0])
    assert torch.equal(got[0], ref[0])


@gpu
def test_unsupported_layers_are_not_offered_the_fused_path(monkeypatch):
    from vllm.model_executor.layers.fused_moe import flashinfer_exact_routing

    device = torch.device("cuda")
    experts, _, _, gate = _make_experts(device, seed=4)
    assert experts.supports_fused_router_routing(gate)

    deepseek, _, _, _ = _make_experts(device, RoutingMethodType.DeepSeekV3, seed=4)
    assert not deepseek.supports_fused_router_routing(gate)

    biased = _Gate(gate.weight.data.clone())
    biased.bias = torch.nn.Parameter(torch.zeros(E, device=device))
    assert not experts.supports_fused_router_routing(biased)

    experts.gemm1_alpha = torch.ones(E, device=device)
    assert not experts.supports_fused_router_routing(gate)
    experts.gemm1_alpha = None

    monkeypatch.setattr(flashinfer_exact_routing, "_installed", False)
    assert not experts.supports_fused_router_routing(gate)
    monkeypatch.setattr(flashinfer_exact_routing, "_installed", True)

    monkeypatch.setenv("VLLM_MOE_FUSED_ROUTING_MAX_TOKENS", "0")
    assert not experts.supports_fused_router_routing(gate)


@gpu
def test_uses_the_autotuned_tactic_of_the_stock_call(fused_launches):
    """With a real autotuner cache (as after vLLM's warmup tuning) the fused
    path finds the stock call's tuned entry and matches it bitwise.
    """
    import flashinfer

    device = torch.device("cuda")
    experts, w1, w2, gate = _make_experts(device, seed=6)
    m = 8
    x, xq, xs = _inputs(m, device, seed=31)
    with flashinfer.autotune(True):
        _apply(experts, w1, w2, xq, xs, gate(x)[0])
    ref = _apply(experts, w1, w2, xq, xs, gate(x)[0])
    got = _apply(experts, w1, w2, xq, xs, FusedRouterInput(x, gate))
    assert fused_launches == [m]
    torch.accelerator.synchronize()
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
