# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused router GEMV + routing in front of FlashInfer's trtllm-gen MXFP8 MoE.

At decode token counts the critical routed branch of every MoE layer is the
router GEMM (tinygemm), FlashInfer's routing kernel(s) and the FC1/FC2 bodies.
With ``VLLM_MOE_FUSED_ROUTING_MAX_TOKENS=N`` (default 0 = off), qualifying
calls of at most N tokens skip the separate router GEMM: the runner hands the
experts a :class:`FusedRouterInput` instead of router logits, one kernel
(``fused_moe/fused_router_routing.py``) computes the bf16 router logits, the
top-k + renormalize and the TRT-LLM routing metadata into graph-stable
buffers, and FlashInfer's prepared-metadata ("DA") MXFP8 body runs FC1/FC2
from them. With the deferred finalize (DLC-8) the body stops after FC2, which
needs the GS2_ROUTE patched FlashInfer module (``flashinfer_exact_routing``).

Exactness: the body runs the tactic the autotuner chose for the stock
``trtllm_fp8_block_scale_moe`` call (same cache entry) on routing metadata
equal to FlashInfer's routing of the same logits (within-expert row order
aside, as in FlashInfer's own kernels), so the routed output equals the stock
path's whenever the fused kernel's logits equal the router GEMM's.

Buffers: one routing-metadata set per tile_n and one field-wise maximum body
workspace per device and MoE shape, shared by every layer of that shape. The
routed branch of all layers runs on one stream and a layer's (deferred)
outputs are consumed before the next layer's MoE starts. Buffers are created
outside CUDA-graph capture (eager warmup of each size); a call that would
need a new one during capture takes the stock path. Workspaces replaced by
larger ones stay alive: graphs captured earlier keep their addresses.
"""

import dataclasses
import functools
import types
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.moe_output import (
    UnfinalizedMoEOutput,
    convert_flashinfer_moe_output,
)
from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class FusedRouterInput:
    """Passed to the routed experts in place of router logits: the router
    input ([tokens, hidden] bf16, contiguous) and the router layer.
    """

    hidden_states: torch.Tensor
    gate: torch.nn.Module

    def logits(self) -> torch.Tensor:
        """Router logits exactly as the runner computes them (stock path)."""
        return self.gate(self.hidden_states)[0]


@functools.cache
def _kernel_module() -> Any:
    """The fused kernel's module with its extension built (once, eagerly;
    the first qualification check runs outside CUDA-graph capture), or None
    when it cannot be built here.
    """
    from vllm.model_executor.layers.fused_moe import fused_router_routing

    try:
        fused_router_routing.load()
    except Exception:
        logger.exception("Fused MoE router + routing: kernel build failed")
        return None
    return fused_router_routing


def router_gate_supported(gate: torch.nn.Module, num_experts: int, hidden: int) -> bool:
    """Whether ``gate`` is a bias-free, unquantized bf16 [E, K] linear."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    weight = getattr(gate, "weight", None)
    return (
        isinstance(getattr(gate, "quant_method", None), UnquantizedLinearMethod)
        and getattr(gate, "bias", None) is None
        and weight is not None
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (num_experts, hidden)
        and weight.is_contiguous()
    )


def fused_router_routing_supported(
    *,
    gate: torch.nn.Module,
    routing_method: RoutingMethodType,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    hidden_size: int,
    has_swiglu_params: bool,
) -> bool:
    """Static part of the qualification (MXFP8 block-scale experts only; the
    caller checks that). Everything here holds for the life of the layer.
    """
    from vllm.model_executor.layers.fused_moe import (
        flashinfer_exact_routing,
        fused_router_routing,
    )
    from vllm.platforms import current_platform

    return (
        envs.VLLM_MOE_FUSED_ROUTING_MAX_TOKENS > 0
        and current_platform.is_cuda()
        and current_platform.is_device_capability(107)
        # Shape and routing-method support at the smallest call; token count
        # and tile are checked per call.
        and fused_router_routing.is_supported(
            1, num_experts, top_k, hidden_size, routing_method, 8
        )
        and num_local_experts == num_experts
        and not has_swiglu_params
        and router_gate_supported(gate, num_experts, hidden_size)
        and flashinfer_exact_routing.installed()
        and _kernel_module() is not None
    )


@dataclasses.dataclass
class _State:
    """Graph-stable buffers for one device and MoE shape (see module doc)."""

    max_tokens: int
    # [max_tokens, E] bf16 stand-in for the logits in the FlashInfer ABI
    # (autotuner key and body launcher arguments); never read.
    logits: torch.Tensor
    # Stock placeholders: empty precomputed-routing tensors, deferred output.
    empty_ids: torch.Tensor
    empty_weights: torch.Tensor
    empty_output: torch.Tensor
    routing: dict[int, Any] = dataclasses.field(default_factory=dict)
    workspace: tuple[torch.Tensor, ...] | None = None
    prepared: set[tuple[int, int, int, bool]] = dataclasses.field(default_factory=set)
    retired: list[tuple[torch.Tensor, ...]] = dataclasses.field(default_factory=list)

    def add_workspace(self, fields: list[torch.Tensor]) -> None:
        if self.workspace is None:
            self.workspace = tuple(fields)
            return
        grown = tuple(
            new if new.numel() > cur.numel() else cur
            for cur, new in zip(self.workspace, fields)
        )
        if any(g is not c for g, c in zip(grown, self.workspace)):
            self.retired.append(self.workspace)
            self.workspace = grown


_STATES: dict[tuple[Any, ...], _State] = {}


def _new_state(device: torch.device, num_experts: int) -> _State:
    max_tokens = min(
        envs.VLLM_MOE_FUSED_ROUTING_MAX_TOKENS, _kernel_module().MAX_TOKENS
    )
    bf16 = torch.bfloat16
    return _State(
        max_tokens=max_tokens,
        logits=torch.empty(max_tokens, num_experts, dtype=bf16, device=device),
        empty_ids=torch.empty(0, dtype=torch.int32, device=device),
        empty_weights=torch.empty(0, dtype=bf16, device=device),
        empty_output=torch.empty(max_tokens, 0, dtype=bf16, device=device),
    )


def _fallback(reason: str) -> None:
    logger.warning_once(
        "Fused MoE router + routing: %s; using the router GEMM + FlashInfer "
        "routing for this call.",
        reason,
    )


@dataclasses.dataclass
class _Call:
    """One planned call: prepared buffers, body and its launcher arguments."""

    state: _State
    slot: Any  # FusedRoutingOutputs view for this token count
    body: Any
    body_runner: Any
    inputs: list[torch.Tensor | None]
    runner_kwargs: dict[str, Any]
    output: torch.Tensor


def _plan(
    experts: Any,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation_type: int,
    deferred: bool,
    use_pdl: bool,
) -> _Call | None:
    """Resolve the stock call's tactic and the prepared buffers for it,
    creating them unless capturing. None: the call takes the stock path.
    """
    import flashinfer.fused_moe.core as fi_core
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

    from vllm.v1.worker.ubatching import dbo_enabled

    tuner = AutoTuner.get()
    num_tokens, hidden = hidden_states.shape
    if (
        tuner.is_tuning_mode  # the autotuner profiles the stock op
        or not 0 < num_tokens <= envs.VLLM_MOE_FUSED_ROUTING_MAX_TOKENS
        or experts.routing_replay_capture_fn is not None
        or dbo_enabled()  # two ubatches would share the buffers
    ):
        return None
    num_experts = experts.local_num_experts
    top_k = experts.topk
    inter = experts.intermediate_size_per_partition
    device = hidden_states.device
    capturing = torch.cuda.is_current_stream_capturing()
    state_key = (device, num_experts, top_k, hidden, inter, activation_type)
    state = _STATES.get(state_key)
    if state is None:
        if capturing:
            _fallback("buffers first needed in CUDA-graph capture")
            return None
        state = _STATES[state_key] = _new_state(device, num_experts)
    if num_tokens > state.max_tokens:
        return None

    # The tactic the stock call would run: same runner, inputs and tuning
    # config as trtllm_fp8_block_scale_moe_op builds, so the same cache entry.
    runtime = fi_core.get_trtllm_moe_sm100_module()
    moe_runner = runtime.MoERunner(
        top_k=top_k,
        num_local_experts=num_experts,
        dtype_act=fi_core.DtypeTrtllmGen.MxE4m3,
        dtype_weights=fi_core.DtypeTrtllmGen.MxE4m3,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        hidden_size=hidden,
        intermediate_size=inter,
        activation_type=activation_type,
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        num_experts=num_experts,
        num_fused_shared_experts=0,
    )
    output = (
        state.empty_output[:num_tokens]
        if deferred
        else torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    )
    moe_inputs = fi_core.MoeRunnerInputs(
        output=output,
        routing_logits=state.logits[:num_tokens],
        topk_ids=state.empty_ids,
        expert_weights=state.empty_weights,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    tuning_config = moe_runner._make_tuning_config(
        moe_inputs,
        tune_max_num_tokens=fi_moe_largest_bucket(experts.moe_config),
        routing_input_mode=fi_core.RoutingInputMode.FromLogits,
        use_cuda_graph=True,
        use_cold_l2_cache=True,
    )
    inputs = moe_inputs.to_list()
    _, tactic = tuner.choose_one(
        "flashinfer::trtllm_fp8_block_scale_moe", [moe_runner], tuning_config, inputs
    )
    if not isinstance(tactic, (list, tuple)) or len(tactic) != 2:
        _fallback(f"no autotuned tactic for {num_tokens} tokens")
        return None
    tile_n, config = int(tactic[0]), int(tactic[1])
    kernel = _kernel_module()
    if not kernel.is_supported(
        num_tokens, num_experts, top_k, hidden, experts.routing_method_type, tile_n
    ):
        return None
    routing = state.routing.get(tile_n)
    if routing is None:
        if capturing:
            _fallback(f"routing buffers for tile {tile_n} first needed in capture")
            return None
        routing = state.routing[tile_n] = kernel.FusedRoutingOutputs.allocate(
            state.max_tokens, top_k, num_experts, tile_n, device
        )
    call = _Call(
        state=state,
        slot=routing.for_tokens(num_tokens),
        body=types.SimpleNamespace(tile_n=tile_n, tactic=config),
        body_runner=runtime.DABodyRunner(moe_runner),
        inputs=inputs,
        runner_kwargs=dict(
            routing_input_mode=int(fi_core.RoutingInputMode.FromLogits),
            routing_bias=None,
            gemm1_weights=w1,
            gemm1_weights_scale=experts.quant_config.w1_scale,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w2,
            gemm2_weights_scale=experts.quant_config.w2_scale,
            num_experts=num_experts,
            n_group=None,
            topk_group=None,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=int(experts.routing_method_type),
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.MajorK),
            do_finalize=not deferred,
            enable_pdl=use_pdl,
            num_fused_shared_experts=0,
            norm_topk_prob=True,
            routing_replay_out=None,
        ),
        output=output,
    )
    body_key = (num_tokens, tile_n, config, deferred)
    if body_key not in state.prepared:
        if capturing:
            _fallback(f"DA body {body_key} first needed in capture")
            return None
        fields = call.body_runner.prepare_body(
            inputs, call.body, call.slot, **call.runner_kwargs
        )
        state.add_workspace([fi_core._torch_view_of_ffi_tensor(f) for f in fields])
        state.prepared.add(body_key)
    return call


def prepare_fused_router_routing(
    experts: Any, w1: torch.Tensor, w2: torch.Tensor, *, activation_type: int
) -> None:
    """Create the buffers and DA bodies of every token count the fused path
    serves, for the tactics the autotuner chose. vLLM captures its decode
    CUDA graphs without eager warmup runs, so this runs before capture
    (kernel_warmup, after FlashInfer autotuning); layers of one shape share
    them, so repeated calls are cheap.
    """
    if torch.cuda.is_current_stream_capturing():
        return
    hidden = experts.hidden_dim
    max_tokens = min(
        envs.VLLM_MOE_FUSED_ROUTING_MAX_TOKENS, _kernel_module().MAX_TOKENS
    )
    # Only shapes matter for planning: the inputs are never read here.
    hidden_states = torch.zeros(
        max_tokens, hidden, dtype=torch.float8_e4m3fn, device=w1.device
    )
    scales = torch.zeros(max_tokens, hidden // 32, dtype=torch.uint8, device=w1.device)
    # Both finalize modes: the deferred finalize is switched on by the first
    # run of the deferring op, which need not have happened yet.
    for num_tokens in range(1, max_tokens + 1):
        for deferred in (True, False):
            _plan(
                experts,
                hidden_states[:num_tokens],
                scales[:num_tokens],
                w1,
                w2,
                activation_type=activation_type,
                deferred=deferred,
                use_pdl=False,
            )


def run_fused_router_routing(
    experts: Any,
    router: FusedRouterInput,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation_type: int,
    deferred: bool,
    use_pdl: bool,
) -> torch.Tensor | UnfinalizedMoEOutput | None:
    """Run one MXFP8 trtllm-gen MoE call as fused router/routing kernel +
    FlashInfer DA body. ``experts`` is the TrtLlmFp8ExpertsMonolithic that
    accepted ``router`` (static qualification done). Returns None when this
    call must take the stock path (the caller then computes the logits).
    """
    call = _plan(
        experts,
        hidden_states,
        hidden_states_scale,
        w1,
        w2,
        activation_type=activation_type,
        deferred=deferred,
        use_pdl=use_pdl,
    )
    if call is None:
        return None
    workspace = call.state.workspace
    assert workspace is not None
    top_k = experts.topk
    _kernel_module().fused_router_routing(
        router.hidden_states,
        router.gate.weight,
        call.slot,
        top_k=top_k,
        tile_n=call.body.tile_n,
        renormalize_mode=int(experts.routing_method_type),
        use_pdl=use_pdl,
    )
    call.body_runner.forward_from_metadata(
        call.inputs, call.body, call.slot, workspace, **call.runner_kwargs
    )
    logger.info_once(
        "Fused MoE router + routing active (calls of <= %d tokens).",
        call.state.max_tokens,
    )
    if not deferred:
        return call.output
    slot = call.slot
    return convert_flashinfer_moe_output(
        [workspace[2], slot.expert_weights, slot.expanded_idx_to_permuted_idx],
        do_finalize=False,
        num_tokens=hidden_states.shape[0],
        top_k=top_k,
    )
