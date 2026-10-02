# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Fused MoE router GEMV + top-K renormalize routing + TRT-LLM routing metadata.

One CUDA launch replaces, for small token counts (M <= 64, E = 256, K = 2048,
top-K <= 8, Renormalize / RenormalizeNaive, no bias, no EP):

* the router GEMM ``logits = bf16(x @ W.T)`` that vLLM runs with FlashInfer's
  ``tinygemm2_sm100`` (zero-bias path), and
* FlashInfer's trtllm-gen routing from those logits (``routingCustom``:
  static/dynamic block kernels up to 16 tokens, BlockScores + permutation above),

writing exactly the routing metadata FlashInfer's MXFP8 MoE body consumes
(``TrtllmMoERoutingMetadataSlot``: padded size, expanded->permuted and
permuted->token maps, bf16 expert weights, per-expert counts, CTA tables, live CTA
count), so the body can run through FlashInfer's prepared-metadata (DA) path.

Numerics (see the module's CUDA source header for the instruction-level detail):

* Logits mirror ``tinygemm2_sm100``'s per-output accumulation: four fp32 chains of
  ``mma.sync.m16n8k16`` (chain w covers K blocks ``[(4*i + w)*256, +256)``, k16 slices
  in ascending order), combined as ``((c0 + c1) + c2) + c3``, plus the zero bias, then
  round-to-nearest bf16.
* Top-K, softmax-over-top-K and the metadata follow FlashInfer 0.6.18.post1's
  ``routingCustom`` (NoOp preprocess + Softmax postprocess, which FlashInfer uses for
  both Renormalize and RenormalizeNaive) instruction for instruction as FlashInfer
  compiles it (``-use_fast_math``): packed (bf16 key, 65535 - expert) maximum with the
  lower expert index winning ties, ``ex2.approx.ftz`` exponentials, the warp butterfly
  sum, ``div.approx.ftz`` and a bf16 round. Within-expert rows are in token order, as
  in FlashInfer's block / dynamic-block kernels.

The kernel: 16 expert tiles x ceil(M / 8) token tiles of 512-thread CTAs (one 8-token
tile per CTA by default). Each CTA streams its 16 router rows into shared memory with
``cp.async`` *before* ``griddepcontrol.wait`` (the weight is constant), then its token
rows, runs the four mma chains of its tile, combines them in shared memory and stores
bf16 logits. The last CTA to arrive (self-resetting arrival counter in
``FusedRoutingOutputs.workspace``) does top-K, softmax and the metadata for all tokens,
one warp per token. Under PDL every global write and every read of ``x`` is after the
wait; dependents are triggered when the routing CTA starts (``EARLY_TRIGGER``; FC1
waits for this grid's completion before reading routing, so only its launch and
pre-wait prologue overlap the routing) or, if disabled, after the last write.

The CUDA source is kept in this module (``_SOURCE``); ``load()`` writes it into the
build directory under VLLM_CACHE_ROOT and builds it with torch.utils.cpp_extension
for the current device's arch-specific target (cpp_extension's file lock serializes
concurrent builders).
"""

import hashlib
import os
from dataclasses import dataclass

import torch

from vllm import envs

MAX_TOKENS = 64
NUM_EXPERTS = 256
HIDDEN_SIZE = 2048
MAX_TOP_K = 8
# FlashInfer RoutingMethodType values with TopK -> Softmax-over-TopK semantics.
_RENORMALIZE = 1
_RENORMALIZE_NAIVE = 4
_SUPPORTED_ROUTING_METHODS = (_RENORMALIZE, _RENORMALIZE_NAIVE)
# int32 words at the head of the workspace (word 0: arrival counter); the bf16
# logits scratch follows.
_WORKSPACE_HEADER_INTS = 32

_ext: list = []  # [module] once load() built or loaded it


def max_num_ctas_in_batch_dim(
    num_tokens: int, top_k: int, num_experts: int, tile_n: int
) -> int:
    """FlashInfer ``Routing::getMaxNumCtasInBatchDim`` (runner.h)."""
    remaining = num_tokens * top_k
    filled = min(num_experts, remaining)
    remaining -= filled
    return filled + (remaining // tile_n if remaining > 0 else 0)


def max_permuted_padded_count(
    num_tokens: int, top_k: int, num_experts: int, tile_n: int
) -> int:
    """FlashInfer ``Routing::getMaxPermutedPaddedCount`` (runner.h)."""
    return max_num_ctas_in_batch_dim(num_tokens, top_k, num_experts, tile_n) * tile_n


def is_supported(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    routing_method_type: int,
    tile_n: int,
) -> bool:
    """Whether ``fused_router_routing`` covers this call."""
    return (
        0 < num_tokens <= MAX_TOKENS
        and num_experts == NUM_EXPERTS
        and 0 < top_k <= MAX_TOP_K
        and hidden_size == HIDDEN_SIZE
        and int(routing_method_type) in _SUPPORTED_ROUTING_METHODS
        and tile_n > 0
        and tile_n & (tile_n - 1) == 0
    )


@dataclass(frozen=True)
class FusedRoutingOutputs:
    """Routing metadata in FlashInfer's ``TrtllmMoERoutingMetadataSlot`` layout plus
    the fused kernel's workspace.

    The nine metadata fields have the names, dtypes and order of
    ``flashinfer.fused_moe.core.TrtllmMoERoutingMetadataSlot`` (``tensors()`` is the
    native nine-tensor ABI). ``allocate`` sizes them for ``max_tokens``;
    ``for_tokens(M)`` returns prefix views with FlashInfer's exact extents for ``M``
    tokens (the ones the stock launcher allocates), sharing the storage and the
    workspace. The workspace (int32, zero-initialized) holds the kernel's
    self-resetting arrival counter and a bf16 logits scratch; one instance must not
    be used by two concurrently running streams.
    """

    tile_n: int
    total_num_padded_tokens: torch.Tensor
    expanded_idx_to_permuted_idx: torch.Tensor
    permuted_idx_to_token_idx: torch.Tensor
    expert_weights: torch.Tensor
    expert_count_histogram: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    cta_idx_xy_to_batch_idx: torch.Tensor
    cta_idx_xy_to_mn_limit: torch.Tensor
    num_non_exiting_ctas: torch.Tensor
    workspace: torch.Tensor

    @classmethod
    def allocate(
        cls,
        max_tokens: int,
        top_k: int,
        num_experts: int,
        tile_n: int,
        device: torch.device | str,
    ) -> "FusedRoutingOutputs":
        max_ctas = max_num_ctas_in_batch_dim(max_tokens, top_k, num_experts, tile_n)
        max_padded = max_ctas * tile_n

        def i32(n: int) -> torch.Tensor:
            return torch.empty(n, dtype=torch.int32, device=device)

        workspace = torch.zeros(
            _WORKSPACE_HEADER_INTS + (max_tokens * num_experts + 1) // 2,
            dtype=torch.int32,
            device=device,
        )
        return cls(
            tile_n=tile_n,
            total_num_padded_tokens=i32(1),
            expanded_idx_to_permuted_idx=i32(max_tokens * top_k),
            permuted_idx_to_token_idx=i32(max_padded + 1),
            expert_weights=torch.empty(
                max_tokens, top_k, dtype=torch.bfloat16, device=device
            ),
            expert_count_histogram=i32(max(2 * num_experts, 2 * 256)),
            num_tokens_per_expert=i32(num_experts),
            cta_idx_xy_to_batch_idx=i32(max_ctas),
            cta_idx_xy_to_mn_limit=i32(max_ctas),
            num_non_exiting_ctas=i32(1),
            workspace=workspace,
        )

    def for_tokens(self, num_tokens: int) -> "FusedRoutingOutputs":
        top_k = self.expert_weights.shape[1]
        num_experts = self.num_tokens_per_expert.shape[0]
        max_ctas = max_num_ctas_in_batch_dim(
            num_tokens, top_k, num_experts, self.tile_n
        )
        if (
            num_tokens * top_k > self.expanded_idx_to_permuted_idx.shape[0]
            or max_ctas > self.cta_idx_xy_to_batch_idx.shape[0]
        ):
            raise ValueError(f"{num_tokens} tokens exceed the allocated capacity")
        return FusedRoutingOutputs(
            tile_n=self.tile_n,
            total_num_padded_tokens=self.total_num_padded_tokens,
            expanded_idx_to_permuted_idx=self.expanded_idx_to_permuted_idx[
                : num_tokens * top_k
            ],
            permuted_idx_to_token_idx=self.permuted_idx_to_token_idx[
                : max_ctas * self.tile_n + 1
            ],
            expert_weights=self.expert_weights.view(-1)[: num_tokens * top_k].view(
                num_tokens, top_k
            ),
            expert_count_histogram=self.expert_count_histogram,
            num_tokens_per_expert=self.num_tokens_per_expert,
            cta_idx_xy_to_batch_idx=self.cta_idx_xy_to_batch_idx[:max_ctas],
            cta_idx_xy_to_mn_limit=self.cta_idx_xy_to_mn_limit[:max_ctas],
            num_non_exiting_ctas=self.num_non_exiting_ctas,
            workspace=self.workspace,
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The nine metadata tensors in FlashInfer's native FFI order."""
        return (
            self.total_num_padded_tokens,
            self.expanded_idx_to_permuted_idx,
            self.permuted_idx_to_token_idx,
            self.expert_weights,
            self.expert_count_histogram,
            self.num_tokens_per_expert,
            self.cta_idx_xy_to_batch_idx,
            self.cta_idx_xy_to_mn_limit,
            self.num_non_exiting_ctas,
        )


def build_dir(arch: str) -> str:
    return os.path.join(envs.VLLM_CACHE_ROOT, "fused_router_routing", f"sm{arch}")


def load():
    """Build (or load the cached build of) the extension for the current device."""
    if _ext:
        return _ext[0]
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    build = build_dir(arch)
    os.makedirs(build, exist_ok=True)
    digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
    src = os.path.join(build, f"fused_router_routing_{digest}.cu")
    if not os.path.exists(src):
        tmp = f"{src}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(_SOURCE)
        os.replace(tmp, src)
    orig = cpp._get_cuda_arch_flags
    cpp._get_cuda_arch_flags = lambda cflags=None: [
        f"-gencode=arch=compute_{arch},code=sm_{arch}"
    ]
    try:
        ext = cpp.load(
            name=f"_fused_router_routing_{digest}",
            sources=[src],
            # No fast-math, no FMA contraction: the routing math is explicit PTX
            # mirroring FlashInfer's fast-math build.
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", "-fmad=false"],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=build,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    _ext.append(ext)
    return ext


def fused_router_routing(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: FusedRoutingOutputs,
    *,
    top_k: int,
    tile_n: int,
    renormalize_mode: int,
    use_pdl: bool,
    logits_out: torch.Tensor | None = None,
    tiles_per_cta: int | None = None,
    early_trigger: bool | None = None,
    timing_out: torch.Tensor | None = None,
) -> None:
    """Router logits ``bf16(x @ weight.T)`` + top-K softmax routing + metadata.

    ``x``: bf16 [M, 2048] with unit inner stride, 1 <= M <= 64; ``weight``: bf16
    [256, 2048] contiguous (the router gate weight); ``out``: a
    ``FusedRoutingOutputs`` with capacity for M tokens and ``out.tile_n == tile_n``
    (``out.for_tokens(M)`` views are fine); ``renormalize_mode``: FlashInfer
    RoutingMethodType value (Renormalize or RenormalizeNaive); ``logits_out``:
    optional bf16 [M, 256] contiguous that receives the router logits.
    ``tiles_per_cta`` (1, 2 or 4 eight-token tiles per CTA) overrides the default
    launch shape; ``early_trigger`` (PDL only) overrides when dependents are
    triggered: when the routing CTA starts (True) or after its last write (False);
    ``timing_out`` (int64, >= 8 per CTA) receives per-CTA
    %globaltimer phase stamps (profiling only).
    """
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    if not is_supported(
        num_tokens,
        weight.shape[0],
        top_k,
        x.shape[1],
        renormalize_mode,
        tile_n,
    ):
        raise ValueError(
            f"fused_router_routing: unsupported call M={num_tokens} "
            f"E={weight.shape[0]} K={x.shape[1]} top_k={top_k} "
            f"routing_method={renormalize_mode} tile_n={tile_n}"
        )
    if out.tile_n != tile_n or out.expert_weights.shape[1] != top_k:
        raise ValueError("FusedRoutingOutputs tile_n/top_k mismatch")
    if logits_out is None:
        logits = (
            out.workspace[_WORKSPACE_HEADER_INTS:]
            .view(torch.bfloat16)[: num_tokens * NUM_EXPERTS]
            .view(num_tokens, NUM_EXPERTS)
        )
    else:
        logits = logits_out
    ext = _ext[0] if _ext else load()
    ext.run(
        x,
        weight,
        logits,
        out.workspace,
        out.total_num_padded_tokens,
        out.expanded_idx_to_permuted_idx,
        out.permuted_idx_to_token_idx,
        out.expert_weights,
        out.num_tokens_per_expert,
        out.cta_idx_xy_to_batch_idx,
        out.cta_idx_xy_to_mn_limit,
        out.num_non_exiting_ctas,
        top_k,
        tile_n,
        use_pdl,
        tiles_per_cta or _default_tiles_per_cta(num_tokens),
        EARLY_TRIGGER if early_trigger is None else early_trigger,
        _NO_TIMING if timing_out is None else timing_out,
    )


_NO_TIMING = torch.empty(0, dtype=torch.int64)
# Under PDL, trigger the dependents (FlashInfer's FC1) when the routing CTA starts
# (VR microbench: -1.9 to -2.2 us per call vs triggering after the last write).
EARLY_TRIGGER = True


def _default_tiles_per_cta(num_tokens: int) -> int:
    return 1


_SOURCE = r"""// Fused MoE router GEMV + top-K softmax routing + TRT-LLM routing metadata (E=256, K=2048, M<=64).
//
// Logits (bitwise mirror of FlashInfer tinygemm2_sm100, zero bias): output (token t, expert e) =
//   bf16_rn( add.ftz( ((c0 + c1) + c2) + c3 , 0.0f ) ),  c_w = fp32 chain of mma.sync.m16n8k16 bf16 (C starts at 0)
//   over the k16 slices k0 = (4*i + w)*256 + su*64 + kk*16, i = 0..K/1024-1, su = 0..3, kk = 0..3 in that order,
//   with the canonical m16n8k16 fragments (A = 16 expert rows, B = 8 tokens), all adds add.ftz.f32.
// Routing (FlashInfer 0.6.18.post1 routingCustom, NoOp preprocess + Softmax postprocess, built -use_fast_math):
//   key = (cub-twiddled bf16 bits << 16) | (65535 - e); top-K = K largest keys, descending (lower e wins ties);
//   m = top-1 score; p_k = ex2.approx.ftz(mul.ftz(sub.ftz(s_k, m), 0x3FB8AA3B)); sum = warp xor-butterfly
//   (masks 16, 8, 4, 2, 1; lanes >= K hold 0) with add.ftz; w_k = cvt.rn.bf16(div.approx.ftz(p_k, sum)).
//   Metadata as routingIndicesBlockKernel / routingIndicesDynBlockKernel: per-expert counts, CTA count
//   ceil(cnt / tile), exclusive scans in expert order, CTA tables, mn limits, padded size, permutation with
//   rows of an expert in token order. num_tokens_per_expert is written as well; the histogram scratch and
//   the padding rows of permuted_idx_to_token_idx are left untouched.
//
// Grid (16 expert tiles, ceil(M / (8 * TPC))), 512 threads. A CTA stages its 16 router rows (64 KB) and its
// 8*TPC token rows (32 KB per 8 tokens) in shared memory with cp.async (rows 128B-swizzled, ldmatrix
// conflict-free); the router rows are requested before griddepcontrol.wait. Warp (w, j) runs chain w of
// token tile j; the chains are combined in shared memory. The last CTA to arrive (arrival counter, reset by
// that CTA) routes all tokens with one warp per token, reusing the shared memory.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace rfr {

constexpr int kE = 256;        // experts (= router rows)
constexpr int kK = 2048;       // hidden size
constexpr int kThreads = 512;  // 16 warps
constexpr int kWarps = kThreads / 32;
constexpr int kMaxTokens = 64;
constexpr int kMaxTopK = 8;
constexpr int kSlices = kK / 64;  // k16 slices per chain
constexpr int kRowBytes = kK * 2;
constexpr int kChunks = kK / 8;              // 16-byte chunks per row
constexpr int kSmemW = 16 * kRowBytes;       // 16 router rows
template <int TPC>
constexpr int smem_bytes() {
  return kSmemW + TPC * 8 * kRowBytes;
}

struct Params {
  const __nv_bfloat16* x;
  int64_t sx;  // row stride of x (elements)
  const __nv_bfloat16* w;
  __nv_bfloat16* logits;  // [M, 256]
  int32_t* counter;
  int32_t* totalPadded;
  int32_t* expToPerm;
  int32_t* permToTok;
  __nv_bfloat16* weights;
  int32_t* tokPerExpert;
  int32_t* ctaBatch;
  int32_t* ctaMn;
  int32_t* numCtas;
  unsigned long long* timing;  // optional per-CTA phase timestamps (profiling), else nullptr
  int M, topK, tileLog2;
  bool usePdl;
  bool earlyTrigger;  // PDL: trigger dependents when the routing CTA starts (else after its last write)
};

__device__ __forceinline__ float add_ftz(float a, float b) {
  float r;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t redux_max(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
// cub::Traits<bf16>::TwiddleIn / TwiddleOut
__device__ __forceinline__ uint32_t twiddle_in(uint32_t b) { return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u); }
__device__ __forceinline__ uint32_t twiddle_out(uint32_t b) { return (b & 0x8000u) ? (b & 0x7fffu) : (~b & 0xffffu); }
__device__ __forceinline__ int slice_k0(int s, int w) {
  return ((s >> 4) * 4 + w) * 256 + ((s >> 2) & 3) * 64 + (s & 3) * 16;
}
__device__ __forceinline__ uint32_t swz(int row, int chunk) {  // byte offset of (row, 16B chunk)
  return static_cast<uint32_t>(row * kRowBytes + ((chunk ^ (row & 7)) << 4));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t srcBytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(srcBytes) : "memory");
}
__device__ __forceinline__ void stamp(const Params& p, int slot) {
  if (p.timing != nullptr && threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.timing[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + slot] = t;
  }
}

#define RFR_CAS(K, i, j)                    \
  {                                         \
    const uint32_t hi_ = max(K[i], K[j]);   \
    const uint32_t lo_ = min(K[i], K[j]);   \
    K[i] = hi_;                             \
    K[j] = lo_;                             \
  }

template <int TPC>
__global__ void __launch_bounds__(kThreads, 1) fused_router_routing_kernel(const Params p) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ int sLast;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int ftile = blockIdx.x;                // 16 experts
  const int tok0 = blockIdx.y * TPC * 8;       // first token of this CTA
  const uint32_t sW = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t sX = sW + kSmemW;
  stamp(p, 0);

  // 1. Router rows -> smem (constant data: requested before the PDL wait).
  {
    const char* src = reinterpret_cast<const char*>(p.w + static_cast<int64_t>(ftile) * 16 * kK);
#pragma unroll
    for (int i = 0; i < 16 * kChunks / kThreads; i++) {
      const int idx = i * kThreads + tid;
      const int r = idx / kChunks, c = idx % kChunks;
      cp_async16(sW + swz(r, c), src + static_cast<int64_t>(r) * kRowBytes + c * 16, 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  if (p.usePdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  // 2. Token rows -> smem (rows past M are zero-filled).
#pragma unroll
  for (int i = 0; i < TPC * 8 * kChunks / kThreads; i++) {
    const int idx = i * kThreads + tid;
    const int r = idx / kChunks, c = idx % kChunks;
    const int tok = tok0 + r;
    const char* src = reinterpret_cast<const char*>(p.x + static_cast<int64_t>(tok < p.M ? tok : 0) * p.sx) + c * 16;
    cp_async16(sX + swz(r, c), src, tok < p.M ? 16u : 0u);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  stamp(p, 1);

  // 3. Chain w of token tile j (warp = 4 j + w).
  const int w = warp & 3;
  const int j = warp >> 2;
  const bool active = j < TPC && tok0 + j * 8 < p.M;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if (active) {
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int achunk = lane >> 4;
    const int xrow = j * 8 + (lane & 7);
    const int xchunk = lane >> 3;  // x4: slice s (k 0-7, 8-15), slice s+1 (k 0-7, 8-15)
#pragma unroll
    for (int s = 0; s < kSlices; s += 2) {
      const int c0 = slice_k0(s, w) >> 3;
      uint32_t b[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                   : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                   : "r"(sX + swz(xrow, c0 + xchunk))
                   : "memory");
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t a0, a1, a2, a3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "r"(sW + swz(arow, c0 + 2 * h + achunk))
                     : "memory");
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[2 * h]), "r"(b[2 * h + 1]));
      }
    }
  }
  __syncthreads();  // shared memory is reused below
  float4* red = reinterpret_cast<float4*>(smem + kSmemW);
  if (active && w != 0) red[warp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  if (active && w == 0) {
    float o[4] = {acc[0], acc[1], acc[2], acc[3]};
#pragma unroll
    for (int c = 1; c < 4; c++) {
      const float4 v = red[(warp + c) * 32 + lane];
      o[0] = add_ftz(o[0], v.x);
      o[1] = add_ftz(o[1], v.y);
      o[2] = add_ftz(o[2], v.z);
      o[3] = add_ftz(o[3], v.w);
    }
    const int e = ftile * 16 + (lane >> 2);
    const int t = tok0 + j * 8 + (lane & 3) * 2;
#pragma unroll
    for (int z = 0; z < 4; z++) {
      const int tz = t + (z & 1);
      const int ez = e + (z >> 1) * 8;
      if (tz < p.M) p.logits[tz * kE + ez] = __float2bfloat16_rn(add_ftz(o[z], 0.0f));
    }
  }
  stamp(p, 2);

  // 4. Arrival (CTA barrier, then one acq_rel atomic, as CUTLASS's semaphore): the last CTA routes
  //    every token and resets the counter for the next launch.
  __syncthreads();
  if (tid == 0) {
    int prev;
    asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], 1;" : "=r"(prev) : "l"(p.counter) : "memory");
    sLast = prev == static_cast<int>(gridDim.x * gridDim.y) - 1;
    if (sLast) *p.counter = 0;
  }
  __syncthreads();
  stamp(p, 3);
  if (!sLast) return;
  // Dependents (FC1) wait for this grid's completion before reading routing; triggering here only lets
  // them launch and run their pre-wait prologue during the routing tail.
  if (p.usePdl && p.earlyTrigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint32_t* sMask = reinterpret_cast<uint32_t*>(smem);  // [2][256] token bitmask per expert (tokens 0-31, 32-63)
  int* sTopE = reinterpret_cast<int*>(smem + 2048);                        // [M * topK] expert per slot
  int* sOffP = reinterpret_cast<int*>(smem + 4096);                        // [256] first row per expert
  int* sWarpTot = reinterpret_cast<int*>(smem + 5120);                     // [16]
  // 5. Top-K + softmax, one warp per token (a warp's up to 4 tokens interleaved for ILP); lane holds
  //    experts 8*lane .. 8*lane+7 as sorted packed keys.
  constexpr int kTokPerWarp = kMaxTokens / kWarps;
  const int nq = (p.M - warp + kWarps - 1) / kWarps;  // tokens of this warp: warp + q * kWarps, q < nq
  uint32_t key[kTokPerWarp][8];
#pragma unroll
  for (int q = 0; q < kTokPerWarp; q++) {
    uint32_t v[4] = {0u, 0u, 0u, 0u};
    if (q < nq) {
      asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                   : "l"(reinterpret_cast<const char*>(p.logits + (warp + q * kWarps) * kE) + lane * 16));
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t bits = (v[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
      key[q][i] = (twiddle_in(bits) << 16) | static_cast<uint32_t>(65535 - (lane * 8 + i));
    }
  }
  sMask[tid] = 0u;  // kThreads == 2 * kE words
  __syncthreads();

  const int topK = p.topK;
#pragma unroll
  for (int q = 0; q < kTokPerWarp; q++) {
    if (q < nq) {
      RFR_CAS(key[q], 0, 2) RFR_CAS(key[q], 1, 3) RFR_CAS(key[q], 4, 6) RFR_CAS(key[q], 5, 7)
      RFR_CAS(key[q], 0, 4) RFR_CAS(key[q], 1, 5) RFR_CAS(key[q], 2, 6) RFR_CAS(key[q], 3, 7)
      RFR_CAS(key[q], 0, 1) RFR_CAS(key[q], 2, 3) RFR_CAS(key[q], 4, 5) RFR_CAS(key[q], 6, 7)
      RFR_CAS(key[q], 2, 4) RFR_CAS(key[q], 3, 5)
      RFR_CAS(key[q], 1, 4) RFR_CAS(key[q], 3, 6)
      RFR_CAS(key[q], 1, 2) RFR_CAS(key[q], 3, 4) RFR_CAS(key[q], 5, 6)
    }
  }
  stamp(p, 7);
  uint32_t mine[kTokPerWarp] = {};
  uint32_t top1[kTokPerWarp] = {};
#pragma unroll
  for (int kk = 0; kk < kMaxTopK; kk++) {
    if (kk < topK) {
#pragma unroll
      for (int q = 0; q < kTokPerWarp; q++) {
        if (q < nq) {
          const uint32_t m = redux_max(key[q][0]);
          if (key[q][0] == m) {
#pragma unroll
            for (int i = 0; i < 7; i++) key[q][i] = key[q][i + 1];
            key[q][7] = 0u;
          }
          if (lane == kk) mine[q] = m;
          if (kk == 0) top1[q] = m;
        }
      }
    }
  }
  const bool sel = lane < topK;
#pragma unroll
  for (int q = 0; q < kTokPerWarp; q++) {
    if (q >= nq) break;
    const int t = warp + q * kWarps;
    const uint32_t mk = sel ? mine[q] : 0u;
    const int e = 65535 - static_cast<int>(mk & 0xffffu);
    const float s = __uint_as_float(twiddle_out(mk >> 16) << 16);
    // FlashInfer's max over the K scores = the top-1 score (a +-0 sign difference cannot change exp).
    const float mx = __uint_as_float(twiddle_out(top1[q] >> 16) << 16);
    float pe = 0.f;
    if (sel) {
      float d, y;
      asm("sub.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(s), "f"(mx));
      asm("mul.ftz.f32 %0, %1, 0f3FB8AA3B;" : "=f"(y) : "f"(d));
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(pe) : "f"(y));
    }
    // FlashInfer's butterfly (masks 16, 8, 4, 2, 1) over lanes holding 0 beyond K <= 8: the 16 and 8
    // steps add exact zeros on lanes 0-7, so lanes 0-7 get the same sum from the 4, 2, 1 steps.
    float sum = pe;
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) sum = add_ftz(sum, __shfl_xor_sync(0xffffffffu, sum, mask));
    if (sel) {
      float wq;
      asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(wq) : "f"(pe), "f"(sum));
      p.weights[t * topK + lane] = __float2bfloat16_rn(wq);
      sTopE[t * topK + lane] = e;
      atomicOr(&sMask[(t >> 5) * kE + e], 1u << (t & 31));
    }
  }
  __syncthreads();
  stamp(p, 4);

  // 6. Per-expert counts, CTA counts, exclusive scan in expert order, CTA tables.
  const int tileLog2 = p.tileLog2;
  int cnt = 0, nCta = 0;
  if (tid < kE) {
    cnt = __popc(sMask[tid]) + __popc(sMask[kE + tid]);
    nCta = (cnt + (1 << tileLog2) - 1) >> tileLog2;
  }
  int inc = nCta;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, inc, d);
    if (lane >= d) inc += n;
  }
  if (lane == 31) sWarpTot[warp] = inc;
  __syncthreads();
  if (tid < kE) {
    int pre = 0, total = 0;
#pragma unroll
    for (int i = 0; i < kE / 32; i++) {
      const int wt = sWarpTot[i];
      pre += i < warp ? wt : 0;
      total += wt;
    }
    const int off = pre + inc - nCta;
    p.tokPerExpert[tid] = cnt;
    for (int c = 0; c < nCta; c++) {
      p.ctaBatch[off + c] = tid;
      p.ctaMn[off + c] = min((off + c + 1) << tileLog2, (off << tileLog2) + cnt);
    }
    sOffP[tid] = off << tileLog2;
    if (tid == 0) {
      p.totalPadded[0] = total << tileLog2;
      p.numCtas[0] = total;
    }
  }
  __syncthreads();
  stamp(p, 5);

  // 7. Permutation: rows of an expert in token order.
  for (int jj = tid; jj < p.M * topK; jj += kThreads) {
    const int t = jj / topK;
    const int e = sTopE[jj];
    const uint32_t lo = sMask[e], hi = sMask[kE + e];
    const int rank = t < 32 ? __popc(lo & ((1u << t) - 1u)) : __popc(lo) + __popc(hi & ((1u << (t - 32)) - 1u));
    const int pi = sOffP[e] + rank;
    p.expToPerm[jj] = pi;
    p.permToTok[pi] = t;
  }
  __syncthreads();
  stamp(p, 6);
  if (p.usePdl && !p.earlyTrigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

static int64_t max_ctas(int64_t m, int64_t k, int64_t e, int64_t tile) {
  int64_t rem = m * k;
  const int64_t filled = std::min(e, rem);
  rem -= filled;
  return filled + (rem > 0 ? rem / tile : 0);
}

template <int TPC>
static void launch(const Params& p, int device, cudaStream_t stream) {
  static bool attrSet[64] = {};
  if (!attrSet[device]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fused_router_routing_kernel<TPC>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_bytes<TPC>()));
    attrSet[device] = true;
  }
  const int tiles = (p.M + 7) / 8;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kE / 16, (tiles + TPC - 1) / TPC);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem_bytes<TPC>();
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = p.usePdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fused_router_routing_kernel<TPC>, p));
}

void run(torch::Tensor x, torch::Tensor w, torch::Tensor logits, torch::Tensor workspace, torch::Tensor totalPadded,
         torch::Tensor expToPerm, torch::Tensor permToTok, torch::Tensor weights, torch::Tensor tokPerExpert,
         torch::Tensor ctaBatch, torch::Tensor ctaMn, torch::Tensor numCtas, int64_t topK, int64_t tileN, bool usePdl,
         int64_t tilesPerCta, bool earlyTrigger, torch::Tensor timing) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
              logits.scalar_type() == at::kBFloat16 && weights.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kK && x.stride(1) == 1 && x.stride(0) % 8 == 0, "x must be [M, 2048], 16B rows");
  TORCH_CHECK(M >= 1 && M <= kMaxTokens, "1 <= M <= 64");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0);
  TORCH_CHECK(w.dim() == 2 && w.size(0) == kE && w.size(1) == kK && w.is_contiguous(), "weight must be [256, 2048]");
  TORCH_CHECK(logits.is_contiguous() && logits.numel() >= M * kE && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0);
  TORCH_CHECK(topK >= 1 && topK <= kMaxTopK);
  TORCH_CHECK(tileN > 0 && (tileN & (tileN - 1)) == 0, "tile_n must be a power of two");
  for (auto* t : {&workspace, &totalPadded, &expToPerm, &permToTok, &tokPerExpert, &ctaBatch, &ctaMn, &numCtas})
    TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous());
  const int64_t mc = max_ctas(M, topK, kE, tileN);
  TORCH_CHECK(expToPerm.numel() >= M * topK && weights.is_contiguous() && weights.numel() >= M * topK);
  TORCH_CHECK(permToTok.numel() >= mc * tileN && ctaBatch.numel() >= mc && ctaMn.numel() >= mc);
  TORCH_CHECK(tokPerExpert.numel() >= kE && totalPadded.numel() >= 1 && numCtas.numel() >= 1 && workspace.numel() >= 1);
  Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.sx = x.stride(0);
  p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.logits = reinterpret_cast<__nv_bfloat16*>(logits.data_ptr());
  p.counter = workspace.data_ptr<int32_t>();
  p.totalPadded = totalPadded.data_ptr<int32_t>();
  p.expToPerm = expToPerm.data_ptr<int32_t>();
  p.permToTok = permToTok.data_ptr<int32_t>();
  p.weights = reinterpret_cast<__nv_bfloat16*>(weights.data_ptr());
  p.tokPerExpert = tokPerExpert.data_ptr<int32_t>();
  p.ctaBatch = ctaBatch.data_ptr<int32_t>();
  p.ctaMn = ctaMn.data_ptr<int32_t>();
  p.numCtas = numCtas.data_ptr<int32_t>();
  const int64_t grid = 16 * ((M + 8 * tilesPerCta - 1) / (8 * tilesPerCta));
  TORCH_CHECK(timing.numel() == 0 || (timing.scalar_type() == at::kLong && timing.numel() >= grid * 8));
  p.timing = timing.numel() ? reinterpret_cast<unsigned long long*>(timing.data_ptr()) : nullptr;
  p.M = static_cast<int>(M);
  p.topK = static_cast<int>(topK);
  p.tileLog2 = 0;
  while ((int64_t{1} << p.tileLog2) < tileN) p.tileLog2++;
  p.usePdl = usePdl;
  p.earlyTrigger = earlyTrigger;

  const c10::cuda::CUDAGuard guard(x.device());
  const int dev = x.device().index();
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  switch (tilesPerCta) {
    case 1: launch<1>(p, dev, stream); break;
    case 2: launch<2>(p, dev, stream); break;
    case 4: launch<4>(p, dev, stream); break;
    default: TORCH_CHECK(false, "tiles_per_cta must be 1, 2 or 4");
  }
}

}  // namespace rfr

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &rfr::run); }
"""
