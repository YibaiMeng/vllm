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

The kernel: 16 feature tiles x ceil(M / 32) token groups of 512-thread CTAs. Each CTA
streams its 16 router rows into shared memory with ``cp.async`` *before*
``griddepcontrol.wait`` (the weight is constant), then loads its tokens, runs the
four mma chains per 8-token tile, combines them and stores bf16 logits. The last CTA
to arrive (self-resetting arrival counter in ``FusedRoutingOutputs.workspace``) does
the routing for all tokens. Under PDL every global write and every read of ``x`` is
after the wait, and dependents are triggered after the last write.

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
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "-fmad=false"],
            extra_cflags=["-O3", "-std=c++17"],
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
) -> None:
    """Router logits ``bf16(x @ weight.T)`` + top-K softmax routing + metadata.

    ``x``: bf16 [M, 2048] with unit inner stride, 1 <= M <= 64; ``weight``: bf16
    [256, 2048] contiguous (the router gate weight); ``out``: a
    ``FusedRoutingOutputs`` with capacity for M tokens and ``out.tile_n == tile_n``
    (``out.for_tokens(M)`` views are fine); ``renormalize_mode``: FlashInfer
    RoutingMethodType value (Renormalize or RenormalizeNaive); ``logits_out``:
    optional bf16 [M, 256] contiguous that receives the router logits.
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
    )


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
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace rfr {

constexpr int kE = 256;        // experts (= router rows)
constexpr int kK = 2048;       // hidden size
constexpr int kThreads = 512;  // 16 warps
constexpr int kWarps = kThreads / 32;
constexpr int kTilesPerCta = 4;  // 8-token tiles per CTA (4 chains x 4 tiles = 16 warps)
constexpr int kMaxTokens = 64;
constexpr int kMaxTopK = 8;
constexpr int kSlices = kK / 64;  // k16 slices per chain
constexpr int kRowBytes = kK * 2;
constexpr int kSmemW = 16 * kRowBytes;                  // 16 router rows
constexpr int kSmemRed = kWarps * 32 * 4 * 4;           // chain partials
constexpr int kSmemBytes = kSmemW + kSmemRed;

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
  int M, topK, tileLog2;
  bool usePdl;
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
__device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
// cub::Traits<bf16>::TwiddleIn / TwiddleOut
__device__ __forceinline__ uint32_t twiddle_in(uint32_t b) { return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u); }
__device__ __forceinline__ uint32_t twiddle_out(uint32_t b) { return (b & 0x8000u) ? (b & 0x7fffu) : (~b & 0xffffu); }

__device__ __forceinline__ int slice_k0(int s, int w) {
  return ((s >> 4) * 4 + w) * 256 + ((s >> 2) & 3) * 64 + (s & 3) * 16;
}

#define RFR_CAS(i, j)                       \
  {                                         \
    uint32_t hi_ = max(key[i], key[j]);     \
    uint32_t lo_ = min(key[i], key[j]);     \
    key[i] = hi_;                           \
    key[j] = lo_;                           \
  }

__global__ void __launch_bounds__(kThreads, 1) fused_router_routing_kernel(const Params p) {
  extern __shared__ __align__(128) char smem[];
  char* sW = smem;
  float4* sRed = reinterpret_cast<float4*>(smem + kSmemW);
  __shared__ int sLast;
  __shared__ unsigned long long sMask[kE];
  __shared__ int sTopE[kMaxTokens * kMaxTopK];
  __shared__ int sOffP[kE];
  __shared__ int sWarpTot[kWarps];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int ftile = blockIdx.x;  // 16 experts
  const uint32_t sWAddr = static_cast<uint32_t>(__cvta_generic_to_shared(sW));

  // 1. Router rows -> smem (constant data: allowed before the PDL wait). Row r, 16-byte chunk c at
  //    r * kRowBytes + (c ^ (r & 7)) * 16 (conflict-free ldmatrix).
  {
    const char* src = reinterpret_cast<const char*>(p.w + static_cast<int64_t>(ftile) * 16 * kK);
#pragma unroll
    for (int i = 0; i < (16 * kK / 8) / kThreads; i++) {
      const int idx = i * kThreads + tid;
      const int r = idx / (kK / 8);
      const int c = idx % (kK / 8);
      const uint32_t dst = sWAddr + r * kRowBytes + ((c ^ (r & 7)) << 4);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src + static_cast<int64_t>(r) * kRowBytes + c * 16)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  if (p.usePdl) asm volatile("griddepcontrol.wait;" ::: "memory");

  // 2. Four fp32 mma chains per 8-token tile (warp = chain w, tile j).
  const int w = warp & 3;
  const int tile = blockIdx.y * kTilesPerCta + (warp >> 2);
  const bool active = tile * 8 < p.M;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t bf[kSlices][2];
  if (active) {
    const int tok = tile * 8 + (lane >> 2);
    const int q2 = (lane & 3) * 2;
    if (tok < p.M) {
      const __nv_bfloat16* xr = p.x + static_cast<int64_t>(tok) * p.sx;
#pragma unroll
      for (int s = 0; s < kSlices; s++) {
        const int k0 = slice_k0(s, w);
        bf[s][0] = ld_cg_u32(xr + k0 + q2);
        bf[s][1] = ld_cg_u32(xr + k0 + 8 + q2);
      }
    } else {
#pragma unroll
      for (int s = 0; s < kSlices; s++) bf[s][0] = bf[s][1] = 0u;
    }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  if (active) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int chalf = lane >> 4;
    const uint32_t rowAddr = sWAddr + row * kRowBytes;
#pragma unroll
    for (int s = 0; s < kSlices; s++) {
      const int c = (slice_k0(s, w) >> 3) + chalf;
      uint32_t a0, a1, a2, a3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                   : "r"(rowAddr + ((c ^ (row & 7)) << 4))
                   : "memory");
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf[s][0]), "r"(bf[s][1]));
    }
    sRed[warp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
  __syncthreads();
  if (active && w == 0) {
    float o[4] = {acc[0], acc[1], acc[2], acc[3]};
#pragma unroll
    for (int c = 1; c < 4; c++) {
      const float4 v = sRed[(warp + c) * 32 + lane];
      o[0] = add_ftz(o[0], v.x);
      o[1] = add_ftz(o[1], v.y);
      o[2] = add_ftz(o[2], v.z);
      o[3] = add_ftz(o[3], v.w);
    }
    const int e = ftile * 16 + (lane >> 2);
    const int t = tile * 8 + (lane & 3) * 2;
#pragma unroll
    for (int z = 0; z < 4; z++) {
      const int tz = t + (z & 1);
      const int ez = e + (z >> 1) * 8;
      if (tz < p.M) p.logits[tz * kE + ez] = __float2bfloat16_rn(add_ftz(o[z], 0.0f));
    }
  }

  // 3. Arrival: the last CTA routes every token.
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(p.counter, 1);
    sLast = prev == static_cast<int>(gridDim.x * gridDim.y) - 1;
  }
  __syncthreads();
  if (!sLast) return;
  __threadfence();
  if (tid == 0) *p.counter = 0;  // self-reset for the next launch
  if (tid < kE) sMask[tid] = 0ull;
  __syncthreads();

  // 4. Top-K + softmax, one warp per token; lane holds experts 8*lane .. 8*lane+7.
  const int topK = p.topK;
  for (int t = warp; t < p.M; t += kWarps) {
    const char* row = reinterpret_cast<const char*>(p.logits + t * kE) + lane * 16;
    uint32_t v[4];
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                 : "l"(row));
    uint32_t key[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t bits = (v[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
      key[i] = (twiddle_in(bits) << 16) | static_cast<uint32_t>(65535 - (lane * 8 + i));
    }
    RFR_CAS(0, 2) RFR_CAS(1, 3) RFR_CAS(4, 6) RFR_CAS(5, 7)
    RFR_CAS(0, 4) RFR_CAS(1, 5) RFR_CAS(2, 6) RFR_CAS(3, 7)
    RFR_CAS(0, 1) RFR_CAS(2, 3) RFR_CAS(4, 5) RFR_CAS(6, 7)
    RFR_CAS(2, 4) RFR_CAS(3, 5)
    RFR_CAS(1, 4) RFR_CAS(3, 6)
    RFR_CAS(1, 2) RFR_CAS(3, 4) RFR_CAS(5, 6)
    uint32_t mine = 0u;
#pragma unroll
    for (int kk = 0; kk < kMaxTopK; kk++) {
      if (kk < topK) {
        const uint32_t m = redux_max(key[0]);
        if (key[0] == m) {
#pragma unroll
          for (int i = 0; i < 7; i++) key[i] = key[i + 1];
          key[7] = 0u;
        }
        if (lane == kk) mine = m;
      }
    }
    const bool sel = lane < topK;
    const int e = 65535 - static_cast<int>(mine & 0xffffu);
    const float s = __uint_as_float(twiddle_out(mine >> 16) << 16);
    const float mx = __shfl_sync(0xffffffffu, s, 0);
    float pe = 0.f;
    if (sel) {
      float d, y;
      asm("sub.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(s), "f"(mx));
      asm("mul.ftz.f32 %0, %1, 0f3FB8AA3B;" : "=f"(y) : "f"(d));
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(pe) : "f"(y));
    }
    float sum = pe;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum = add_ftz(sum, __shfl_xor_sync(0xffffffffu, sum, mask));
    if (sel) {
      float wq;
      asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(wq) : "f"(pe), "f"(sum));
      p.weights[t * topK + lane] = __float2bfloat16_rn(wq);
      sTopE[t * topK + lane] = e;
      atomicOr(&sMask[e], 1ull << t);
    }
  }
  __syncthreads();

  // 5. Per-expert counts, CTA counts, exclusive scan in expert order, CTA tables.
  const int tileLog2 = p.tileLog2;
  int cnt = 0, nCta = 0;
  if (tid < kE) {
    cnt = __popcll(sMask[tid]);
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

  // 6. Permutation: rows of an expert in token order.
  for (int j = tid; j < p.M * topK; j += kThreads) {
    const int t = j / topK;
    const int e = sTopE[j];
    const int pi = sOffP[e] + __popcll(sMask[e] & ((1ull << t) - 1ull));
    p.expToPerm[j] = pi;
    p.permToTok[pi] = t;
  }
  if (p.usePdl) {
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
}

static int64_t max_ctas(int64_t m, int64_t k, int64_t e, int64_t tile) {
  int64_t rem = m * k;
  const int64_t filled = std::min(e, rem);
  rem -= filled;
  return filled + (rem > 0 ? rem / tile : 0);
}

void run(torch::Tensor x, torch::Tensor w, torch::Tensor logits, torch::Tensor workspace, torch::Tensor totalPadded,
         torch::Tensor expToPerm, torch::Tensor permToTok, torch::Tensor weights, torch::Tensor tokPerExpert,
         torch::Tensor ctaBatch, torch::Tensor ctaMn, torch::Tensor numCtas, int64_t topK, int64_t tileN, bool usePdl) {
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
  p.M = static_cast<int>(M);
  p.topK = static_cast<int>(topK);
  p.tileLog2 = 0;
  while ((int64_t{1} << p.tileLog2) < tileN) p.tileLog2++;
  p.usePdl = usePdl;

  const c10::cuda::CUDAGuard guard(x.device());
  static bool attrSet[64] = {};
  const int dev = x.device().index();
  if (!attrSet[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fused_router_routing_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    attrSet[dev] = true;
  }
  const int tiles = static_cast<int>((M + 7) / 8);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kE / 16, (tiles + kTilesPerCta - 1) / kTilesPerCta);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = usePdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fused_router_routing_kernel, p));
}

}  // namespace rfr

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &rfr::run); }
"""
