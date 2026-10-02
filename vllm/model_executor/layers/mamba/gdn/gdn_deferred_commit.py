# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the GDN spec-decode deferred single-state commit.

A deferred (D) spec-decode step runs the CUDA MTP kernel in its deferred mode
(``gdn_mtp_cuda.gdn_mtp_cuda_deferred``): per request it commits only the state
at the last accepted token of the previous step and leaves replay records of
this step's tokens in the request's last scratch slot, described by a per
(layer, request state, value head) header. Readers other than the D kernel need
the state at ``slots[num_accepted - 1]``: before the first non-D step after D
steps, the GDN metadata builder of each KV-cache group materializes it
(``gdn_mtp_cuda.gdn_mtp_materialize``) for the request states that ran a D step
since (``pending``). Headers of request states that are added or removed are
invalidated, batched into one fill per step, so a stale header never points at
freed blocks.

The decision itself lives in ``GDNAttentionMetadataBuilder`` (the
``spec_deferred`` metadata flag); this module owns the shared state: the header
tensor, the per-group pending sets and the per-step request rows.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops import gdn_mtp_cuda
from vllm.platforms import current_platform
from vllm.utils.torch_utils import async_tensor_h2d

logger = init_logger(__name__)

# Opt-in until the gates pass.
GDN_DEFERRED_COMMIT = os.environ.get("VLLM_GDN_DEFERRED_COMMIT", "0") == "1"
# Spec batches (padded size for FULL-graph decode buffers) of at least this
# many requests run deferred; smaller ones keep the snapshot kernels.
GDN_DEFERRED_COMMIT_MIN_REQS = int(
    os.environ.get("VLLM_GDN_DEFERRED_COMMIT_MIN_REQS", "16")
)
# The kernel's replay records hold at most this many tokens per request.
_MAX_DEFERRED_TOKENS = 8
_HEAD_DIM = 128
# Steps between evidence-counter log lines.
_LOG_EVERY = 4096


@dataclass
class GdnDeferredCommitGroup:
    """The deferred-commit state of one GDN metadata builder (KV-cache group):
    its layers and the request states that ran a D step since its last
    materialize (a superset is harmless: invalid headers are no-ops).
    """

    manager: "GdnDeferredCommit"
    layers: list
    pending: np.ndarray
    has_pending: bool = False
    # Evidence counters (logged by the first group every _LOG_EVERY steps):
    # deferred / other steps, materialize launches and request states.
    log: bool = False
    num_deferred: int = 0
    num_other: int = 0
    num_materialize: int = 0
    num_materialized_rows: int = 0

    def note_step(self, deferred: bool) -> None:
        if deferred:
            self.num_deferred += 1
        else:
            self.num_other += 1
        if self.log and (self.num_deferred + self.num_other) % _LOG_EVERY == 0:
            logger.info(
                "GDN deferred commit: %d deferred / %d other steps, %d "
                "materializes (%d request states)",
                self.num_deferred,
                self.num_other,
                self.num_materialize,
                self.num_materialized_rows,
            )

    @property
    def min_reqs(self) -> int:
        return self.manager.min_reqs

    def mark_pending(self, req_rows: np.ndarray) -> None:
        self.pending[req_rows] = True
        self.has_pending = True

    def materialize_pending(self) -> None:
        """Write ``slots[num_accepted - 1]`` of every pending request state in
        every layer of the group and invalidate the headers. Eager, before the
        forward; never inside a CUDA graph capture.
        """
        if not self.has_pending:
            return
        rows = np.flatnonzero(self.pending).astype(np.int32)
        self.pending[:] = False
        self.has_pending = False
        self._materialize(rows)

    def materialize_rows(self, req_rows: np.ndarray) -> None:
        """``materialize_pending`` restricted to the pending request states
        among ``req_rows``: those a deferred step reads through another path
        (its non-spec rows). Their pending bits are cleared.
        """
        if not self.has_pending or req_rows.size == 0:
            return
        rows = req_rows[self.pending[req_rows]].astype(np.int32)
        self.pending[rows] = False
        self._materialize(rows)

    def _materialize(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        self.num_materialize += 1
        self.num_materialized_rows += int(rows.size)
        manager = self.manager
        rows_gpu = async_tensor_h2d(rows, device=manager.device)
        for layer in self.layers:
            gdn_mtp_cuda.gdn_mtp_materialize(
                layer.kv_cache[1],
                layer._gdn_dc_hdr,
                rows_gpu,
                manager.num_accepted_tokens,
            )


@dataclass
class GdnDeferredStep:
    """One prepare_attn call's batch rows -> request state rows, shared by the
    GDN builders of the step (their spec rows all derive from the step's
    num_decode_draft_tokens_cpu).
    """

    idx_mapping: torch.Tensor  # [num_reqs] int64, device
    idx_mapping_np: np.ndarray  # [num_reqs]
    _rows_np: np.ndarray | None = None
    _rows_dev: torch.Tensor | None = None

    def spec_req_rows_np(self, spec_mask_cpu: torch.Tensor) -> np.ndarray:
        """int32 request state rows of the spec rows, in batch order."""
        if self._rows_np is None:
            batch_rows = np.flatnonzero(spec_mask_cpu.numpy())
            self._rows_np = self.idx_mapping_np[batch_rows].astype(np.int32)
        return self._rows_np

    def spec_req_rows(
        self, spec_mask_cpu: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Device copy of ``spec_req_rows_np`` (one upload per step)."""
        if self._rows_dev is None:
            self._rows_dev = async_tensor_h2d(
                self.spec_req_rows_np(spec_mask_cpu), device=device
            )
        return self._rows_dev


class GdnDeferredCommit:
    """Shared deferred-commit state of a model's Qwen GDN layers."""

    def __init__(
        self,
        layers: dict[str, torch.nn.Module],
        max_num_reqs: int,
        num_accepted_tokens: torch.Tensor,
        align: bool,
        min_reqs: int,
        device: torch.device,
    ) -> None:
        self.layers = layers
        self.max_num_reqs = max_num_reqs
        self.device = device
        # [max_num_reqs] accepted tokens of the last step per request state.
        self.num_accepted_tokens = num_accepted_tokens
        self.align = align
        # Align mode: the Mamba (state) block size of the GDN builders, i.e. the
        # block the align copies and prefix checkpoints work in (the kernel's
        # boundary window); set by bind_builders. 0 = no window (mode none).
        self.block_size = 0
        self.min_reqs = min_reqs
        # Bound by the model runner before any capture or step: the persistent
        # req_states.num_computed_tokens.gpu (FULL graphs bake its address).
        self.num_computed: torch.Tensor | None = None
        first = next(iter(layers.values()))
        num_v_heads = first.num_v_heads // first.tp_size
        self.hdr = torch.zeros(
            (
                len(layers),
                max_num_reqs,
                num_v_heads,
                gdn_mtp_cuda.GDN_DEFERRED_HDR_INTS,
            ),
            dtype=torch.int32,
            device=device,
        )
        for i, layer in enumerate(layers.values()):
            layer._gdn_dc_hdr = self.hdr[i]
            layer._gdn_dc = self
        self._groups: list[GdnDeferredCommitGroup] = []
        self._bound_builders: tuple[int, ...] | None = None
        # Holding the builders keeps their ids unique.
        self._builders: list = []
        self._invalidate_rows: list[int] = []

    @classmethod
    def maybe_create(
        cls,
        vllm_config: VllmConfig,
        max_num_reqs: int,
        num_accepted_tokens: torch.Tensor,
        device: torch.device,
    ) -> "GdnDeferredCommit | None":
        if not GDN_DEFERRED_COMMIT:
            return None
        reason = _unsupported_reason(vllm_config)
        layers: dict[str, torch.nn.Module] = {}
        if reason is None:
            layers, reason = _eligible_layers(vllm_config)
        if reason is not None:
            logger.info_once("GDN deferred commit is off: %s", reason)
            return None
        align = vllm_config.cache_config.mamba_cache_mode == "align"
        logger.info_once(
            "GDN deferred commit: %d layers, spec batches of >= %d requests, align %s",
            len(layers),
            GDN_DEFERRED_COMMIT_MIN_REQS,
            align,
        )
        return cls(
            layers,
            max_num_reqs,
            num_accepted_tokens,
            align,
            GDN_DEFERRED_COMMIT_MIN_REQS,
            device,
        )

    def bind_num_computed(self, num_computed: torch.Tensor) -> None:
        assert num_computed.dtype == torch.int32
        assert num_computed.numel() >= self.max_num_reqs
        self.num_computed = num_computed

    def kernel_ready(self) -> bool:
        return gdn_mtp_cuda.ready()

    def bind_builders(self, builders: list) -> None:
        """Bind a group to every GDN metadata builder of the model (idempotent
        for the same builders). All or nothing: if some builder has a layer
        that is not a deferred-commit layer, or (align) another Mamba block
        size, none defers, so the fused decode-metadata path stays uniform.
        In align mode the builders' Mamba block size sets ``block_size``.
        """
        key = tuple(id(builder) for builder in builders)
        if key == self._bound_builders:
            return
        # Builders are recreated only at initialization (e.g. after the CUDA
        # graph memory profile), before any step marked pending rows.
        assert not any(group.has_pending for group in self._groups)
        self._bound_builders = key
        self._builders = list(builders)
        self._groups = []
        reason = None
        block_sizes = {builder.kv_cache_spec.block_size for builder in builders}
        for builder in builders:
            names = builder.layer_names
            if not names or not all(name in self.layers for name in names):
                reason = f"GDN builder of {names} has non-deferred layers"
        if self.align and len(block_sizes) != 1:
            reason = f"GDN builders with different Mamba block sizes {block_sizes}"
        if reason is not None:
            logger.info_once("GDN deferred commit is off: %s", reason)
            return
        if self.align:
            self.block_size = next(iter(block_sizes))
            assert self.block_size > 0
            logger.info_once(
                "GDN deferred commit: align Mamba block size %d", self.block_size
            )
        for builder in builders:
            group = GdnDeferredCommitGroup(
                manager=self,
                layers=[self.layers[name] for name in builder.layer_names],
                pending=np.zeros(self.max_num_reqs, dtype=np.bool_),
            )
            group.log = not self._groups
            self._groups.append(group)
            builder.bind_deferred_commit(group)

    def invalidate(self, req_index: int) -> None:
        """Queue the headers of a request state that is added or removed."""
        self._invalidate_rows.append(req_index)
        for group in self._groups:
            group.pending[req_index] = False

    def flush_invalidations(self) -> None:
        """One indexed fill of the queued headers; before any build of a step."""
        if not self._invalidate_rows:
            return
        rows = async_tensor_h2d(
            np.asarray(self._invalidate_rows, dtype=np.int64), device=self.device
        )
        self._invalidate_rows.clear()
        self.hdr.index_fill_(1, rows, 0)

    def begin_step(
        self, idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray
    ) -> GdnDeferredStep:
        return GdnDeferredStep(idx_mapping, idx_mapping_np)


def _unsupported_reason(vllm_config: VllmConfig) -> str | None:
    spec = vllm_config.speculative_config
    num_spec = 0 if spec is None else (spec.num_speculative_tokens or 0)
    if num_spec < 1:
        return "no speculative decoding"
    if num_spec + 1 > _MAX_DEFERRED_TOKENS:
        return f"{num_spec} speculative tokens (records hold {_MAX_DEFERRED_TOKENS})"
    mode = vllm_config.cache_config.mamba_cache_mode
    if mode not in ("align", "none"):
        return f"mamba_cache_mode={mode}"
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        return "pipeline parallelism"
    if not current_platform.is_cuda():
        return "not a CUDA platform"
    return None


def _eligible_layers(
    vllm_config: VllmConfig,
) -> tuple[dict[str, torch.nn.Module], str | None]:
    """The Qwen GDN layers, or why they do not all take the fused-norm CUDA
    spec-decode path (the only one with a deferred mode).
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        FUSED_GDN_STATE_DTYPES,
        QwenGatedDeltaNetAttention,
    )

    context = vllm_config.compilation_config.static_forward_context
    layers = {
        name: layer
        for name, layer in context.items()
        if isinstance(layer, QwenGatedDeltaNetAttention)
    }
    if not layers:
        return {}, "no Qwen GDN layers"
    for name, layer in layers.items():
        if not layer.enable_fused_gdn_decode:
            return {}, f"{name} does not use the fused CUDA decode path"
        if layer.head_k_dim != _HEAD_DIM or layer.head_v_dim != _HEAD_DIM:
            return {}, f"{name} head dims are not {_HEAD_DIM}"
        if layer.get_state_dtype()[1] not in FUSED_GDN_STATE_DTYPES:
            return {}, f"{name} recurrent state dtype"
        if layer.norm.weight.dtype not in (torch.bfloat16, torch.float32):
            return {}, f"{name} norm weight dtype"
    return layers, None
