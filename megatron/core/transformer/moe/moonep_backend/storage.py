# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-layer VMM expert pools of the MoonEP backend.

Layout (per projection ``n`` in gate/up/down; one EP rank ``r`` owns experts
``[lo, hi) = [r*epn, (r+1)*epn)``; ``B == epn`` prefetch slots):

* ``w[n]``: bf16 ``[E+B, out, in]`` compute weights. Rows ``[0, E)`` map every rank's
  own-expert chunk (only ``[lo, hi)`` is local memory), rows ``[E, E+B)`` are this rank's
  prefetch slots, which ``Buffer.prefetch_weight`` fills from peers over NVLink.
* ``g[n]``: fp32 ``[E+B, out, in]`` gradients in the same layout. Rows ``[lo, hi)`` are this
  rank's reduced expert gradients (the masters' ``.grad``); rows ``[E, E+B)`` are this rank's
  reduce chunk, i.e. the same memory as ``rb[n][r]``.
* ``rb[n]``: fp32 ``[R, B, out, in]`` view of every rank's reduce chunk, which
  ``Buffer.reduce_grad`` reads.

Weights are stored ``[out, in]`` like Megatron's linear layers, so checkpoints match
``SequentialMLP``/``TEGroupedMLP``; MoonEP's prefetch and reduce kernels do not care what the
two inner dims mean. A rank writes only its own rows and its slots; every other row is a live
mapping of peer memory.

This module uses private MoonEP helpers (``moonep.buffer._*`` and ``moonep._C``) because
MoonEP exposes no public expert-pool API yet.
"""

import os
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

PROJECTIONS = ("gate", "up", "down")


class MoonEPExpertStorage:
    """VMM expert weight and gradient pools of one MoE layer. Construction is collective.

    Args:
        ep_group: expert-parallel group; its ranks must be one NVLink domain.
        num_experts: total routed experts E.
        hidden_size: model hidden size H.
        ffn_hidden_size: expert FFN hidden size Hi.
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_size: int,
        ffn_hidden_size: int,
    ) -> None:
        from moonep._C import get_vmm_granularity
        from moonep.buffer import _use_fabric_for_group

        self.ep_group = ep_group
        self.ep_ranks = tuple(dist.get_process_group_ranks(ep_group))
        self.R = ep_group.size()
        self.rank = ep_group.rank()
        assert num_experts % self.R == 0, (
            f"MoonEP: num_moe_experts={num_experts} must be divisible by the EP size {self.R}."
        )
        self.E = num_experts
        self.epn = num_experts // self.R
        # B == E/R: equal chunk shapes for nvl_dist_map, a slot for every remote expert, and
        # empty groups outside [lo, hi) and [E, E+B) (MoonEP/README.md).
        self.B = self.epn
        self.lo, self.hi = self.rank * self.epn, (self.rank + 1) * self.epn
        self.layer_number = None
        # True between a gradient-tracking dispatch and its backward (phase-1 guard).
        self.fwd_outstanding = False

        H, Hi = hidden_size, ffn_hidden_size
        self.chunk_shapes: Dict[str, Tuple[int, int, int]] = {
            "gate": (self.epn, Hi, H),
            "up": (self.epn, Hi, H),
            "down": (self.epn, H, Hi),
        }
        granularity = int(get_vmm_granularity())
        for shape in self.chunk_shapes.values():
            nbytes = shape[0] * shape[1] * shape[2] * torch.bfloat16.itemsize
            assert nbytes % granularity == 0, (
                f"MoonEP: expert chunk {shape} bf16 = {nbytes} bytes must be a multiple of the "
                f"VMM granularity {granularity} bytes (moonep._C.get_vmm_granularity()); choose "
                "num_moe_experts/EP, hidden_size and moe_ffn_hidden_size accordingly."
            )

        self._keepalives: List[torch.Tensor] = []
        # Collective (an all-gather) unless MOONEP_MEM_HANDLE_TYPE=fd; decided once per layer.
        self._use_fabric = bool(_use_fabric_for_group(ep_group))
        self.w: Dict[str, torch.Tensor] = {}
        self.g: Dict[str, torch.Tensor] = {}
        self.rb: Dict[str, torch.Tensor] = {}
        for name in PROJECTIONS:
            self.w[name] = self._composite_weight(self.chunk_shapes[name])
        for name in PROJECTIONS:
            self.g[name], self.rb[name] = self._grad_buffers(self.chunk_shapes[name])
        # cuMemCreate memory is not zero-filled and reduce_grad only clears the slots it
        # consumes, so the owned rows and the reduce chunk start at zero.
        self.zero_grads_()

    # ---- VMM mapping helpers (collective; every rank calls them in the same order) ----------

    def _alloc_chunk(self, chunk_shape, dtype) -> torch.Tensor:
        """Allocate one VMM chunk on this GPU; return its exported shareable."""
        from moonep._C import nvl_dist_alloc, nvl_release_mem_handle

        keepalive, shareable, owned = nvl_dist_alloc(
            shape=list(chunk_shape), dtype=dtype, use_fabric=self._use_fabric
        )
        self._keepalives.append(keepalive)
        nvl_release_mem_handle(owned)
        return shareable

    def _local_item(self, shareable):
        return shareable.cpu().view(-1) if self._use_fabric else int(shareable.item())

    def _gathered_items(self, shareable) -> list:
        """Every rank's item (fabric handle or dup'ed fd) for the chunk ``shareable`` names."""
        from moonep.buffer import _all_gather_shareables, _exchange_ipc_fds

        if self._use_fabric:
            handles = _all_gather_shareables(shareable, self.ep_group)
            return [handles[r] for r in range(self.R)]
        fds = _exchange_ipc_fds(
            int(shareable.item()), list(range(self.R)), self.rank, self.R, self.ep_group
        )
        return [fds[r] for r in range(self.R)]

    def _map_items(self, chunk_shape, dtype, items) -> torch.Tensor:
        """Map the chunks in ``items`` back to back into one contiguous virtual range."""
        from moonep._C import nvl_dist_map

        if self._use_fabric:
            shareables = torch.stack(items)
        else:
            shareables = torch.tensor(items, dtype=torch.int64)
        mapped = nvl_dist_map(
            chunk_shape=list(chunk_shape),
            dtype=dtype,
            shareables=shareables,
            local_rank=self.rank,
            world_size=len(items),
            use_fabric=self._use_fabric,
        )
        mapped._keepalive = self._keepalives
        return mapped

    def _close_items(self, items) -> None:
        if not self._use_fabric:
            for fd in items:
                os.close(fd)

    def _composite_weight(self, chunk_shape) -> torch.Tensor:
        """bf16 ``[E+B, ...]``: all R ranks' expert chunks, then this rank's prefetch chunk."""
        w_share = self._alloc_chunk(chunk_shape, torch.bfloat16)
        b_share = self._alloc_chunk(chunk_shape, torch.bfloat16)
        w_items = self._gathered_items(w_share)
        b_item = self._local_item(b_share)
        try:
            return self._map_items(chunk_shape, torch.bfloat16, w_items + [b_item])
        finally:
            self._close_items(w_items + [b_item, self._local_item(w_share)])

    def _grad_buffers(self, chunk_shape) -> Tuple[torch.Tensor, torch.Tensor]:
        """fp32 (grad ``[E+B, ...]``, reduce view ``[R, B, ...]``) for one projection."""
        g_share = self._alloc_chunk(chunk_shape, torch.float32)
        r_share = self._alloc_chunk(chunk_shape, torch.float32)
        g_items = self._gathered_items(g_share)
        r_items = self._gathered_items(r_share)
        r_local = self._local_item(r_share)
        try:
            grad = self._map_items(chunk_shape, torch.float32, g_items + [r_local])
            reduce = self._map_items(chunk_shape, torch.float32, r_items)
        finally:
            self._close_items(g_items + r_items + [r_local, self._local_item(g_share)])
        return grad, reduce.view(self.R, self.B, *chunk_shape[1:])

    # ---- accessors ----------------------------------------------------------------------------

    def fp32_targets(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Where the grouped-GEMM backward adds dW: (own rows ``[lo, hi)``, slots ``[E, E+B)``)."""
        return self.g[name][self.lo : self.hi], self.g[name][self.E :]

    @torch.no_grad()
    def zero_grads_(self) -> None:
        """Zero this rank's owned gradient rows and its reduce chunk (never peers' rows)."""
        for name in PROJECTIONS:
            self.g[name][self.lo : self.hi].zero_()
            self.g[name][self.E :].zero_()
