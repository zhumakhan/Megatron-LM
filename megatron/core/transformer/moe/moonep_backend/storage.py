# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The MoonEP pool set shared by every MoE layer of an EP group.

``moonep.ExpertPools`` gives, per projection ``n`` in gate/up/down (stored ``[out, in]`` like
Megatron's linear layers):

* a bf16 compute view ``[2*epn, out, in]``: rows ``[0, epn)`` stage this rank's experts, rows
  ``[epn, 2*epn)`` hold the copies of remote experts that peers push into its prefetch slots,
  in the row order of MoonEP's ``cu_seqlens[2*epn]``;
* an fp32 view of the same shape: rows ``[0, epn)`` collect this rank's experts' gradients, rows
  ``[epn, 2*epn)`` are its reduce slots, which the owners read in ``Buffer.reduce_grad``.

The expert weights themselves are ordinary parameters owned by DDP or Megatron-FSDP; the pools
only hold copies. A layer uses them from its dispatch to its combine and again in its backward:

* ``stage`` copies the layer's weights into the staging rows and runs the collective
  ``Buffer.prefetch_weight`` with the layer's plan, unless the pools already hold exactly that
  (layer, plan). The forward, every recompute and the backward of a layer each need the copies,
  and any other layer that ran in between overwrote them.
* The grouped-GEMM backward writes the layer's dW for all ``2*epn`` groups into the fp32 view;
  ``reduce_and_copy_out`` adds the peers' slot dW of this rank's experts into its rows and
  hands the sums to the parameters.

Every rank makes these calls in the same order with the same decisions (all of them host state),
so the barrier-bearing MoonEP calls match up across the EP group.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moonep_backend.autograd import align_ep_group_on_first_use

PROJECTIONS = ("gate", "up", "down")


def interleaved_backward_message(layer_number, other_layer_number) -> str:
    """Text of the error for two layers' gradients in the shared pools at once."""
    return (
        f"MoonEP: the backward of MoE layer {layer_number} started while the expert gradients "
        f"of layer {other_layer_number} were still waiting in the shared pools for their "
        "reduction. The pools hold one layer's gradients at a time, so MoE layer backwards must "
        "not interleave."
    )


def ungathered_fsdp_message(layer_number) -> str:
    """Text of the error for expert weights that no Megatron-FSDP unit hook gathered."""
    return (
        f"MoonEP: Megatron-FSDP never gathered the expert weights of MoE layer {layer_number} "
        "through an FSDP unit's hooks, so it is unknown whether their gradient buffer must be "
        "overwritten (ZeRO-2) or accumulated, and their values may be stale. The modules "
        "holding MoonEP's expert weights never run forward: keep every MoE layer inside an "
        "FSDP unit module (the default units do) and do not enable fine-grained parameter "
        "gathering."
    )


@dataclass
class MoonEPLayerState:
    """Per-layer MoonEP bookkeeping, shared by a layer's dispatcher and experts."""

    layer_number: Optional[int] = None
    # True between a gradient-tracking dispatch and its backward (phase-1 guard).
    fwd_outstanding: bool = False
    # Checkpointed first forwards of this layer whose recompute has not run yet. The
    # dispatcher replays a recorded plan only while exactly one is waiting.
    pending_checkpoints: int = 0
    # Tokens that recomputes of checkpointed forwards dispatched with a replayed plan since
    # the last check (finalize_model_grads), and how many of them the recompute routed to
    # another expert set than the forward that made the plan.
    recompute_tokens: int = 0
    recompute_mismatches: torch.Tensor = field(
        default_factory=lambda: torch.zeros(
            (), dtype=torch.int64, device=torch.cuda.current_device()
        )
    )


def _main_grad(param: torch.nn.Parameter) -> torch.Tensor:
    """The wrapper's gradient buffer of ``param`` (DDP's ``main_grad`` or Megatron-FSDP's)."""
    if getattr(param, "__fsdp_param__", False):
        # Megatron-FSDP hands out the buffer lazily (ZeRO-2: a temporary bucket per backward).
        param.main_grad = param.get_main_grad()
    return param.main_grad


def _dummy_grad(param: torch.nn.Parameter) -> torch.Tensor:
    """What the expert weights return to autograd after their grads went into main_grad.

    A fresh tensor: AccumulateGrad then adopts it as ``.grad`` without a copy, and the wrapper's
    hook drops it. The hook fires on the backward thread, as overlap_grad_reduce requires.
    """
    if getattr(param, "zero_out_wgrad", False):
        return torch.zeros_like(param)
    return torch.empty_like(param)


class MoonEPExpertPools:
    """The process-global MoonEP pools of one EP group and expert shape. Creation is collective.

    Args:
        buffer: the process-global ``moonep.Buffer``.
        ep_group: its expert-parallel group.
        hidden_size: H.
        ffn_hidden_size: expert FFN hidden size Hi.
    """

    def __init__(
        self, buffer, ep_group: dist.ProcessGroup, hidden_size: int, ffn_hidden_size: int
    ) -> None:
        from moonep import ExpertPools

        H, Hi = hidden_size, ffn_hidden_size
        self.buffer = buffer
        self.ep_group = ep_group
        self.ep_ranks = tuple(dist.get_process_group_ranks(ep_group))
        self.hidden_size, self.ffn_hidden_size = H, Hi
        shapes = {"gate": (Hi, H), "up": (Hi, H), "down": (H, Hi)}
        self.pools = ExpertPools(buffer, ep_group, shapes)
        self.epn = self.pools.epn
        # (layer state, plan) whose weight copies the bf16 pools hold, compared by identity.
        self.occupant: Optional[tuple] = None
        # Layer state whose dW sit in the fp32 pools between its GEMM backward and its reduce.
        self.grads_owner: Optional[MoonEPLayerState] = None

    # ---- views --------------------------------------------------------------------------------

    def weights(self, name: str) -> torch.Tensor:
        """bf16 ``[2*epn, out, in]`` compute view: staged own experts, then prefetch slots."""
        return self.pools.weights(name)

    def grads(self, name: str) -> torch.Tensor:
        """fp32 ``[2*epn, out, in]``: own experts' grads, then this rank's reduce slots."""
        return self.pools.grads(name)

    # ---- staging --------------------------------------------------------------------------------

    @torch.no_grad()
    def stage(
        self,
        state: MoonEPLayerState,
        plan: Any,
        fc1: Sequence[torch.nn.Parameter],
        fc2: Sequence[torch.nn.Parameter],
    ) -> bool:
        """Make the pools hold ``fc1``/``fc2`` copied with ``plan``; returns whether it pushed.

        Collective unless the pools already hold this (layer, plan): every rank keeps the same
        occupant history, so all of them skip or push together.
        """
        if (
            self.occupant is not None
            and self.occupant[0] is state
            and self.occupant[1] is plan
        ):
            return False
        # The copy below overwrites the staging rows, so forget the occupant first: if the
        # push fails, a later stage of the previous occupant must not be skipped.
        self.occupant = None
        Hi = self.ffn_hidden_size
        gate, up, down = (self.pools.staging(n) for n in PROJECTIONS)
        # fc1 holds gate rows, then up rows; the pools keep them as separate projections.
        torch._foreach_copy_(
            [*gate.unbind(0), *up.unbind(0), *down.unbind(0)],
            [*(w[:Hi] for w in fc1), *(w[Hi:] for w in fc1), *fc2],
        )
        # Every rank pushes its own experts into the slots of the ranks this plan copies them
        # to; the call ends in a cross-rank barrier. Peers overwrite those slots only in a later
        # push, which each rank issues after a barrier-bearing dispatch or combine that follows
        # its last read of the slots, so no read races a write.
        align_ep_group_on_first_use(self.ep_group, ("prefetch_weight", self.ep_ranks))
        self.pools.push(plan)
        self.occupant = (state, plan)
        return True

    # ---- gradients -------------------------------------------------------------------------------

    def begin_grads(self, state: MoonEPLayerState) -> None:
        """A layer's backward is about to write its dW into the fp32 pools."""
        owner = self.grads_owner
        if owner is not None and owner is not state:
            raise RuntimeError(interleaved_backward_message(state.layer_number, owner.layer_number))
        self.grads_owner = state

    def reduce_and_copy_out(
        self, plan: Any, fc1: Sequence[torch.nn.Parameter], fc2: Sequence[torch.nn.Parameter]
    ) -> List[torch.Tensor]:
        """Sum the peers' slot dW of this rank's experts into its own rows (``reduce_grad``),
        hand the sums to the parameters, and return what autograd gets for them.

        Under DDP and Megatron-FSDP (params carry ``grad_added_to_main_grad``) the fp32 sums go
        into ``main_grad`` like Megatron's fused-wgrad linear layers (overwriting it when the
        wrapper asks for that, ZeRO-2) and autograd gets throw-away dummies. Without a wrapper
        autograd gets the gradients themselves, in the parameters' dtype.
        """
        align_ep_group_on_first_use(self.ep_group, ("reduce_grad", self.ep_ranks))
        self.pools.reduce(plan)
        owner, self.grads_owner = self.grads_owner, None
        Hi = self.ffn_hidden_size
        gate, up, down = (self.pools.local_grads(n) for n in PROJECTIONS)
        params = [*fc1, *fc2]
        if not all(hasattr(p, "grad_added_to_main_grad") for p in params):
            return [torch.cat((gate[i], up[i])).to(w.dtype) for i, w in enumerate(fc1)] + [
                down[i].to(w.dtype, copy=True) for i, w in enumerate(fc2)
            ]
        # Megatron-FSDP marks every parameter __fsdp_param__ when it wraps the model, but sets
        # overwrite_main_grad only for parameters that an FSDP unit's hooks gathered. The
        # modules holding the expert weights never run forward, so without an enclosing unit
        # nothing says whether the buffer must be overwritten (ZeRO-2) or accumulated.
        # Every rank wraps the same model, so every rank raises here, after the reduce.
        if any(
            getattr(p, "__fsdp_param__", False) and not hasattr(p, "overwrite_main_grad")
            for p in params
        ):
            raise RuntimeError(ungathered_fsdp_message(getattr(owner, "layer_number", None)))
        # (destination, source) pairs, split by whether the wrapper accumulates into main_grad
        # (DDP, ZeRO-1) or wants it overwritten (ZeRO-2 hands out a fresh bucket per backward).
        pairs = {False: ([], []), True: ([], [])}
        for i, w in enumerate(fc1):
            main_grad = _main_grad(w)
            dst, src = pairs[bool(getattr(w, "overwrite_main_grad", False))]
            dst += [main_grad[:Hi], main_grad[Hi:]]
            src += [gate[i], up[i]]
        for i, w in enumerate(fc2):
            dst, src = pairs[bool(getattr(w, "overwrite_main_grad", False))]
            dst.append(_main_grad(w))
            src.append(down[i])
        with torch.no_grad():
            if pairs[False][0]:
                torch._foreach_add_(*pairs[False])
            if pairs[True][0]:
                torch._foreach_copy_(*pairs[True])
        dummies = []
        for w in params:
            w.grad_added_to_main_grad = True
            dummies.append(_dummy_grad(w))
        return dummies

    def end_step(self) -> None:
        """Forget the step's occupant and gradient owner (``finalize_moonep_step``): no plan
        of this step matches the next step's staging, and a backward that an exception cut
        short does not block the next step's."""
        self.occupant, self.grads_owner = None, None

    def destroy(self) -> None:
        """Unmap the pools. Collective over the EP group."""
        self.occupant, self.grads_owner = None, None
        self.pools.destroy()
