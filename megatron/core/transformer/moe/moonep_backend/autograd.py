# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Autograd functions of the MoonEP backend.

Dispatch and combine are transposes of each other, so each backward is the other call with the
forward's saved plan. Every call runs synchronously on the current stream: all calls on one
``moonep.Buffer`` must be serialized, and every EP rank must issue the same sequence of
barrier-bearing calls, which the per-layer, per-micro-batch autograd order guarantees.
"""

from typing import Hashable

import torch
import torch.distributed as dist

_FIRST_USE_ALIGNED: set = set()


def align_ep_group_on_first_use(group: dist.ProcessGroup, key: Hashable) -> None:
    """Host-align the EP group the first time a MoonEP call site runs with ``key``.

    MoonEP JIT-compiles its kernels on first use and its device barriers trap after a fixed
    timeout, so the ranks must enter a newly compiled call together rather than staggered by a
    slow host. Every rank reaches each call site in the same order, so the barrier matches.
    """
    if key in _FIRST_USE_ALIGNED:
        return
    torch.cuda.synchronize()
    dist.barrier(group=group)
    _FIRST_USE_ALIGNED.add(key)


class MoonEPDispatch(torch.autograd.Function):
    """Dispatch forward / combine backward, plus the per-micro-batch expert-grad reduction.

    Forward scatters ``[s, H]`` tokens and their ``[s, K]`` fp32 routing weights into the
    expert-grouped ``[plan.nvs_s, ...]`` layout. Backward combines the slot gradients back to
    ``[s, H]`` and ``[s, K]`` and then runs ``Buffer.reduce_grad``: the prefetched copies' dW,
    which the grouped GEMMs' backward already wrote into this rank's reduce chunk, are summed
    into their owners' gradient rows. Those GEMMs are autograd ancestors of this backward, so
    their writes are enqueued earlier on the same stream; peers are fenced by an inter-rank
    sync because ``reduce_grad`` has no entry barrier.
    """

    @staticmethod
    def forward(ctx, x, route_weights, topk_experts, tokens_per_expert, buffer, storage, holder):
        # pylint: disable=missing-function-docstring
        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
            x.contiguous(), route_weights.contiguous(), topk_experts, tokens_per_expert,
            zero_copy=False,
        )
        holder["plan"] = plan
        ctx.buffer, ctx.plan, ctx.storage = buffer, plan, storage
        ctx.mark_non_differentiable(cu_seqlens)
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    @staticmethod
    def backward(ctx, grad_hidden, grad_route_weights, _grad_cu_seqlens):
        # pylint: disable=missing-function-docstring
        from moonep.inter_rank_sync import launch_inter_rank_sync

        storage = ctx.storage
        align_ep_group_on_first_use(storage.ep_group, ("dispatch_backward", storage.ep_ranks))
        grad_x, grad_route_weights_sk, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden.contiguous(),
            route_weights_nvs=(
                None if grad_route_weights is None else grad_route_weights.contiguous()
            ),
        )
        launch_inter_rank_sync(ctx.buffer._require_ctx())
        ctx.buffer.reduce_grad(
            plan=ctx.plan,
            full_gate_grad=storage.g["gate"],
            full_up_grad=storage.g["up"],
            full_down_grad=storage.g["down"],
            gate_reduce_buffer=storage.rb["gate"],
            up_reduce_buffer=storage.rb["up"],
            down_reduce_buffer=storage.rb["down"],
        )
        storage.fwd_outstanding = False
        return grad_x, grad_route_weights_sk, None, None, None, None, None


class MoonEPCombine(torch.autograd.Function):
    """Combine forward / plan-reuse dispatch backward.

    Forward gathers every token's K expert outputs back to its source rank and sums them into
    ``[s, H]`` (fp32 accumulation, bf16 output). It does not apply routing weights; the experts
    scale their rows before fc2. Backward re-dispatches the output gradient with the saved plan.
    """

    @staticmethod
    def forward(ctx, expert_output, buffer, plan, group, group_ranks):
        # pylint: disable=missing-function-docstring
        assert expert_output.shape[0] == plan.nvs_s, (
            f"MoonEP combine expects the [plan.nvs_s={plan.nvs_s}, H] layout dispatch returned, "
            f"got {tuple(expert_output.shape)}"
        )
        output, _, _ = buffer.combine(plan=plan, hidden_nvsh=expert_output.contiguous())
        ctx.buffer, ctx.plan, ctx.group, ctx.group_ranks = buffer, plan, group, group_ranks
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        align_ep_group_on_first_use(ctx.group, ("combine_backward", ctx.group_ranks))
        grad_expert_output, _, _, _ = ctx.buffer.dispatch(grad_output.contiguous(), plan=ctx.plan)
        return grad_expert_output, None, None, None, None


class GroupedMMLiveGrad(torch.autograd.Function):
    """``torch._grouped_mm`` over the ``[E+B, out, in]`` pool whose dW goes straight to fp32.

    On this rank only the own experts ``[lo, hi)`` and the prefetch slots ``[E, E+B)`` hold
    tokens; every other group is an empty segment of ``offs``. dX runs over all groups (empty
    ones cost nothing), dW only over the live ones (``offs_live``), and the bf16 dW is added
    into the fp32 targets (own rows, then slots), so no pool-sized weight gradient exists.
    """

    @staticmethod
    def forward(ctx, x, weight, offs, offs_live, fp32_targets):
        # pylint: disable=missing-function-docstring
        # The pool is not saved for backward: prefetch_weight rewrites its slot rows in place
        # and it carries no autograd grad, so version tracking would only raise false positives.
        ctx.save_for_backward(x)
        ctx.weight, ctx.fp32_targets = weight, fp32_targets
        ctx.offs, ctx.offs_live = offs, offs_live
        return torch._grouped_mm(x, weight.transpose(-2, -1), offs=offs)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        (x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = torch._grouped_mm(grad_output, ctx.weight, offs=ctx.offs)
        grad_weight = torch._grouped_mm(grad_output.t(), x, offs=ctx.offs_live)
        with torch.no_grad():
            own, slots = ctx.fp32_targets
            own.add_(grad_weight[: own.shape[0]])
            slots.add_(grad_weight[own.shape[0] :])
        return grad_x, None, None, None, None
