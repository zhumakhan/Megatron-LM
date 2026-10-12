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


def reset_first_use_alignment() -> None:
    """Forget the aligned call sites: a new Buffer or new pools compile their kernels anew."""
    _FIRST_USE_ALIGNED.clear()


class MoonEPDispatch(torch.autograd.Function):
    """Dispatch forward / combine backward, plus the expert weights' gradients.

    Forward scatters ``[s, H]`` tokens and their ``[s, K]`` fp32 routing weights into the
    expert-grouped ``[plan.nvs_s, ...]`` layout. Backward combines the slot gradients back to
    ``[s, H]`` and ``[s, K]``. With ``wgrad`` it then completes the experts' weight gradients:
    the grouped GEMMs' backward (autograd descendants of this dispatch, so earlier on the same
    stream) wrote the layer's dW into the shared fp32 pools; ``Buffer.reduce_grad`` sums the
    peers' slot dW of this rank's experts into its rows, and the sums go to the parameters.

    The expert weights (``params``: every expert's fc1, then every expert's fc2) are inputs here
    only so that their gradients reach autograd from this node, after ``reduce_grad``: DDP and
    Megatron-FSDP start reducing a parameter's gradient as soon as its AccumulateGrad runs. The
    forward never reads them (the dispatcher stages them into the pools afterwards). They also
    keep the dispatched tokens differentiable when neither the input nor the router needs grad.

    With a ``plan`` (the recompute of a checkpointed forward), the dispatch reuses that plan:
    no planning, and ``topk_experts``/``tokens_per_expert`` are ignored. Pass ``plan=None`` for
    a fresh dispatch. The plan and ``cu_seqlens`` (``None`` on reuse) come back via ``holder``.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        route_weights,
        topk_experts,
        tokens_per_expert,
        buffer,
        pools,
        state,
        holder,
        plan,
        total_num_tokens,
        wgrad,
        *params,
    ):
        # pylint: disable=missing-function-docstring
        if plan is None:
            hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
                x.contiguous(),
                route_weights.contiguous(),
                topk_experts,
                tokens_per_expert, 
                total_num_tokens=total_num_tokens, 
                zero_copy=False,
            )
        else:
            hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
                x.contiguous(), route_weights.contiguous(), plan=plan, zero_copy=False
            )
        holder["plan"], holder["cu_seqlens"] = plan, cu_seqlens
        ctx.buffer, ctx.plan, ctx.pools, ctx.state = buffer, plan, pools, state
        # Parameter objects, not saved tensors: like Megatron's fused-wgrad linear layers, the
        # backward writes into their main_grad and never reads their values.
        ctx.params = params if wgrad else None
        ctx.num_params = len(params)
        return hidden_nvsh, route_weights_nvs

    @staticmethod
    def backward(ctx, grad_hidden, grad_route_weights):
        # pylint: disable=missing-function-docstring
        pools = ctx.pools
        align_ep_group_on_first_use(pools.ep_group, ("dispatch_backward", pools.ep_ranks))
        grad_x, grad_route_weights_sk, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden.contiguous(),
            route_weights_nvs=(
                None if grad_route_weights is None else grad_route_weights.contiguous()
            ),
        )
        if ctx.params is None:
            param_grads = (None,) * ctx.num_params
        else:
            num_fc1 = ctx.num_params // 2
            param_grads = pools.reduce_and_copy_out(
                ctx.plan, ctx.params[:num_fc1], ctx.params[num_fc1:]
            )
        ctx.state.fwd_outstanding = False
        return (grad_x, grad_route_weights_sk) + (None,) * 9 + tuple(param_grads)


class MoonEPCombine(torch.autograd.Function):
    """Combine forward / plan-reuse dispatch backward.

    Forward gathers every token's K expert outputs back to its source rank and sums them into
    ``[s, H]`` (fp32 accumulation, bf16 output). It does not apply routing weights; the experts
    scale their rows before fc2. Backward re-dispatches the output gradient with the saved plan
    and then re-stages the layer's expert weights in the shared pools (other layers used them
    since this layer's forward), so the experts' GEMM backward finds them.

    ``restage`` is ``(pools, state, fc1, fc2, wgrad)`` from the layer's dispatch.
    """

    @staticmethod
    def forward(ctx, expert_output, buffer, plan, group, group_ranks, restage):
        # pylint: disable=missing-function-docstring
        assert expert_output.shape[0] == plan.nvs_s, (
            f"MoonEP combine expects the [plan.nvs_s={plan.nvs_s}, H] layout dispatch returned, "
            f"got {tuple(expert_output.shape)}"
        )
        output, _, _ = buffer.combine(plan=plan, hidden_nvsh=expert_output.contiguous())
        ctx.buffer, ctx.plan, ctx.group, ctx.group_ranks = buffer, plan, group, group_ranks
        ctx.restage = restage
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        align_ep_group_on_first_use(ctx.group, ("combine_backward", ctx.group_ranks))
        grad_expert_output, _, _, _ = ctx.buffer.dispatch(grad_output.contiguous(), plan=ctx.plan)
        pools, state, fc1, fc2, wgrad = ctx.restage
        if wgrad:
            pools.begin_grads(state)
        pools.stage(state, ctx.plan, fc1, fc2)
        return grad_expert_output, None, None, None, None, None


class GroupedMMLiveGrad(torch.autograd.Function):
    """``torch._grouped_mm`` over a ``[2*epn, out, in]`` pool whose dW goes straight to fp32.

    The pool rows are this rank's experts and then its prefetch slots, the groups of MoonEP's
    ``cu_seqlens[2*epn]``, so there are no dead groups. A group without tokens this step (an
    own expert nobody routed to, an unused slot) costs no FLOPs and its dW is exactly zero.
    The bf16 dW of all groups overwrites the fp32 target (own experts' rows, then this rank's
    reduce slots), so the pool never gets a weight gradient of its own; with ``fp32_target``
    ``None`` (frozen experts) dW is not computed.
    """

    @staticmethod
    def forward(ctx, x, weight, offs, fp32_target):
        # pylint: disable=missing-function-docstring
        # The pool is not saved for backward: it is shared by every layer and restaged before
        # this backward runs, and it carries no autograd grad, so version tracking would only
        # raise false positives.
        ctx.save_for_backward(x)
        ctx.weight, ctx.offs, ctx.fp32_target = weight, offs, fp32_target
        return torch._grouped_mm(x, weight.transpose(-2, -1), offs=offs)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        (x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = torch._grouped_mm(grad_output, ctx.weight, offs=ctx.offs)
        if ctx.fp32_target is not None:
            grad_weight = torch._grouped_mm(grad_output.t(), x, offs=ctx.offs)
            with torch.no_grad():
                ctx.fp32_target.copy_(grad_weight)
        return grad_x, None, None, None
