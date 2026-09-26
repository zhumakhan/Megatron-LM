# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Once-per-step MoonEP expert-gradient finalization, called from ``finalize_model_grads``.

After the per-micro-batch ``Buffer.reduce_grad`` the owner rows hold exactly what the alltoall
backward would have accumulated on the owner. Scaling them by ``1/|dp_cp|`` and summing over
expert-DP then equals what DDP does for expert parameters in both of its modes (pre-scaled SUM,
or ``edp/dp_cp`` pre-scaled AVG over expert-DP); with per-token loss DDP's factor is 1.0 and the
``1/num_tokens`` scaling follows once ``num_tokens`` is reduced.
"""

from functools import partial
from typing import List, Optional

import torch

from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.moe.moonep_backend import iter_moonep_experts


def finalized_twice_message(layer_number) -> str:
    """Text of the error for a second finalization without zeroing in between."""
    return (
        f"MoonEP: expert gradients of MoE layer {layer_number} were finalized twice without "
        "zero_moe_grads() or optimizer.zero_grad() in between; finalize_model_grads must run "
        "once per iteration."
    )


def _check_grad(rsm, ddp_config, grad: torch.Tensor, where: str) -> None:
    """NaN/Inf (fatal) and unexpectedly-large (non-fatal) checks, mirroring DDP's check_grads."""
    norm = grad.norm(p=2)
    if ddp_config.check_for_nan_in_grad:
        rsm.validate_result(
            result=norm,
            rejection_func=torch.isnan,
            message=f"found NaN in {where}",
            tolerance=0.001,
            fatal=True,
        )
        rsm.validate_result(
            result=norm,
            rejection_func=torch.isinf,
            message=f"found Inf in {where}",
            tolerance=0.001,
            fatal=True,
        )
    if ddp_config.check_for_large_grads:
        # A separate context keeps this history apart from DDP's "grads" history.
        rsm.validate_result(
            result=norm,
            rejection_func=partial(rsm.is_unexpectedly_large, threshold=10, context="moonep_grads"),
            message=f"found unexpected large grads in {where}",
            tolerance=0.001,
            fatal=False,
        )


@torch.no_grad()
def reduce_moonep_expert_grads(
    model: List[torch.nn.Module],
    config,
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Scale MoonEP expert grads like DDP and sum them over expert-DP; marks them finalized.

    Args:
        model: model chunks (DDP-wrapped in training).
        config: transformer config (``calculate_per_token_loss`` is read).
        dp_cp_group: fallback data-parallel x context-parallel group for chunks that are not
            DDP-wrapped; DDP chunks use their own ``dp_cp_group`` like DDP's scaling does.
    """
    rsm = get_rerun_state_machine()
    for chunk in model:
        ddp_config = getattr(chunk, "ddp_config", None)
        chunk_dp_cp_group = getattr(chunk, "dp_cp_group", None) or dp_cp_group
        for experts in iter_moonep_experts(chunk):
            if experts.finalized:
                raise RuntimeError(finalized_twice_message(experts.layer_number))
            for name, grad in experts.owned_grads():
                if ddp_config is not None and (
                    ddp_config.check_for_nan_in_grad or ddp_config.check_for_large_grads
                ):
                    _check_grad(
                        rsm,
                        ddp_config,
                        grad,
                        f"MoonEP expert grad '{name}' of MoE layer {experts.layer_number} before "
                        "the expert-data-parallel reduction",
                    )
                if not config.calculate_per_token_loss and chunk_dp_cp_group is not None:
                    grad.mul_(1.0 / chunk_dp_cp_group.size())
                if experts.expt_dp_group.size() > 1:
                    torch.distributed.all_reduce(grad, group=experts.expt_dp_group)
            experts.finalized = True


@torch.no_grad()
def scale_moonep_expert_grads(model: List[torch.nn.Module], scaling: torch.Tensor) -> None:
    """Per-token loss: multiply MoonEP expert grads by ``scaling`` (``1/num_tokens``)."""
    for chunk in model:
        for experts in iter_moonep_experts(chunk):
            for _, grad in experts.owned_grads():
                grad.mul_(scaling)
