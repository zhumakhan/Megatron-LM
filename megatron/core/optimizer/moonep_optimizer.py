# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``MoonEPOptimizer``: steps the fp32 MoonEP expert masters and re-syncs their bf16 pools."""

from typing import Callable, List

import torch

from .optimizer import FP32Optimizer
from .optimizer_config import OptimizerConfig


def not_finalized_message(layer_number) -> str:
    """Text of the error for expert grads that skipped the MoonEP finalization."""
    return (
        f"MoonEP expert gradients of MoE layer {layer_number} reached the optimizer without "
        "reduce_moonep_expert_grads() since the last zero point (the 1/|dp_cp| scaling and "
        "expert-DP sum are missing). Use megatron.core.distributed.finalize_model_grads as "
        "config.finalize_model_grads_func, or call reduce_moonep_expert_grads() from your "
        "finalize hook."
    )


class MoonEPOptimizer(FP32Optimizer):
    """FP32 optimizer over the MoonEP expert masters, appended last to the ``ChainedOptimizer``.

    Its grad norm joins the chain's joint norm and single clip coefficient. Its gradients are
    persistent fp32 VMM storage aliased by the masters' ``.grad``, so ``zero_grad`` never
    deletes or zeroes them (``model.zero_moe_grads()`` zeroes them in place); it only records
    that the framework reached its zeroing point. After every step the bf16 compute pools are
    refreshed from the masters.

    Args:
        optimizer: the torch optimizer over the masters' param groups.
        config: the optimizer config shared with the rest of the chain.
        init_state_fn: optimizer-state initializer (used when loading checkpoints).
        moonep_experts: every ``MoonEPGroupedMLP`` of the model chunks.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable,
        moonep_experts: List[torch.nn.Module],
    ) -> None:
        super().__init__(optimizer, config, init_state_fn)
        self.moonep_experts = list(moonep_experts)
        for experts in self.moonep_experts:
            experts.raise_on_broken_invariants("at optimizer construction (after wrapping)")

    def zero_grad(self, set_to_none: bool = True):
        """Record the zero point; MoonEP grads are zeroed by ``model.zero_moe_grads()``."""
        del set_to_none  # the grads are persistent storage and are never set to None
        for experts in self.moonep_experts:
            experts.mark_zero_point_()

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Check the grads went through the MoonEP finalization, then mark them consumed."""
        for experts in self.moonep_experts:
            experts.repair_grad_aliases_("optimizer.prepare_grads")
            if not experts.finalized:
                raise RuntimeError(not_finalized_message(experts.layer_number))
            experts.mark_grads_consumed_()
        # Rebinds .grad = .main_grad, which is the same persistent view.
        return super().prepare_grads()

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the masters, then refresh this rank's rows of the bf16 pools."""
        success = super().step_with_ready_grads()
        self._sync_compute_weights()
        return success

    def _sync_compute_weights(self) -> None:
        for experts in self.moonep_experts:
            experts.sync_compute_weights()

    def reload_model_params(self, state_dict=None):
        """The masters are the model params; only the bf16 pools need refreshing."""
        del state_dict
        self._sync_compute_weights()

    def _stage_model_params_from_main_params(self) -> None:
        self._sync_compute_weights()

    def offload_to_cpu(self):
        raise NotImplementedError(
            "MoonEP masters alias VMM grad storage; CPU offload is unsupported."
        )

    def restore_from_cpu(self):
        raise NotImplementedError(
            "MoonEP masters alias VMM grad storage; CPU offload is unsupported."
        )
