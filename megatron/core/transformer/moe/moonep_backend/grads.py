# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Once-per-step MoonEP bookkeeping, called from ``finalize_model_grads``.

The expert weights are ordinary parameters, so DDP or Megatron-FSDP scales, reduces and
checks their gradients; nothing here touches them. This module only checks the data-parallel
wrapper once and closes the step's activation-recompute bookkeeping.
"""

from typing import List

import torch

from megatron.core.transformer.moe.moonep_backend import iter_moonep_experts
from megatron.core.transformer.moe.moonep_backend.buffer import end_moonep_pools_step


def _report_recompute_routing(model: List[torch.nn.Module]) -> None:
    """Report recomputes that routed tokens differently from their checkpointed forward.

    The recompute replays the forward's plan (``_MoonEPManager.dispatch``) and counts, on the
    device, the tokens it sends to another expert set; reading the counts costs one host sync,
    and only in steps that replayed a plan.
    """
    replayed = [
        experts
        for chunk in model
        for experts in iter_moonep_experts(chunk)
        if experts.moonep_state.recompute_tokens
    ]
    if not replayed:
        return
    counts = torch.stack([e.moonep_state.recompute_mismatches for e in replayed]).tolist()
    for experts, count in zip(replayed, counts):
        state = experts.moonep_state
        if count:
            experts.report_recompute_mismatches_(count, state.recompute_tokens)
        state.recompute_mismatches.zero_()
        state.recompute_tokens = 0


def assert_moonep_wrapper_supported(model: List[torch.nn.Module]) -> None:
    """MoonEP supports ZeRO-1 and ZeRO-2 data parallelism only.

    MoonEP stages each layer's whole local experts from their persistent parameters, so the
    parameters must not be sharded: plain DDP, DDP with the distributed optimizer (ZeRO-1), or
    Megatron-FSDP v1 with ``optim`` (ZeRO-1) or ``optim_grads`` (ZeRO-2). The modules holding
    the expert weights never run forward, so under Megatron-FSDP the hooks of an enclosing FSDP
    unit must gather them and mark how their gradients are written (``overwrite_main_grad``).
    ``validate_args`` checks the same for training scripts at startup; this covers library
    users from their first ``finalize_model_grads``: after the first backward, before the
    first optimizer step. ``MoonEPExpertPools.reduce_and_copy_out`` also raises in the first
    backward if no FSDP unit gathered the expert weights.
    """
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
    from megatron.core.distributed.torch_fully_sharded_data_parallel import (
        TorchFullyShardedDataParallel,
    )

    for chunk in model:
        if isinstance(chunk, (FullyShardedDataParallelV2, TorchFullyShardedDataParallel)):
            raise ValueError(
                "MoonEP supports Megatron DDP (optionally with the distributed optimizer) and "
                "Megatron-FSDP v1 with the 'optim' or 'optim_grads' sharding strategies; "
                f"{type(chunk).__name__} shards parameters."
            )
        ddp_config = getattr(chunk, "ddp_config", None)
        if ddp_config is None or not ddp_config.use_megatron_fsdp:
            continue
        strategies = {
            ddp_config.data_parallel_sharding_strategy,
            ddp_config.expert_data_parallel_sharding_strategy
            or ddp_config.data_parallel_sharding_strategy,
        }
        if not strategies <= {"optim", "optim_grads"}:
            raise ValueError(
                "MoonEP with Megatron-FSDP supports the 'optim' (ZeRO-1) and 'optim_grads' "
                "(ZeRO-2) sharding strategies only, for dense and expert parameters; got "
                f"{sorted(strategies)}."
            )
        if (
            ddp_config.num_distributed_optimizer_instances > 1
            or ddp_config.outer_dp_sharding_strategy != "no_shard"
        ):
            raise ValueError("MoonEP with Megatron-FSDP does not support hybrid sharding.")
        if ddp_config.megatron_fsdp_enable_fine_grained_param_gather:
            raise ValueError(
                "MoonEP with Megatron-FSDP needs whole-layer parameter gathering, not "
                "megatron_fsdp_enable_fine_grained_param_gather: the modules holding MoonEP's "
                "expert weights never run forward, so per-module hooks never gather them."
            )
        # Megatron-FSDP resolves class-path strings; the adapter keeps the list it was given.
        units = getattr(getattr(chunk, "module", None), "fsdp_unit_modules", None)
        if units is None:
            units = getattr(chunk, "fsdp_unit_modules", None)
        unit_types = tuple(units or ())
        modules = list(chunk.named_modules())
        units = [name for name, module in modules if unit_types and isinstance(module, unit_types)]
        for name, module in modules:
            inside = any(name.startswith(f"{unit}.") for unit in units)
            if getattr(module, "_is_moonep_experts", False) and not inside:
                raise ValueError(
                    "MoonEP with Megatron-FSDP needs every MoE layer inside an FSDP unit "
                    f"module; {name} is outside every unit "
                    f"({[unit_type.__name__ for unit_type in unit_types]}). The modules holding "
                    "MoonEP's expert weights never run forward, so only an enclosing unit's "
                    "hooks gather them."
                )


def finalize_moonep_step(model: List[torch.nn.Module]) -> None:
    """Close the step's MoonEP bookkeeping. Collective-free; identical on every rank.

    Every backward of the step has run by now, so a checkpointed forward still counted as
    waiting for its recompute never gets one (a forward-only pass in training mode); forgetting
    it lets the next step replay plans again. The pools' occupant is dropped too, so no plan of
    this step can match the next step's staging.
    """
    if not all(getattr(chunk, "_moonep_wrapper_checked", False) for chunk in model):
        assert_moonep_wrapper_supported(model)
        for chunk in model:
            chunk._moonep_wrapper_checked = True
    _report_recompute_routing(model)
    for chunk in model:
        for experts in iter_moonep_experts(chunk):
            experts.moonep_state.pending_checkpoints = 0
            experts.moonep_state.fwd_outstanding = False
    end_moonep_pools_step()
