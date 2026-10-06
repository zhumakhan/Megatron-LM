# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Grouped-GEMM SwiGLU experts of the MoonEP backend.

The expert weights are ordinary trainable parameters, named like ``TEGroupedMLP``'s: per local
expert ``linear_fc1.weight{i}`` ``[2*Hi, H]`` (gate rows, then up rows) and
``linear_fc2.weight{i}`` ``[H, Hi]``, in ``params_dtype`` and marked ``allreduce=False``. DDP
with the distributed optimizer (ZeRO-1) and Megatron-FSDP (``optim``/ZeRO-1 and
``optim_grads``/ZeRO-2) treat them like any other expert linear weight: they keep the fp32 main
weights, reduce the gradients over expert-DP and step them.

MoonEP computes on one VMM pool set shared by every MoE layer (``storage.MoonEPExpertPools``).
The dispatcher stages a layer's weights into the pools and pushes the copies peers need when
the layer computes, and hands the reduced gradients back to the parameters after
``Buffer.reduce_grad``. The grouped GEMMs here read the staged weights and their backward
writes dW into the pools' fp32 gradient rows.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moonep_backend.autograd import GroupedMMLiveGrad
from megatron.core.transformer.moe.moonep_backend.storage import MoonEPLayerState
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_object_for_checkpoint

logger = logging.getLogger(__name__)

# State-dict key prefix of the per-expert ``_extra_state`` placeholders (see sharded_state_dict).
_EXTRA_STATE_COMPAT = "_moonep_extra_state_compat."


def recompute_mismatch_message(layer_number, rank, count, total) -> str:
    """Text of the notice for recomputes that routed many tokens differently from their forward."""
    return (
        f"MoonEP: activation recomputes of MoE layer {layer_number} (rank {rank}) sent {count} "
        f"of {total} tokens to other experts than the checkpointed forwards whose plans they "
        "replayed. The recomputes and their gradients follow the forwards' routing. A recompute "
        "normally reproduces its forward bitwise (the router sorts its top-k in every training "
        "forward, with or without grad), so this many mean an op is not deterministic or an "
        "input changed between the forward and the backward."
    )


def mixed_requires_grad_message(layer_number) -> str:
    """Text of the error for a layer whose expert weights are only partly trainable."""
    return (
        f"MoonEP: MoE layer {layer_number} has trainable and frozen expert weights. MoonEP "
        "reduces all of a layer's expert gradients together; freeze all of them or none."
    )


class GroupedExpertWeights(torch.nn.Module):
    """One linear projection's per-expert weights ``weight0..weight{n-1}``.

    The names match TE's GroupedLinear inside ``TEGroupedMLP``, so Megatron's checkpoint,
    FSDP and conversion tooling handle them like any other grouped expert linear layer. The
    module only holds the weights: MoonEP's grouped GEMMs read the staged copies in the shared
    pools, never these tensors directly.
    """

    def __init__(self, num_weights: int, shape: Tuple[int, int], dtype: torch.dtype) -> None:
        super().__init__()
        self.num_weights = num_weights
        for i in range(num_weights):
            weight = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            )
            # An expert parameter: DDP, the distributed optimizer and Megatron-FSDP reduce and
            # shard it over expert-DP instead of the dense data-parallel group.
            weight.allreduce = False
            self.register_parameter(f"weight{i}", weight)

    def weights(self) -> List[torch.nn.Parameter]:
        """The weights, read through the attributes (wrappers may swap them between steps)."""
        return [getattr(self, f"weight{i}") for i in range(self.num_weights)]


class MoonEPGroupedMLP(MegatronModule):
    """SwiGLU experts of one MoE layer on MoonEP's shared expert pools.

    Args:
        num_local_experts: experts owned by this EP rank (``num_moe_experts / EP``).
        config: transformer config; must pass ``TransformerConfig._validate_moonep``.
        pg_collection: process groups; ``ep`` and ``expt_dp`` are used.
        name: module name, for logging.
    """

    _is_moonep_experts = True

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(config=config)
        assert pg_collection is not None, "MoonEPGroupedMLP requires a ProcessGroupCollection."
        self.name = name
        self.num_local_experts = num_local_experts
        self.ep_group = pg_collection.ep
        self.expt_dp_group = pg_collection.expt_dp
        self.tp_group = pg_collection.expt_tp
        self.layer_number = None
        assert num_local_experts * self.ep_group.size() == config.num_moe_experts, (
            f"MoonEP: num_local_experts={num_local_experts} x EP={self.ep_group.size()} != "
            f"num_moe_experts={config.num_moe_experts}."
        )
        hidden, ffn = config.hidden_size, config.moe_ffn_hidden_size
        dtype = config.params_dtype
        self.linear_fc1 = GroupedExpertWeights(num_local_experts, (2 * ffn, hidden), dtype)
        self.linear_fc2 = GroupedExpertWeights(num_local_experts, (hidden, ffn), dtype)
        if config.perform_initialization:
            self._init_weights_()
        # Per-layer bookkeeping shared with the layer's dispatcher, and the process-global
        # pools, which the dispatcher binds (MoELayer builds the dispatcher first).
        self.moonep_state = MoonEPLayerState()
        self.pools = None

    @torch.no_grad()
    def _init_weights_(self) -> None:
        """Initialize like ``SequentialMLP``: per local expert fc1, then fc2, in params_dtype under
        the expert-parallel RNG tracker, so the weights equal SequentialMLP's bit for bit."""
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            for fc1, fc2 in zip(self.linear_fc1.weights(), self.linear_fc2.weights()):
                self.config.init_method(fc1)
                self.config.output_layer_init_method(fc2)

    # ---- public helpers ---------------------------------------------------------------------

    def set_layer_number(self, layer_number: int) -> None:
        """Record the (1-based) transformer layer number used in messages."""
        self.layer_number = layer_number
        self.moonep_state.layer_number = layer_number

    def expert_weights(self) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        """``(fc1 weights, fc2 weights)`` of the local experts, in local expert order."""
        return self.linear_fc1.weights(), self.linear_fc2.weights()

    def trainable_expert_weights(self) -> Optional[bool]:
        """True if every expert weight is trainable, False if none is, None if mixed."""
        flags = {w.requires_grad for w in self.parameters()}
        return flags.pop() if len(flags) == 1 else None

    def weights_require_grad(self) -> bool:
        """Whether the expert weights are trainable (all of them or none)."""
        trainable = self.trainable_expert_weights()
        if trainable is None:
            raise RuntimeError(mixed_requires_grad_message(self.layer_number))
        return trainable

    def report_recompute_mismatches_(self, count: int, total: int) -> None:
        """Report that recomputes sent ``count`` of the ``total`` tokens they replayed since the
        last report (one step) to other experts.

        A recompute normally reproduces its forward bitwise; an op that is not deterministic
        (outside --deterministic-mode) can still flip a few near ties. Above 1% the recompute
        did not reproduce its forward.
        The counts differ between ranks, so this only warns (a raise on some ranks would leave
        the others waiting in the next collective).
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        message = recompute_mismatch_message(self.layer_number, rank, count, total)
        if count > 0.01 * total:
            logger.warning(message)
        else:
            logger.debug(message)

    def backward_dw(self) -> None:
        """dW is computed inline by the grouped-GEMM backward."""

    def forward(
        self, hidden_nvsh: torch.Tensor, cu_seqlens: torch.Tensor, route_weights_nvs: torch.Tensor
    ):
        """SwiGLU over the dispatched ``[plan.nvs_s, H]`` layout.

        The dispatcher staged this layer's weights in the shared pools right before.

        Args:
            hidden_nvsh: dispatched tokens in physical expert-group order.
            cu_seqlens: ``[2*epn]`` int32 padded segment END offsets (the ``offs`` format of
                ``torch._grouped_mm``): groups ``[0, epn)`` are this rank's experts, groups
                ``[epn, 2*epn)`` its prefetch slots, the row order of the pools.
            route_weights_nvs: ``[plan.nvs_s]`` fp32 routing weight of every slot.

        Returns:
            ``(output, None)``. Rows past ``cu_seqlens[-1]`` are uninitialized and never read.
        """
        pools = self.pools
        assert pools is not None, "MoonEP: the dispatcher did not bind its pools to the experts."
        # dW goes to the pools' fp32 rows only when the weights are trained.
        wgrad = torch.is_grad_enabled() and self.weights_require_grad()
        gmm = GroupedMMLiveGrad.apply

        def project(x, name):
            return gmm(x, pools.weights(name), cu_seqlens, pools.grads(name) if wgrad else None)

        gate = project(hidden_nvsh, "gate")
        up = project(hidden_nvsh, "up")
        # Routing weights apply after the activation and before fc2, in fp32, like Megatron's
        # MLP. Padding and tail slots hold stale weights: nan_to_num keeps them from minting
        # NaNs that a GEMM backward would spread.
        weights = torch.nan_to_num(route_weights_nvs).unsqueeze(-1)
        intermediate = ((F.silu(gate) * up).float() * weights).to(torch.bfloat16)
        output = project(intermediate, "down")
        return output, None

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Megatron-native expert keys and shapes, interchangeable with ``SequentialMLP`` and
        ``TEGroupedMLP``.

        Every local expert's fc1 and fc2 become fragments of ``{prefix}experts.linear_fc1.weight``
        and ``{prefix}experts.linear_fc2.weight`` with a global expert axis, exactly like a
        ``SequentialMLP`` expert (fc1 through the SwiGLU factory). Each entry's data is the
        parameter itself, so the distributed optimizer can map its state. Per-expert
        ``linear_fc{1,2}._extra_state`` objects (``None``, like Megatron's local linear layers)
        let ``SequentialMLP``/``TEGroupedMLP`` load a MoonEP checkpoint strictly; on load they
        are ignored.
        """
        # The experts need no process group from the metadata (the SwiGLU factory gets none).
        metadata = metadata or {}
        if metadata.get("singleton_local_shards", False):
            raise NotImplementedError(
                "MoonEP (phase 1) does not support singleton_local_shards checkpoints."
            )
        ep_rank = self.ep_group.rank()
        num_global_experts = self.ep_group.size() * self.num_local_experts
        replica_id = (0, 0, self.expt_dp_group.rank())
        sharded_state_dict = {}
        weights = zip(self.linear_fc1.weights(), self.linear_fc2.weights())
        for i, (fc1, fc2) in enumerate(weights):
            global_index = ep_rank * self.num_local_experts + i
            offsets = (*sharded_offsets, (len(sharded_offsets), global_index, num_global_experts))
            prepend = len(offsets)
            fc1_tensor = ShardedTensor.from_rank_offsets(
                f"{prefix}experts.linear_fc1.weight",
                fc1,
                *offsets,
                (prepend, 0, 1),  # the (size 1) expert-TP axis, as a ColumnParallelLinear
                replica_id=replica_id,
                prepend_axis_num=prepend,
            )
            sharded_state_dict[f"{prefix}linear_fc1.weight{i}"] = apply_swiglu_sharded_factory(
                fc1_tensor, offsets
            )
            sharded_state_dict[f"{prefix}linear_fc2.weight{i}"] = ShardedTensor.from_rank_offsets(
                f"{prefix}experts.linear_fc2.weight",
                fc2,
                *offsets,
                (prepend + 1, 0, 1),  # the (size 1) expert-TP axis, as a RowParallelLinear
                replica_id=replica_id,
                prepend_axis_num=prepend,
            )
            for linear in ("linear_fc1", "linear_fc2"):
                extra_state = make_sharded_object_for_checkpoint(
                    None, f"{prefix}experts.{linear}._extra_state", offsets, replica_id=replica_id
                )
                sharded_state_dict[f"{prefix}{_EXTRA_STATE_COMPAT}{linear}.{i}"] = extra_state
        return sharded_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Drop the loaded _extra_state placeholders; they have no module attribute.
        for key in [k for k in state_dict if k.startswith(prefix + _EXTRA_STATE_COMPAT)]:
            del state_dict[key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
