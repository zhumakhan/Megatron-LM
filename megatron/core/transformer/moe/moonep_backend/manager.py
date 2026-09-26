# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``_MoonEPManager``: the MoonEP backend of ``MoEFlexTokenDispatcher``."""

import importlib.util

import torch
import torch.distributed as dist

from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.moe.moonep_backend.autograd import (
    MoonEPCombine,
    MoonEPDispatch,
    align_ep_group_on_first_use,
)
from megatron.core.transformer.moe.moonep_backend.buffer import get_moonep_buffer
from megatron.core.transformer.moe.moonep_backend.topology import (
    assert_moonep_ep_group_nvlink_coverage,
)
from megatron.core.transformer.moe.token_dispatcher import _DispatchManager
from megatron.core.transformer.transformer_config import TransformerConfig

INFERENCE_MSG = (
    "MoonEP (phase 1) does not support Megatron inference engines: MoonEP needs the same token "
    "count on every EP rank for every call. Use the alltoall dispatcher or the deepep/hybridep "
    "flex backends for inference."
)


def capacity_message(layer_number, num_tokens, capacity) -> str:
    """Text of the error for more tokens than the MoonEP Buffer holds."""
    return (
        f"MoonEP: MoE layer {layer_number} received {num_tokens} tokens on this rank but "
        f"moe_moonep_max_tokens_per_rank={capacity}. Raise --moe-moonep-max-tokens-per-rank to "
        "at least seq_length*max(micro_batch_size, eval_micro_batch_size)/context_parallel_size "
        "(divided by tensor_model_parallel_size with sequence parallelism)."
    )


def dtype_message(dtype) -> str:
    """Text of the error for non-bf16 hidden states."""
    return f"MoonEP dispatch requires bf16 hidden states (run with --bf16), got {dtype}."


def guard_message(layer_number) -> str:
    """Text of the error for a second gradient-tracking forward before the backward."""
    return (
        f"MoonEP: layer {layer_number} started a second gradient-tracking forward before the "
        "backward of the previous one. Phase 1 keeps one prefetch/reduce slot set per layer, so "
        "this would silently corrupt gradients. Causes: pipeline/virtual pipeline parallelism, "
        "overlap_moe_expert_parallel_comm, MoE activation recompute, mlp_chunks_for_training > "
        "1, repeated MTP layers, or calling the model twice before backward."
    )


class _MoonEPManager(_DispatchManager):
    """Dispatch/combine through one process-global ``moonep.Buffer``.

    Workflow per MoE layer call:
    (1) ``setup_metadata`` (inside the dispatcher's compiled preprocess): masked top-k of the
        dense probs into fp32 ``[T, K]`` routing weights, int32 expert ids and per-expert counts.
    (2) ``dispatch``: ``Buffer.dispatch`` (autograd: combine backward plus ``reduce_grad``),
        then ``Buffer.prefetch_weight`` of this layer's pools with the fresh plan.
    (3) The experts run on the dispatched layout; ``get_number_of_tokens_per_expert`` returns
        the ``[E+B]`` cumulative END offsets, not counts.
    (4) ``combine``: ``Buffer.combine`` (autograd: plan-reuse dispatch backward).

    Args:
        group: the expert-parallel group (expert TP must be 1).
        num_local_experts: experts per EP rank.
        router_topk: K.
        num_experts: E.
        config: transformer config.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: TransformerConfig,
    ) -> None:
        if importlib.util.find_spec("moonep") is None:
            raise ImportError(
                "moe_flex_dispatcher_backend='moonep' needs the moonep package "
                "(https://github.com/zhumakhan/MoonEP)."
            )
        assert group.size() > 1 and num_experts % group.size() == 0, (
            f"MoonEP needs EP > 1 and num_experts divisible by EP, got EP={group.size()}, "
            f"num_experts={num_experts}."
        )
        assert config.moe_moonep_max_tokens_per_rank is not None, (
            "MoonEP needs moe_moonep_max_tokens_per_rank (megatron/training derives it in "
            "validate_args; library users must set it)."
        )
        self.group = group
        self.num_local_experts = num_local_experts
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.config = config
        self._ranks = tuple(dist.get_process_group_ranks(group))
        # Cached after initialize_megatron's WORLD-consensus check; covers library users.
        assert_moonep_ep_group_nvlink_coverage(group)
        self.buffer = get_moonep_buffer(
            group,
            max_tokens_per_rank=config.moe_moonep_max_tokens_per_rank,
            hidden_size=config.hidden_size,
            topk=router_topk,
            num_experts=num_experts,
            num_sms=config.moe_flex_dispatcher_num_sms,
            token_padding=config.moe_moonep_token_padding,
        )
        assert int(self.buffer._require_ctx()["B"]) == num_experts // group.size(), (
            "MoonEP requires B == num_experts / EP prefetch slots."
        )
        self.storage = None
        self.token_probs = None
        self.token_indices = None
        self.tokens_per_expert = None
        self.dispatched_probs = None
        self.cu_seqlens = None
        self.plan = None
        self.last_plan = None

    def bind_storage(self, storage) -> None:
        """Attach the layer's ``MoonEPExpertStorage`` (the experts are built after us)."""
        self.storage = storage

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        # Runs under torch.compile (dispatch_preprocess is @jit_fuser): no MoonEP calls here.
        num_tokens = routing_map.shape[0]
        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)
        # A routed prob that underflows to 0 must still win over unrouted experts, so mask
        # before the top-k instead of taking the top-k of the dense probs.
        token_probs, token_indices = torch.topk(
            probs.masked_fill(~routing_map, float("-inf")), self.router_topk, dim=-1
        )
        self.token_probs = token_probs.float()
        self.token_indices = token_indices.to(torch.int32)
        self.tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int32)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # Phase 1 always runs synchronously on the current stream.
        del async_finish, allocate_on_comm_stream
        storage = self.storage
        assert storage is not None, "MoonEP: bind_storage() was not called for this MoE layer."
        num_tokens = hidden_states.shape[0]
        capacity = self.config.moe_moonep_max_tokens_per_rank
        # Every guard raises before the first barrier-bearing MoonEP call, and its condition is
        # identical on all EP ranks, so the ranks raise together instead of hanging.
        if InferenceMode.is_active():
            raise RuntimeError(INFERENCE_MSG)
        if num_tokens > capacity:
            raise RuntimeError(capacity_message(storage.layer_number, num_tokens, capacity))
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(dtype_message(hidden_states.dtype))
        # Function.forward runs with grad disabled, so decide this before apply().
        grad_tracking = torch.is_grad_enabled() and (
            hidden_states.requires_grad or self.token_probs.requires_grad
        )
        if grad_tracking:
            if storage.fwd_outstanding:
                raise RuntimeError(guard_message(storage.layer_number))
            storage.fwd_outstanding = True
        first_use_key = ("dispatch", self._ranks, num_tokens, grad_tracking)
        align_ep_group_on_first_use(self.group, first_use_key)
        holder = {}
        hidden_nvsh, route_weights_nvs, cu_seqlens = MoonEPDispatch.apply(
            hidden_states,
            self.token_probs,
            self.token_indices,
            self.tokens_per_expert,
            self.buffer,
            storage,
            holder,
        )
        self.plan = holder["plan"]
        # One-sided NVLink read of the peers' expert rows this plan copies into the local
        # slots. Correct because a rank writes its own rows only in sync_compute_weights, which
        # the inter-rank sync at the start of the dispatch above orders before this read.
        self.buffer.prefetch_weight(
            plan=self.plan,
            full_gate_weight=storage.w["gate"],
            full_up_weight=storage.w["up"],
            full_down_weight=storage.w["down"],
        )
        self.dispatched_probs, self.cu_seqlens = route_weights_nvs, cu_seqlens
        return hidden_nvsh

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
        """Dispatch already produced the expert-grouped layout."""
        return hidden_states, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """The ``[E+B]`` int32 cumulative END offsets of the dispatched expert segments."""
        return self.cu_seqlens

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor):
        """Combine reads the expert-grouped layout directly."""
        return hidden_states

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        del async_finish, allocate_on_comm_stream
        align_ep_group_on_first_use(self.group, ("combine", self._ranks))
        output = MoonEPCombine.apply(hidden_states, self.buffer, self.plan, self.group, self._ranks)
        self.last_plan, self.plan = self.plan, None
        self.dispatched_probs, self.cu_seqlens = None, None
        return output
