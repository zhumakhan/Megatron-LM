# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``_MoonEPManager``: the MoonEP backend of ``MoEFlexTokenDispatcher``."""

import importlib.util
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist

from megatron.core.inference.utils import InferenceMode
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.moe.moonep_backend.autograd import (
    MoonEPCombine,
    MoonEPDispatch,
    align_ep_group_on_first_use,
)
from megatron.core.transformer.moe.moonep_backend.buffer import (
    get_moonep_buffer,
    get_moonep_expert_pools,
)
from megatron.core.transformer.moe.moonep_backend.topology import (
    assert_moonep_ep_group_nvlink_coverage,
)
from megatron.core.transformer.moe.token_dispatcher import _DispatchManager
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import ensure_params_ready

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
        "backward of the previous one. Phase 1 supports one outstanding gradient-tracking "
        "forward per MoE layer: the expert weights receive their gradients once per backward, "
        "and under ZeRO-2 that write overwrites the wrapper's gradient buffer. Causes: "
        "pipeline/virtual pipeline parallelism, overlap_moe_expert_parallel_comm, "
        "mlp_chunks_for_training > 1, repeated MTP layers, calling the model twice before "
        "backward, or a gradient-tracking forward whose backward never ran (run forward-only "
        "passes under torch.no_grad())."
    )


def trainability_message(layer_number, trainable, frozen, mixed) -> str:
    """Text of the error for expert weights trainable on some EP ranks only."""
    return (
        f"MoonEP: the expert weights of MoE layer {layer_number} must be all trainable or all "
        "frozen on every EP rank: the weight-gradient reduce is collective, and each rank "
        f"computes dW for the peers' experts it prefetched. Got {trainable} rank(s) trainable, "
        f"{frozen} frozen and {mixed} mixed. Freeze or unfreeze a layer's experts on every "
        "rank together."
    )


def recompute_tokens_message(layer_number, num_tokens, recorded) -> str:
    """Text of the error for a recompute whose token count differs from its forward."""
    return (
        f"MoonEP: the activation recompute of MoE layer {layer_number} dispatches {num_tokens} "
        f"tokens, but the checkpointed forward whose plan it reuses dispatched {recorded}."
    )


def _in_backward() -> bool:
    """Whether this thread runs an autograd backward, as every checkpoint recompute does."""
    return torch._C._current_graph_task_id() != -1


@dataclass
class _CheckpointedDispatch:
    """What the first forward of a checkpointed MoE layer leaves for its recompute."""

    plan: Any  # moonep.MoonEPCommPlan
    cu_seqlens: torch.Tensor  # [2*epn] int32 segment END offsets
    token_indices: torch.Tensor  # [T, K] int32 routed experts, the order the plan was built in
    token_probs: torch.Tensor  # [T, K] fp32 routing weights of that forward
    num_tokens: int


class _MoonEPManager(_DispatchManager):
    """Dispatch/combine through one process-global ``moonep.Buffer`` and expert pool set.

    Workflow per MoE layer call:
    (1) ``setup_metadata`` (inside the dispatcher's compiled preprocess): masked top-k of the
        dense probs into fp32 ``[T, K]`` routing weights, int32 expert ids and per-expert counts.
    (2) ``dispatch``: ``Buffer.dispatch`` (autograd: combine backward, then the experts' weight
        gradients), then ``MoonEPExpertPools.stage``: the layer's expert weights are copied into
        the shared pools and the collective ``Buffer.prefetch_weight`` pushes every rank's own
        experts into the prefetch slots of the ranks that copy them.
    (3) The experts run on the dispatched layout; ``get_number_of_tokens_per_expert`` returns
        the ``[2*epn]`` cumulative END offsets, not counts.
    (4) ``combine``: ``Buffer.combine`` (autograd: plan-reuse dispatch backward, then the
        layer's weights are staged again for its GEMM backward unless the pools still hold them).

    Under full block-wise activation recompute (the only MoE recompute ``_validate_moonep``
    accepts) a layer's forward runs twice: first without grad inside the checkpoint, then again
    in the backward. The first forward plans as usual and records its plan; the recompute
    dispatches with that plan instead of planning again, and stages the weights unless the
    pools still hold that forward's copies. It replays only when its forward is the layer's one
    checkpointed forward waiting for a recompute, as in Megatron's schedules; otherwise it plans
    again (see ``dispatch``).

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
        ctx = self.buffer._require_ctx()
        assert int(ctx["E"]) // int(ctx["R"]) == num_local_experts, (
            "MoonEP derives num_experts / EP local experts and prefetch slots per rank from the "
            f"Buffer, got E={ctx['E']}, R={ctx['R']} for {num_local_experts} local experts."
        )
        # Collective on first use; every rank builds its MoE layers in the same order.
        self.pools = get_moonep_expert_pools(
            self.buffer, group, config.hidden_size, config.moe_ffn_hidden_size
        )
        self._experts = None
        self.dense_probs = None
        self.token_probs = None
        self.token_indices = None
        self.tokens_per_expert = None
        self._checkpointed = None
        self._restage = None
        self._trainability_checked = False
        self.dispatched_probs = None
        self.cu_seqlens = None
        self.plan = None
        self.last_plan = None

    def bind_experts(self, experts) -> None:
        """Attach the layer's ``MoonEPGroupedMLP`` (the experts are built after us)."""
        self._experts = experts
        experts.pools = self.pools

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
        self.dense_probs = probs
        self.token_probs = token_probs.float()
        self.token_indices = token_indices.to(torch.int32)
        self.tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int32)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        total_num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        # Phase 1 always runs synchronously on the current stream.
        del async_finish, allocate_on_comm_stream
        experts = self._experts
        assert experts is not None, "MoonEP: bind_experts() was not called for this MoE layer."
        state, pools = experts.moonep_state, self.pools
        num_tokens = hidden_states.shape[0]
        capacity = self.config.moe_moonep_max_tokens_per_rank
        # Every guard raises before the first barrier-bearing MoonEP call, and its condition is
        # identical on all EP ranks, so the ranks raise together instead of hanging.
        if InferenceMode.is_active():
            raise RuntimeError(INFERENCE_MSG)
        if num_tokens > capacity:
            raise RuntimeError(capacity_message(state.layer_number, num_tokens, capacity))
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(dtype_message(hidden_states.dtype))
        # Read the weights through the module attributes on every call: DDP re-points parameter
        # storage, and Megatron-FSDP swaps the attributes for optimizer shards between steps.
        fc1, fc2 = experts.expert_weights()
        params = (*fc1, *fc2)
        if torch.is_grad_enabled() and not self._trainability_checked:
            self._check_trainability(experts)
        # Function.forward runs with grad disabled, so decide these before apply().
        wgrad = torch.is_grad_enabled() and experts.weights_require_grad()
        grad_tracking = torch.is_grad_enabled() and (
            hidden_states.requires_grad or self.token_probs.requires_grad or wgrad
        )
        dense_probs, self.dense_probs = self.dense_probs, None
        if grad_tracking:
            if state.fwd_outstanding:
                raise RuntimeError(guard_message(state.layer_number))
        # An activation checkpoint runs its first forward without grad and the recompute, in the
        # backward, with grad; is_checkpointing() is set in both. Megatron sets and clears that
        # flag without try/finally, so a recompute must also run inside a backward: a flag left
        # set by an exception never turns an ordinary forward into a replay. The record can only
        # be this recompute's own forward when that forward is the only checkpointed one of the
        # layer still waiting for its recompute. With more waiting (two forwards before a
        # backward, or a training-mode forward under no_grad that never gets one), the
        # recompute plans again. The count changes the same way on every EP rank, so all of them
        # take the same path.
        recorded = None
        if is_checkpointing() and grad_tracking and _in_backward():
            if state.pending_checkpoints == 1:
                recorded = self._checkpointed
            self._checkpointed = None
            state.pending_checkpoints = max(state.pending_checkpoints - 1, 0)
            if recorded is not None and recorded.num_tokens != num_tokens:
                raise RuntimeError(
                    recompute_tokens_message(state.layer_number, num_tokens, recorded.num_tokens)
                )
        if grad_tracking:
            state.fwd_outstanding = True
        holder = {}
        if recorded is None:
            # stage() below reads the weights. A backend that publishes them asynchronously
            # (DDP with overlap_param_gather) guarantees them only in the owning module's forward
            # pre-hook, and the modules holding the expert weights never run forward, so finish
            # the publication here. A recompute runs inside the backward, after the forward
            # already did this.
            if not _in_backward():
                ensure_params_ready(params)
            first_use_key = ("dispatch", self._ranks, grad_tracking)
            align_ep_group_on_first_use(self.group, first_use_key)
            hidden_nvsh, route_weights_nvs = MoonEPDispatch.apply(
                hidden_states,
                self.token_probs,
                self.token_indices,
                self.tokens_per_expert,
                self.buffer,
                pools,
                state,
                holder,
                None,
                total_num_tokens,
                wgrad,
                *params,
            )
            self.plan, cu_seqlens = holder["plan"], holder["cu_seqlens"]
            # The first forward of a checkpoint records its plan, unless another checkpointed
            # forward of the layer is still waiting for its recompute. Any other dispatch of the
            # layer drops the record, so a replayed plan always comes from the forward right
            # before its recompute.
            self._checkpointed = None
            if is_checkpointing() and not torch.is_grad_enabled():
                state.pending_checkpoints += 1
                if state.pending_checkpoints == 1:
                    self._checkpointed = _CheckpointedDispatch(
                        self.plan, cu_seqlens, self.token_indices, self.token_probs, num_tokens
                    )
        else:
            # Recompute of the layer's only waiting checkpointed forward: dispatch the recomputed
            # tokens with that forward's plan, so they land where its layout put them, without
            # planning again. Every EP rank replays its own record, so all of them skip the same
            # barrier-bearing planning.
            align_ep_group_on_first_use(self.group, ("dispatch_replay", self._ranks))
            # The routing weights are the recomputed probabilities at the recorded experts, so
            # the router gets their gradient. The recompute normally reproduces the forward
            # bitwise; a recorded expert that it no longer selects (a near tie flipped by an op
            # that is not deterministic) keeps the forward's weight. finalize_model_grads
            # reports a layer where many tokens flipped.
            token_indices = recorded.token_indices.long()
            reselected = (
                token_indices.unsqueeze(-1) == self.token_indices.long().unsqueeze(-2)
            ).any(dim=-1)
            route_weights = torch.where(
                reselected, dense_probs.gather(1, token_indices), recorded.token_probs
            )
            state.recompute_mismatches.add_((~reselected).any(dim=-1).sum())
            state.recompute_tokens += num_tokens
            hidden_nvsh, route_weights_nvs = MoonEPDispatch.apply(
                hidden_states,
                route_weights,
                None,
                None,
                self.buffer,
                pools,
                state,
                holder,
                recorded.plan,
                total_num_tokens,
                wgrad,
                *params,
            )
            self.plan, cu_seqlens = holder["plan"], recorded.cu_seqlens
        # Copy the layer's weights into the shared pools and push the copies its plan assigns
        # to peers. Skipped when the pools still hold this layer with this plan: a recompute
        # right after its own checkpointed forward.
        pools.stage(state, self.plan, fc1, fc2)
        self._restage = (pools, state, fc1, fc2, wgrad)
        self.dispatched_probs, self.cu_seqlens = route_weights_nvs, cu_seqlens
        return hidden_nvsh

    def _check_trainability(self, experts) -> None:
        """Raise on every EP rank unless the layer's experts are trainable on all of them or
        frozen on all of them.

        Whether a layer computes and reduces dW drives the collective ``reduce_grad``, and a
        rank's prefetch slots hold its peers' experts, so the decision must be the same on
        every rank. This runs at the layer's first dispatch with grad enabled, which every
        rank reaches together; later changes of ``requires_grad`` must keep it uniform.
        """
        trainable = experts.trainable_expert_weights()
        counts = torch.zeros(3, dtype=torch.int64, device=torch.cuda.current_device())
        counts[{True: 0, False: 1, None: 2}[trainable]] = 1
        dist.all_reduce(counts, group=self.group)
        num_trainable, num_frozen, num_mixed = counts.tolist()
        if num_mixed or (num_trainable and num_frozen):
            raise RuntimeError(
                trainability_message(experts.layer_number, num_trainable, num_frozen, num_mixed)
            )
        self._trainability_checked = True

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
        """Dispatch already produced the expert-grouped layout."""
        return hidden_states, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """The ``[2*epn]`` int32 cumulative END offsets of the dispatched expert segments."""
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
        output = MoonEPCombine.apply(
            hidden_states, self.buffer, self.plan, self.group, self._ranks, self._restage
        )
        self.last_plan, self.plan, self._restage = self.plan, None, None
        self.dispatched_probs, self.cu_seqlens = None, None
        return output
