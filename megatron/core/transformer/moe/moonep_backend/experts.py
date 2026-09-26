# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Grouped-GEMM SwiGLU experts of the MoonEP backend, with fp32 masters and bf16 VMM pools.

Mixed precision follows MoonEP's reference model: the optimizer steps fp32 master parameters
(``weight_gate``/``weight_up``/``weight_down``, this rank's experts only), the forward and
backward run on bf16 compute copies in the layer's VMM pools, and the masters' ``.grad`` and
``.main_grad`` permanently alias fp32 VMM gradient rows that MoonEP reduces into.

The masters are ``requires_grad=False`` parameters: autograd never produces their gradients,
and the flag keeps them out of DDP buffers, DDP hooks, the distributed-optimizer layout and the
dense/expert Megatron optimizers. ``MoonEPOptimizer`` steps them and re-syncs the pools.
Because their gradients are persistent storage that must never be set to None, they are zeroed
in place by ``MegatronModule.zero_moe_grads()``; a per-layer state machine notices every
forward that would accumulate into stale gradients.
"""

import logging
import sys
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe.moonep_backend.autograd import GroupedMMLiveGrad
from megatron.core.transformer.moe.moonep_backend.storage import PROJECTIONS, MoonEPExpertStorage
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_object_for_checkpoint,
)

logger = logging.getLogger(__name__)

# State-dict key prefix of the per-expert ``_extra_state`` placeholders (see sharded_state_dict).
_EXTRA_STATE_COMPAT = "_moonep_extra_state_compat."

# Gradient states of one MoonEP layer.
CLEAN = "CLEAN"  # zeroed since the last zero point
ACCUMULATING = "ACCUMULATING"  # gradient-producing forwards ran since zeroing
STALE = "STALE"  # the framework passed a zero point (or the optimizer consumed the grads)


def _notify(msg: str, *, strict: bool) -> None:
    """Report a MoonEP gradient-lifecycle notice on every occurrence and every rank."""
    if strict:
        raise RuntimeError(msg)
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(msg)
    else:
        # --logging-level / MEGATRON_LOGGING_LEVEL above WARNING would hide the notice.
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def stale_grad_message(layer_number, rank) -> str:
    """Text of the notice for a forward that adds to non-zeroed gradients."""
    return (
        f"MoonEP expert gradients of MoE layer {layer_number} (rank {rank}) were not zeroed since "
        "the last optimizer step / zero_grad(): this micro-batch's gradients will be ADDED to "
        "stale values. Call model.zero_moe_grads() at the start of every iteration "
        "(megatron/training/training.py:train_step does this right after optimizer.zero_grad())."
    )


def discard_message(layer_number, rank) -> str:
    """Text of the notice for zeroing finalized gradients that no optimizer step consumed."""
    return (
        f"MoonEP: zero_moe_grads() discarded finalized expert gradients of MoE layer "
        f"{layer_number} (rank {rank}) that no optimizer step consumed. Call it at the start of "
        "an iteration, not between finalize_model_grads and optimizer.step()."
    )


def alias_message(name, layer_number, desc, where) -> str:
    """Text of the notice for a master whose ``.grad`` no longer aliases its VMM rows."""
    return (
        f"MoonEP expert master '{name}' of MoE layer {layer_number}: .grad was {desc} at {where}; "
        "expected the persistent fp32 VMM grad view. Something deleted or replaced it "
        "(torch.nn.Module.zero_grad / torch.optim.Optimizer.zero_grad with set_to_none=True?). "
        "Re-binding it."
    )


def no_grad_input_message(layer_number) -> str:
    """Text of the error for a training forward whose inputs carry no gradient."""
    return (
        f"MoonEP: MoE layer {layer_number} runs a training forward with grad enabled, but neither "
        "its input nor its router probabilities require grad, so no expert gradient can be "
        "produced (MoonEP masters are not autograd leaves). Unfreeze the router or an upstream "
        "layer, or run this forward under torch.no_grad()."
    )


class MoonEPGroupedMLP(MegatronModule):
    """SwiGLU experts of one MoE layer on MoonEP's composite VMM pools.

    Construction is collective over the EP group (VMM handle exchange).

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
        self.storage = MoonEPExpertStorage(self.ep_group, config.num_moe_experts, hidden, ffn)
        self.weight_gate = self._make_master((num_local_experts, ffn, hidden))
        self.weight_up = self._make_master((num_local_experts, ffn, hidden))
        self.weight_down = self._make_master((num_local_experts, hidden, ffn))
        if config.perform_initialization:
            self._init_masters_()

        self._grad_views = {}
        self._grad_view_ptrs = {}
        for name_ in PROJECTIONS:
            self._bind_grad_alias_(name_)
        self._master_ptrs = {n: p.data_ptr() for n, p in self._named_masters()}
        self._pool_ptrs = {n: self.storage.w[n].data_ptr() for n in PROJECTIONS}

        self.grad_state = CLEAN
        self.finalized = False
        self.sync_compute_weights()
        self.register_load_state_dict_post_hook(_sync_after_load)
        torch.cuda.synchronize()
        dist.barrier(group=self.ep_group)

    # ---- construction helpers ---------------------------------------------------------------

    @staticmethod
    def _make_master(shape) -> torch.nn.Parameter:
        param = torch.nn.Parameter(
            torch.empty(shape, dtype=torch.float32, device=torch.cuda.current_device()),
            requires_grad=False,
        )
        param.allreduce = False  # expert parallel: never all-reduced over the dense DP group
        param.moonep_master = True
        mark_keep_in_fp32(param)
        return param

    @torch.no_grad()
    def _init_masters_(self) -> None:
        """Initialize like ``SequentialMLP`` on the GPU, then upcast into the fp32 masters.

        Per local expert, a ``params_dtype`` fc1 ``[2*Hi, H]`` (gate rows first) gets
        ``init_method`` and a fc2 ``[H, Hi]`` gets ``output_layer_init_method``, in that order,
        under the expert-parallel RNG tracker: the starting point Megatron's own fp32 main
        params would have (``param.detach().clone().float()``).
        """
        ffn = self.config.moe_ffn_hidden_size
        hidden = self.config.hidden_size
        device = torch.cuda.current_device()
        dtype = self.config.params_dtype
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            for i in range(self.num_local_experts):
                fc1 = torch.empty(2 * ffn, hidden, dtype=dtype, device=device)
                self.config.init_method(fc1)
                fc2 = torch.empty(hidden, ffn, dtype=dtype, device=device)
                self.config.output_layer_init_method(fc2)
                self.weight_gate[i].copy_(fc1[:ffn])
                self.weight_up[i].copy_(fc1[ffn:])
                self.weight_down[i].copy_(fc2)

    def _named_masters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return [("gate", self.weight_gate), ("up", self.weight_up), ("down", self.weight_down)]

    def _master(self, name: str) -> torch.nn.Parameter:
        return getattr(self, f"weight_{name}")

    def _bind_grad_alias_(self, name: str) -> None:
        """Point the master's ``.grad`` and ``.main_grad`` at a fresh view of its VMM rows."""
        view = self.storage.g[name][self.storage.lo : self.storage.hi]
        mark_keep_in_fp32(view)
        master = self._master(name)
        master.grad = view
        master.main_grad = view
        self._grad_views[name] = view
        self._grad_view_ptrs[name] = view.data_ptr()

    # ---- public helpers ---------------------------------------------------------------------

    def set_layer_number(self, layer_number: int) -> None:
        """Record the (1-based) transformer layer number used in messages."""
        self.layer_number = layer_number
        self.storage.layer_number = layer_number

    def moonep_master_parameters(self) -> List[torch.nn.Parameter]:
        """The three fp32 master parameters of this layer."""
        return [p for _, p in self._named_masters()]

    def owned_grads(self) -> List[Tuple[str, torch.Tensor]]:
        """``(name, fp32 grad view)`` of this rank's experts; the masters' ``.grad``."""
        return [(f"weight_{n}", self._grad_views[n]) for n in PROJECTIONS]

    @torch.no_grad()
    def sync_compute_weights(self) -> None:
        """Copy the fp32 masters into this rank's rows of the bf16 pools.

        Peers read those rows over NVLink at their next ``prefetch_weight``, which the
        inter-rank sync at the start of every dispatch orders after this copy.
        """
        storage = self.storage
        for name, master in self._named_masters():
            storage.w[name][storage.lo : storage.hi].copy_(master)

    def refresh_cache(self) -> None:
        """A refit wrote the masters in place; refresh the bf16 pools derived from them."""
        self.sync_compute_weights()

    def check_invariants(self) -> List[str]:
        """Mixed-precision and aliasing invariants; returns the broken ones."""
        problems = []
        for name, master in self._named_masters():
            if master.dtype != torch.float32:
                problems.append(f"weight_{name} is {master.dtype}, expected torch.float32")
            if master.requires_grad:
                problems.append(f"weight_{name}.requires_grad is True, expected False")
            if master.data_ptr() != self._master_ptrs[name]:
                problems.append(f"weight_{name} storage moved")
            view = self._grad_views[name]
            for attr in ("grad", "main_grad"):
                grad = getattr(master, attr, None)
                if grad is not view or grad.data_ptr() != self._grad_view_ptrs[name]:
                    problems.append(f"weight_{name}.{attr} does not alias its VMM grad rows")
                elif grad.dtype != torch.float32:
                    problems.append(f"weight_{name}.{attr} is {grad.dtype}, expected fp32")
            pool = self.storage.w[name]
            if pool.dtype != torch.bfloat16 or pool.data_ptr() != self._pool_ptrs[name]:
                problems.append(f"the bf16 {name} pool was replaced or converted")
        return problems

    def raise_on_broken_invariants(self, where: str) -> None:
        """Raise if :meth:`check_invariants` finds any problem."""
        problems = self.check_invariants()
        if problems:
            raise RuntimeError(f"MoonEP invariants broken {where}: {problems}")

    def repair_grad_aliases_(self, where: str) -> None:
        """Notify about, and re-bind, every master whose ``.grad`` lost its VMM alias."""
        for name, master in self._named_masters():
            view = self._grad_views[name]
            grad = master.grad
            ok = (
                grad is view
                and view.data_ptr() == self._grad_view_ptrs[name]
                and view.dtype == torch.float32
                and getattr(master, "main_grad", None) is view
            )
            if ok:
                continue
            if grad is None:
                desc = "None"
            elif grad is view:
                desc = "the grad view with modified storage or dtype"
            else:
                desc = f"a different {grad.dtype} tensor"
            _notify(
                alias_message(f"weight_{name}", self.layer_number, desc, where),
                strict=self.config.moe_moonep_strict_zero_grad,
            )
            self._bind_grad_alias_(name)

    # ---- gradient-state machine -------------------------------------------------------------

    def zero_moonep_grads_(self) -> None:
        """Zero this layer's gradients in place and return to CLEAN."""
        if self.storage.fwd_outstanding:
            raise RuntimeError(
                f"MoonEP: zero_moe_grads() called while MoE layer {self.layer_number} has a "
                "forward without its backward."
            )
        strict = self.config.moe_moonep_strict_zero_grad
        if self.grad_state == ACCUMULATING and self.finalized:
            _notify(discard_message(self.layer_number, _rank()), strict=strict)
        self.repair_grad_aliases_("zero_moe_grads")
        self.storage.zero_grads_()
        self.grad_state, self.finalized = CLEAN, False

    def mark_zero_point_(self) -> None:
        """The framework reached its zeroing point (``MoonEPOptimizer.zero_grad``)."""
        if self.grad_state != CLEAN:
            self.grad_state = STALE
        self.finalized = False

    def mark_grads_consumed_(self) -> None:
        """The optimizer consumed the gradients (``MoonEPOptimizer.prepare_grads``)."""
        self.grad_state, self.finalized = STALE, False

    def _on_grad_forward_(self) -> None:
        """Every gradient-producing forward: warn if it would add to stale gradients."""
        self.repair_grad_aliases_("forward")
        if self.grad_state == STALE:
            _notify(
                stale_grad_message(self.layer_number, _rank()),
                strict=self.config.moe_moonep_strict_zero_grad,
            )
        self.grad_state = ACCUMULATING

    # ---- nn.Module overrides ----------------------------------------------------------------

    def _apply(self, fn, recurse=True):
        """Ignore dtype conversions and refuse device moves.

        ``Float16Module`` and ``.to()``/``.cuda()`` go through ``_apply``, which would replace
        the fp32 masters and convert their ``.grad`` (breaking the VMM alias). The masters stay
        fp32, and moving them is impossible because their gradients live in VMM memory mapped
        across the EP group.
        """
        del recurse  # no child modules
        for name, param in self._parameters.items():
            target = fn(torch.empty(0, dtype=param.dtype, device=param.device)).device
            if target != param.device:
                raise RuntimeError(
                    f"MoonEP: refusing to move '{name}' from {param.device} to {target}: MoonEP "
                    "expert tensors live in (or alias) VMM memory mapped across the EP group."
                )
        return self

    def backward_dw(self) -> None:
        """dW is computed inline by the grouped-GEMM backward."""

    def forward(
        self, hidden_nvsh: torch.Tensor, cu_seqlens: torch.Tensor, route_weights_nvs: torch.Tensor
    ):
        """SwiGLU over the dispatched ``[plan.nvs_s, H]`` layout.

        Args:
            hidden_nvsh: dispatched tokens in physical expert-group order.
            cu_seqlens: ``[E+B]`` int32 padded segment END offsets (the ``offs`` format of
                ``torch._grouped_mm``); segments outside ``[lo, hi)`` and ``[E, E+B)`` are
                empty on this rank.
            route_weights_nvs: ``[plan.nvs_s]`` fp32 routing weight of every slot.

        Returns:
            ``(output, None)``. Rows past ``cu_seqlens[-1]`` are uninitialized and never read.
        """
        grad_producing = torch.is_grad_enabled() and hidden_nvsh.requires_grad
        if self.training and torch.is_grad_enabled() and not hidden_nvsh.requires_grad:
            raise RuntimeError(no_grad_input_message(self.layer_number))
        if grad_producing:
            self._on_grad_forward_()

        storage = self.storage
        # The live groups are contiguous runs of END offsets, so concatenating them is a valid
        # compact `offs` over the same rows: a device-side cat, no host sync.
        offs_live = torch.cat((cu_seqlens[storage.lo : storage.hi], cu_seqlens[storage.E :]))
        gmm, pools, targets = GroupedMMLiveGrad.apply, storage.w, storage.fp32_targets
        gate = gmm(hidden_nvsh, pools["gate"], cu_seqlens, offs_live, targets("gate"))
        up = gmm(hidden_nvsh, pools["up"], cu_seqlens, offs_live, targets("up"))
        # Routing weights apply after the activation and before fc2, in fp32, like Megatron's
        # MLP. Padding and tail slots hold stale weights: nan_to_num keeps them from minting
        # NaNs that a GEMM backward would spread.
        weights = torch.nan_to_num(route_weights_nvs).unsqueeze(-1)
        intermediate = ((F.silu(gate) * up).float() * weights).to(torch.bfloat16)
        output = gmm(intermediate, pools["down"], cu_seqlens, offs_live, targets("down"))
        return output, None

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Megatron-native expert keys and shapes, interchangeable with ``SequentialMLP``.

        fc1 is ``gate || up`` along its output axis, expressed as two fragments of
        ``{prefix}experts.linear_fc1.weight`` like the SwiGLU sharded factory; the expert axis is
        sharded over EP with this rank's ``[epn, ...]`` chunk. Per-expert
        ``linear_fc{1,2}._extra_state`` objects (``None``, like Megatron's local linear layers)
        let ``SequentialMLP``/``TEGroupedMLP`` load a MoonEP checkpoint strictly; on load they
        are ignored (MoonEP experts are bf16 and carry no FP8 metadata).
        """
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        if metadata.get("singleton_local_shards", False):
            raise NotImplementedError(
                "MoonEP (phase 1) does not support singleton_local_shards checkpoints."
            )
        axis = len(sharded_offsets)
        ep_rank, ep_size = self.ep_group.rank(), self.ep_group.size()
        kwargs = dict(
            replica_id=(0, 0, self.expt_dp_group.rank()), prepend_axis_num=len(sharded_offsets)
        )
        fc1_key = f"{prefix}experts.linear_fc1.weight"
        fc2_key = f"{prefix}experts.linear_fc2.weight"
        ep_offset = (axis, ep_rank, ep_size)
        sharded_state_dict = {
            f"{prefix}weight_gate": ShardedTensor.from_rank_offsets(
                fc1_key, self.weight_gate, *sharded_offsets, ep_offset, (axis + 1, 0, 2), **kwargs
            ),
            f"{prefix}weight_up": ShardedTensor.from_rank_offsets(
                fc1_key, self.weight_up, *sharded_offsets, ep_offset, (axis + 1, 1, 2), **kwargs
            ),
            f"{prefix}weight_down": ShardedTensor.from_rank_offsets(
                fc2_key, self.weight_down, *sharded_offsets, ep_offset, **kwargs
            ),
        }
        num_global_experts = ep_size * self.num_local_experts
        for i in range(self.num_local_experts):
            global_index = ep_rank * self.num_local_experts + i
            expert_offsets = (*sharded_offsets, (axis, global_index, num_global_experts))
            for linear in ("linear_fc1", "linear_fc2"):
                extra_state = make_sharded_object_for_checkpoint(
                    None,
                    f"{prefix}experts.{linear}._extra_state",
                    expert_offsets,
                    replica_id=kwargs["replica_id"],
                )
                sharded_state_dict[f"{prefix}{_EXTRA_STATE_COMPAT}{linear}.{i}"] = extra_state
        return sharded_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Drop the loaded _extra_state placeholders; they have no module attribute.
        for key in [k for k in state_dict if k.startswith(prefix + _EXTRA_STATE_COMPAT)]:
            del state_dict[key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def _sync_after_load(module: MoonEPGroupedMLP, incompatible_keys) -> None:
    """``load_state_dict`` copies into the masters in place; refresh the bf16 pools."""
    del incompatible_keys
    module.sync_compute_weights()
