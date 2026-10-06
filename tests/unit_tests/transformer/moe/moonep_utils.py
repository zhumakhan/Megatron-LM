# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared helpers for the MoonEP backend unit tests."""

import gc
import importlib.util
from contextlib import contextmanager
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _moonep_skip_reason():
    if importlib.util.find_spec("moonep") is None:
        return "moonep not installed"
    if not torch.cuda.is_available():
        return "no CUDA"
    if torch.cuda.get_device_capability()[0] < 9:
        return "MoonEP needs compute capability >= 9.0"
    if Utils.world_size < 2:
        return "MoonEP needs at least 2 GPUs"
    return None


MOONEP_SKIP_REASON = _moonep_skip_reason()
requires_moonep = pytest.mark.skipif(
    MOONEP_SKIP_REASON is not None, reason=MOONEP_SKIP_REASON or ""
)

NUM_EXPERTS, TOPK, HIDDEN, FFN = 16, 4, 1024, 1024


def moe_config_kwargs(ep_size, **overrides):
    """A phase-1-compatible MoE config shared by the MoonEP layer and its alltoall baseline."""
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=FFN,
        ffn_hidden_size=FFN,
        moe_router_topk=TOPK,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        moe_router_dtype="fp32",
        use_cpu_initialization=False,
    )
    kwargs.update(overrides)
    return kwargs


def moonep_config(ep_size, max_tokens, **overrides):
    """TransformerConfig selecting the MoonEP flex backend."""
    return TransformerConfig(
        **moe_config_kwargs(
            ep_size,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="moonep",
            moe_moonep_max_tokens_per_rank=max_tokens,
            **overrides,
        )
    )


def alltoall_config(ep_size, **overrides):
    """The same model on the alltoall dispatcher with SequentialMLP experts."""
    return TransformerConfig(
        **moe_config_kwargs(ep_size, moe_token_dispatcher_type="alltoall", **overrides)
    )


def build_moe_layer(config, layer_number=1):
    """An MoELayer from the local spec (SequentialMLP unless the backend is MoonEP)."""
    submodules = get_submodules(get_gpt_layer_local_submodules(num_experts=NUM_EXPERTS).mlp)
    layer = MoELayer(config, submodules).cuda()
    layer.set_layer_number(layer_number)
    return layer


@torch.no_grad()
def copy_sequential_to_moonep(base_layer, moonep_layer):
    """Copy router and SequentialMLP expert weights into a MoonEP layer: expert ``i``'s
    ``local_experts.{i}.linear_fc{1,2}.weight`` becomes ``linear_fc{1,2}.weight{i}``."""
    moonep_layer.router.weight.copy_(base_layer.router.weight)
    fc1, fc2 = moonep_layer.experts.expert_weights()
    for i, expert in enumerate(base_layer.experts.local_experts):
        fc1[i].copy_(expert.linear_fc1.weight)
        fc2[i].copy_(expert.linear_fc2.weight)
    torch.cuda.synchronize()
    torch.distributed.barrier()


@torch.no_grad()
def copy_moonep_layer(src, dst):
    """Copy every parameter of a MoonEP MoE layer into another one."""
    dst_params = dict(dst.named_parameters())
    for name, param in src.named_parameters():
        dst_params[name].copy_(param)
    torch.cuda.synchronize()
    torch.distributed.barrier()


def sequential_expert_grads(base_layer):
    """Per local expert ``(gate, up, down)`` fp32 grads of a SequentialMLP layer."""
    grads = []
    ffn = base_layer.config.moe_ffn_hidden_size
    for expert in base_layer.experts.local_experts:
        fc1 = expert.linear_fc1.weight
        fc2 = expert.linear_fc2.weight
        g1 = getattr(fc1, "main_grad", None) if fc1.grad is None else fc1.grad
        g2 = getattr(fc2, "main_grad", None) if fc2.grad is None else fc2.grad
        grads.append((g1[:ffn].float(), g1[ffn:].float(), g2.float()))
    return grads


def moonep_expert_grads(moonep_layer):
    """Per local expert ``(gate, up, down)`` fp32 grads of a MoonEP layer: ``.grad`` without a
    wrapper, ``main_grad`` once DDP or Megatron-FSDP took the gradients over."""
    experts = moonep_layer.experts
    ffn = experts.config.moe_ffn_hidden_size
    grads = []
    for fc1, fc2 in zip(*experts.expert_weights()):
        g1 = fc1.main_grad if getattr(fc1, "grad_added_to_main_grad", False) else fc1.grad
        g2 = fc2.main_grad if getattr(fc2, "grad_added_to_main_grad", False) else fc2.grad
        grads.append((g1[:ffn].float(), g1[ffn:].float(), g2.float()))
    return grads


@contextmanager
def moonep_calls(buffer):
    """Count the Buffer's dispatches by kind and its prefetch_weight / reduce_grad calls."""
    counts = {}
    with mock.patch.object(buffer, "dispatch", wraps=buffer.dispatch) as dispatch:
        with mock.patch.object(buffer, "prefetch_weight", wraps=buffer.prefetch_weight) as push:
            with mock.patch.object(buffer, "reduce_grad", wraps=buffer.reduce_grad) as reduce:
                yield counts
                calls = dispatch.call_args_list
                # A fresh dispatch plans; a replay reuses a plan for tokens and routing weights;
                # the backward of combine reuses it for the output gradient only.
                reused = [c for c in calls if c.kwargs.get("plan") is not None]
                counts["fresh"] = len(calls) - len(reused)
                counts["replay"] = sum(len(c.args) > 1 for c in reused)
                counts["combine_backward"] = sum(len(c.args) == 1 for c in reused)
                counts["prefetch"] = push.call_count
                counts["reduce"] = reduce.call_count


def rel_err(actual, expected):
    """Relative L2 error."""
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


class RankChecks:
    """Collect checks, then fail on every rank together.

    A rank that raised alone would leave its peers spinning in MoonEP's device barriers (which
    trap after a timeout) or in the next NCCL collective, so tests record their checks and call
    :meth:`assert_all` once, after their last collective.
    """

    def __init__(self) -> None:
        self.failures = []

    def check(self, ok, message: str) -> None:
        """Record a failure unless ``ok``."""
        if not bool(ok):
            self.failures.append(message)

    def close(self, name: str, actual, expected, tol: float) -> None:
        """Record a failure unless the relative L2 error is below ``tol``."""
        err = rel_err(actual, expected)
        self.check(err < tol, f"{name}: relative error {err:.3e} >= {tol}")

    def equal(self, name: str, actual, expected) -> None:
        """Record a failure unless the tensors are bitwise equal."""
        self.check(torch.equal(actual, expected), f"{name}: tensors differ")

    def assert_all(self) -> None:
        """Raise on every rank if any rank recorded a failure. Collective (WORLD)."""
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, self.failures)
        failed = {rank: failures for rank, failures in enumerate(gathered) if failures}
        if failed:
            raise AssertionError(f"MoonEP check failures by rank: {failed}")


def free_moonep_models():
    """Drop MoonEP models collectively: finish all work before any rank unmaps VMM memory."""
    gc.collect()
    torch.cuda.synchronize()
    torch.distributed.barrier()


def teardown_moonep():
    """Collective teardown: models, the process-global Buffer, then the process groups."""
    from megatron.core.transformer.moe.moonep_backend.buffer import destroy_moonep_buffer

    free_moonep_models()
    destroy_moonep_buffer()
    Utils.destroy_model_parallel()
