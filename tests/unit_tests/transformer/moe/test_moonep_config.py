# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static (config-construction) validation of the MoonEP flex dispatcher backend."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.moonep_backend import uses_moonep
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(importlib.util.find_spec("moonep") is None, reason="moonep not installed"),
]

PREFIX = "moe_flex_dispatcher_backend='moonep' requires "


def moonep_config_kwargs(**overrides):
    """A valid phase-1 MoonEP config (EP=8, TP=1, bf16, SwiGLU, no bias, fp32 router)."""
    kwargs = dict(
        num_layers=2,
        hidden_size=1024,
        num_attention_heads=8,
        num_moe_experts=16,
        moe_ffn_hidden_size=1024,
        moe_router_topk=4,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="moonep",
        bf16=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        moe_router_dtype="fp32",
        moe_moonep_max_tokens_per_rank=512,
        use_cpu_initialization=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_config():
    config = TransformerConfig(**moonep_config_kwargs())
    assert uses_moonep(config)
    assert config.moe_moonep_token_padding == 128
    assert config.moe_moonep_strict_zero_grad is False


def test_other_backends_unaffected():
    config = TransformerConfig(
        **moonep_config_kwargs(
            moe_flex_dispatcher_backend="deepep", add_bias_linear=True, moe_router_dtype=None
        )
    )
    assert not uses_moonep(config)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(moe_token_dispatcher_type="alltoall"), "moe_token_dispatcher_type='flex'"),
        (dict(expert_model_parallel_size=1), "1 < expert_model_parallel_size <= 128"),
        (dict(num_moe_experts=12), "num_moe_experts divisible by expert_model_parallel_size"),
        (dict(expert_tensor_parallel_size=2), "expert_tensor_parallel_size == 1"),
        (
            dict(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16),
            "pipeline_model_parallel_size == 1",
        ),
        (dict(bf16=False, params_dtype=torch.float32), "bf16=True and fp16=False"),
        (dict(bf16=False, fp16=True, params_dtype=torch.float16), "bf16=True and fp16=False"),
        (dict(gated_linear_unit=False), "SwiGLU experts"),
        (dict(activation_func=F.gelu), "SwiGLU experts"),
        (dict(glu_linear_offset=1.0), "a plain SwiGLU"),
        (dict(activation_func_clamp_value=7.0), "a plain SwiGLU"),
        (dict(add_bias_linear=True), "add_bias_linear=False"),
        (dict(moe_expert_capacity_factor=1.0), "dropless routing"),
        (
            dict(moe_expert_capacity_factor=1.0, moe_pad_expert_input_to_capacity=True),
            "dropless routing",
        ),
        (dict(moe_router_dtype=None), "moe_router_dtype='fp32'"),
        (dict(fp8="e4m3"), r"bf16 expert compute \(no fp8/fp4\)"),
        (dict(moe_latent_size=256), "no MoE latent projections"),
        (
            dict(moe_shared_expert_intermediate_size=1024, moe_shared_expert_overlap=True),
            "moe_shared_expert_overlap=False",
        ),
        (dict(overlap_moe_expert_parallel_comm=True), "no EP communication overlap"),
        (dict(cuda_graph_impl="local"), "cuda_graph_impl='none'"),
        (
            dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1),
            "no MoE activation recompute",
        ),
        (
            dict(recompute_granularity="selective", recompute_modules=["moe"]),
            "no MoE activation recompute",
        ),
        (dict(moe_layer_recompute=True), "no MoE activation recompute"),
        (dict(variable_seq_lengths=True), "the same token count on every EP rank"),
        (dict(mlp_chunks_for_training=2), "one call per MoE layer per micro-batch"),
        (dict(mtp_num_layers=1, mtp_use_repeated_layer=True), "one call per MoE layer"),
        (dict(cpu_offloading=True, cpu_offloading_num_layers=1), "no activation offloading"),
        (dict(init_model_with_meta_device=True), "GPU initialization at construction"),
        (dict(use_cpu_initialization=True), "GPU initialization at construction"),
        (dict(mtp_num_layers=1, freeze_base_model_for_mtp=True), "freeze_base_model_for_mtp"),
        (dict(hidden_size=1000), "hidden_size and moe_ffn_hidden_size to be multiples of 128"),
        (dict(moe_ffn_hidden_size=1000), "hidden_size and moe_ffn_hidden_size to be multiples"),
        (dict(num_moe_experts=64, moe_router_topk=33), r"1 <= moe_router_topk <= 32"),
        (dict(moe_moonep_max_tokens_per_rank=0), "moe_moonep_max_tokens_per_rank None or > 0"),
        (dict(moe_moonep_token_padding=0), "moe_moonep_token_padding >= 1"),
    ],
)
def test_forbidden_overrides(overrides, match):
    with pytest.raises(ValueError, match=PREFIX + match):
        TransformerConfig(**moonep_config_kwargs(**overrides))


def test_moe_enable_deepep_conflict():
    """The deprecated flag would otherwise silently rewrite the backend to 'deepep'."""
    with pytest.raises(ValueError, match="moe_enable_deepep forces the 'deepep' flex backend"):
        TransformerConfig(**moonep_config_kwargs(moe_enable_deepep=True))


def test_missing_package(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "moonep":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(ValueError, match=PREFIX + "the moonep package"):
        TransformerConfig(**moonep_config_kwargs())
