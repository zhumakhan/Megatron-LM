# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""attention_output_gate_in_fp32: fused fp32 output gate (default) or eager input-dtype gate."""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import Attention, SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("transformer_engine")

from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)

HEADS, HEAD_DIM, SEQUENCE, BATCH = 4, 64, 96, 3


def _config(**kwargs):
    options = dict(
        num_layers=1,
        hidden_size=HEADS * HEAD_DIM,
        num_attention_heads=HEADS,
        kv_channels=HEAD_DIM,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_output_gate=True,
        add_bias_linear=False,
    )
    options.update(kwargs)
    return TransformerConfig(**options)


class InputDtypeGateSelfAttention(SelfAttention):
    """SelfAttention with the output gate overridden to the eager input-dtype expression."""

    def _apply_output_gate(self, x, gate):
        return x * torch.sigmoid(gate.reshape_as(x))


def _attention(config, cls=SelfAttention):
    return cls(
        config=config,
        submodules=SelfAttentionSubmodules(
            linear_qkv=TEColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
        layer_number=1,
        attn_mask_type=AttnMaskType.no_mask,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    )


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.int16)


def _run(module, hidden, grad_output):
    module.zero_grad(set_to_none=True)
    leaf = hidden.detach().clone().requires_grad_()
    output, _ = module(leaf, None)
    output.backward(grad_output)
    return [output.detach(), leaf.grad] + [p.grad for _, p in sorted(module.named_parameters())]


class TestAttentionOutputGatePrecision:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_is_fp32(self):
        config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
        assert config.attention_output_gate_in_fp32 is True

    def test_input_dtype_gate_is_eager_expression(self):
        attention = _attention(_config(attention_output_gate_in_fp32=False))
        generator = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(SEQUENCE, BATCH, HEADS * HEAD_DIM, device="cuda", generator=generator)
        # The gate arrives as a non-contiguous [sq, b, np, hn] slice of the fused QKV(G) output.
        fused = torch.randn(
            SEQUENCE, BATCH, HEADS, 4 * HEAD_DIM, device="cuda", generator=generator
        ).to(torch.bfloat16)
        gate = fused[..., HEAD_DIM : 2 * HEAD_DIM] * 4
        x = x.to(torch.bfloat16)
        actual = attention._apply_output_gate_in_input_dtype(x, gate)
        expected = x * torch.sigmoid(gate.reshape_as(x))
        assert actual.dtype == torch.bfloat16
        assert torch.equal(_bits(actual), _bits(expected))
        # The fused fp32 gate rounds differently (sigmoid is not rounded to bf16 first).
        assert not torch.equal(attention._apply_output_gate(x, gate), actual)

    def test_flag_false_bitwise_equal_to_input_dtype_gate_override(self, monkeypatch):
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        native = _attention(_config(attention_output_gate_in_fp32=False))
        reference = _attention(_config(), cls=InputDtypeGateSelfAttention)
        for parameter in reference.parameters():
            parameter.data.normal_(0, 0.05)
        native.load_state_dict(reference.state_dict())
        generator = torch.Generator(device="cuda").manual_seed(1)
        hidden = torch.randn(
            SEQUENCE, BATCH, HEADS * HEAD_DIM, device="cuda", generator=generator
        ).to(torch.bfloat16)
        grad_output = torch.randn(hidden.shape, device="cuda", generator=generator).to(hidden.dtype)
        names = ["output", "dhidden"] + [name for name, _ in sorted(native.named_parameters())]
        for name, actual, expected in zip(
            names, _run(native, hidden, grad_output), _run(reference, hidden, grad_output)
        ):
            mismatches = (_bits(actual) != _bits(expected)).sum().item()
            assert mismatches == 0, f"{name}: {mismatches} elements differ"

    @pytest.mark.parametrize("in_fp32", [True, False])
    def test_flag_selects_gate(self, in_fp32, monkeypatch):
        calls = []
        for name in ("_apply_output_gate", "_apply_output_gate_in_input_dtype"):
            original = getattr(Attention, name)
            monkeypatch.setattr(
                Attention,
                name,
                lambda self, x, gate, name=name, original=original: calls.append(name)
                or original(self, x, gate),
            )
        attention = _attention(_config(attention_output_gate_in_fp32=in_fp32))
        hidden = torch.randn(SEQUENCE, BATCH, HEADS * HEAD_DIM, device="cuda").to(torch.bfloat16)
        attention(hidden, None)
        assert calls == (
            ["_apply_output_gate"] if in_fp32 else ["_apply_output_gate_in_input_dtype"]
        )
