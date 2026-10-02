# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""rope_freqs_per_token: one rotary frequency row per packed (THD) token."""

import inspect

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("transformer_engine")

from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from tests.unit_tests.transformer.dit_attention_parity_utils import (  # noqa: E402
    DiTMaskedAttention,
    DiTSelfAttention,
)

try:
    from transformer_engine.pytorch.attention.rope import (
        apply_rotary_pos_emb as te_apply_rotary_pos_emb,
    )
except ImportError:  # Transformer Engine < 2.3 keeps the function elsewhere.
    te_apply_rotary_pos_emb = None

# Not every Transformer Engine version takes per-sequence start positions for THD RoPE.
TE_ROPE_HAS_START_POSITIONS = te_apply_rotary_pos_emb is not None and (
    "start_positions" in inspect.signature(te_apply_rotary_pos_emb).parameters
)

HEADS, HEAD_DIM = 4, 64
LAYOUTS = {"small": [98, 177, 25], "large": [290, 465, 73]}


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
    )
    options.update(kwargs)
    return TransformerConfig(**options)


def _cu_seqlens(lengths):
    return torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )


def _per_token_freqs(total, seed=3):
    """Per-token angles that do not restart per sequence (e.g. multi-axis coordinates)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(total, 1, 1, HEAD_DIM, device="cuda", generator=generator) * 600


def _tokens(total, dtype, seed=4):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = torch.randn(total, HEADS, HEAD_DIM, device="cuda", generator=generator).to(dtype)
    grad = torch.randn(total, HEADS, HEAD_DIM, device="cuda", generator=generator).to(dtype)
    return tokens, grad


def _forward_backward(function, tokens, grad):
    leaf = tokens.detach().clone().requires_grad_()
    output = function(leaf)
    output.backward(grad)
    return output.detach(), leaf.grad


def _bits(tensor):
    integer = {2: torch.int16, 4: torch.int32}
    return tensor.detach().contiguous().view(integer[tensor.element_size()])


def _assert_bitwise(actual, expected, what):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, what
    mismatches = (_bits(actual) != _bits(expected)).sum().item()
    assert mismatches == 0, f"{what}: {mismatches} of {actual.numel()} elements differ"


class TestRopeFreqsPerToken:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.cp_group = parallel_state.get_context_parallel_group()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    @pytest.mark.parametrize("fused", [True, False])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_equals_sbhd_view(self, layout, fused, dtype):
        """THD per-token RoPE == RoPE of the [t, 1, h, d] view (the hand-written packed path)."""
        lengths = LAYOUTS[layout]
        cu_seqlens = _cu_seqlens(lengths)
        total = sum(lengths)
        freqs = _per_token_freqs(total)
        tokens, grad = _tokens(total, dtype)
        per_token = _config(apply_rope_fusion=fused, rope_freqs_per_token=True)
        reference = _config(apply_rope_fusion=fused)

        actual = _forward_backward(
            lambda t: apply_rotary_pos_emb(
                t,
                freqs,
                per_token,
                cu_seqlens=cu_seqlens,
                cp_group=self.cp_group,
                max_seqlen=max(lengths),
            ),
            tokens,
            grad,
        )
        expected = _forward_backward(
            lambda t: apply_rotary_pos_emb(
                t.unsqueeze(1), freqs, reference, cp_group=self.cp_group
            ).squeeze(1),
            tokens,
            grad,
        )
        for name, a, e in zip(("output", "grad"), actual, expected):
            _assert_bitwise(a, e, f"{layout} fused={fused} {dtype} {name}")

    @pytest.mark.skipif(
        not TE_ROPE_HAS_START_POSITIONS,
        reason="Transformer Engine's apply_rotary_pos_emb takes no start_positions",
    )
    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_fused_equals_te_thd_with_start_positions(self, layout, dtype):
        lengths = LAYOUTS[layout]
        cu_seqlens = _cu_seqlens(lengths)
        total = sum(lengths)
        freqs = _per_token_freqs(total)
        tokens, grad = _tokens(total, dtype)
        config = _config(apply_rope_fusion=True, rope_freqs_per_token=True)
        actual = _forward_backward(
            lambda t: apply_rotary_pos_emb(
                t, freqs, config, cu_seqlens=cu_seqlens, cp_group=self.cp_group
            ),
            tokens,
            grad,
        )
        expected = _forward_backward(
            lambda t: te_apply_rotary_pos_emb(
                t,
                freqs,
                tensor_format="thd",
                fused=True,
                cu_seqlens=cu_seqlens,
                start_positions=cu_seqlens[:-1],
            ),
            tokens,
            grad,
        )
        for name, a, e in zip(("output", "grad"), actual, expected):
            _assert_bitwise(a, e, f"{layout} {dtype} {name}")

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_unfused_positions_equal_stock_unfused_thd(self, dtype):
        """The stock unfused THD path already indexes a token-sized table per token."""
        lengths = LAYOUTS["small"]
        cu_seqlens = _cu_seqlens(lengths)
        total = sum(lengths)
        freqs = _per_token_freqs(total)
        tokens, grad = _tokens(total, dtype)
        results = []
        for per_token in (True, False):
            config = _config(apply_rope_fusion=False, rope_freqs_per_token=per_token)
            results.append(
                _forward_backward(
                    lambda t: apply_rotary_pos_emb(
                        t,
                        freqs,
                        config,
                        cu_seqlens=cu_seqlens,
                        cp_group=self.cp_group,
                        max_seqlen=max(lengths),
                    ),
                    tokens,
                    grad,
                )
            )
        for name, a, e in zip(("output", "grad"), *results):
            _assert_bitwise(a, e, f"{dtype} {name}")

    def test_default_fused_thd_restarts_positions(self):
        """Without the flag, fused THD RoPE indexes positions from each sequence start."""
        lengths = LAYOUTS["small"]
        cu_seqlens = _cu_seqlens(lengths)
        total = sum(lengths)
        freqs = _per_token_freqs(total)
        tokens, _ = _tokens(total, torch.float32)
        default = apply_rotary_pos_emb(
            tokens, freqs, _config(apply_rope_fusion=True), cu_seqlens=cu_seqlens
        )
        per_token = apply_rotary_pos_emb(
            tokens,
            freqs,
            _config(apply_rope_fusion=True, rope_freqs_per_token=True),
            cu_seqlens=cu_seqlens,
        )
        restart = torch.cat(
            [
                apply_rotary_pos_emb(
                    chunk.unsqueeze(1), freqs[: len(chunk)], _config(apply_rope_fusion=True)
                ).squeeze(1)
                for chunk in tokens.split(lengths)
            ]
        )
        assert torch.equal(default, restart)
        first = lengths[0]
        assert torch.equal(per_token[:first], default[:first])
        assert not torch.equal(per_token[first:], default[first:])

    def test_requires_context_parallel_size_1(self):
        with pytest.raises(ValueError, match="rope_freqs_per_token"):
            _config(rope_freqs_per_token=True, context_parallel_size=2)

    @pytest.mark.parametrize("fused", [True, False])
    @pytest.mark.parametrize(
        "mask_type", [AttnMaskType.key_padding, AttnMaskType.arbitrary, AttnMaskType.padding]
    )
    def test_self_attention_thd_bitwise_equal_to_hand_written_packed_branch(
        self, fused, mask_type, monkeypatch
    ):
        """Stock SelfAttention (THD, per-token RoPE, bf16 gate) == the hand-written packed path."""
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        lengths = LAYOUTS["small"]
        cu_seqlens = _cu_seqlens(lengths)
        total = sum(lengths)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        options = dict(
            attention_output_gate=True,
            apply_rope_fusion=fused,
            qk_layernorm=True,
            normalization="RMSNorm",
            add_bias_linear=False,
        )
        submodules = dict(
            linear_qkv=TEColumnParallelLinear,
            linear_proj=TERowParallelLinear,
            q_layernorm=TENorm,
            k_layernorm=TENorm,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        reference_config = _config(**options)
        reference = DiTSelfAttention(
            config=reference_config,
            submodules=SelfAttentionSubmodules(core_attention=DiTMaskedAttention, **submodules),
            layer_number=1,
            attn_mask_type=AttnMaskType.arbitrary,
            pg_collection=pg_collection,
        )
        for parameter in reference.parameters():
            if parameter.dim() == 1:  # q/k RMSNorm weights
                parameter.data.normal_(1.0, 0.1)
            else:
                parameter.data.normal_(0, 0.05)
        native = SelfAttention(
            config=_config(
                rope_freqs_per_token=True, attention_output_gate_in_fp32=False, **options
            ),
            submodules=SelfAttentionSubmodules(core_attention=TEDotProductAttention, **submodules),
            layer_number=1,
            attn_mask_type=mask_type,
            pg_collection=pg_collection,
        )
        native.load_state_dict(reference.state_dict())

        generator = torch.Generator(device="cuda").manual_seed(5)
        hidden = torch.randn(total, 1, HEADS * HEAD_DIM, device="cuda", generator=generator).to(
            torch.bfloat16
        )
        grad_output = torch.randn(hidden.shape, device="cuda", generator=generator).to(hidden.dtype)
        freqs = _per_token_freqs(total)
        results = []
        for module in (native, reference):
            module.zero_grad(set_to_none=True)
            leaf = hidden.detach().clone().requires_grad_()
            output, bias = module(
                leaf, None, rotary_pos_emb=freqs, packed_seq_params=packed_seq_params
            )
            assert bias is None
            output.backward(grad_output)
            results.append(
                [output.detach(), leaf.grad]
                + [p.grad for _, p in sorted(module.named_parameters())]
            )
        names = ["output", "dhidden"] + [name for name, _ in sorted(native.named_parameters())]
        for name, actual, expected in zip(names, *results):
            _assert_bitwise(actual, expected, f"fused={fused} {mask_type} {name}")
