# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cp_comm_type='ulysses': Megatron-side all-to-all context parallelism around a non-CP kernel."""

import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    clear_key_padding_metadata_cache,
    ulysses_attention,
    ulysses_inner_config,
)
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
    dit_key_padding_mask,
)

HEADS, HEAD_DIM, BATCH = 8, 64, 4
REGISTERS, CAPTION_SLOTS, VIDEO_TOKENS = 4, 28, 64
SEQUENCE = REGISTERS + CAPTION_SLOTS + VIDEO_TOKENS
MASK_TYPES = ["key_padding", "padding", "no_mask", "arbitrary"]


def _config(tp=1, cp=1, **kwargs):
    options = dict(
        num_layers=1,
        hidden_size=HEADS * HEAD_DIM,
        num_attention_heads=HEADS,
        kv_channels=HEAD_DIM,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    options.update(kwargs)
    return TransformerConfig(**options)


def _pg_collection():
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])


def _global_mask(mask_type):
    """CP-global attention mask (True = masked) for every mask type."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    if mask_type == "key_padding":
        mask = dit_key_padding_mask([0, 13, 28], REGISTERS, CAPTION_SLOTS, VIDEO_TOKENS)
        return torch.cat((mask, torch.ones_like(mask[:1])))  # plus a sample without valid keys
    if mask_type == "padding":
        lengths = torch.tensor([SEQUENCE, 50, 71, 1], device="cuda")
        return (torch.arange(SEQUENCE, device="cuda") >= lengths[:, None])[:, None, None, :]
    if mask_type == "arbitrary":
        mask = torch.rand(BATCH, 1, SEQUENCE, SEQUENCE, device="cuda", generator=generator) < 0.4
        return mask & ~torch.eye(SEQUENCE, dtype=torch.bool, device="cuda")
    return None


def _global_inputs(heads, dtype=torch.bfloat16, seed=11):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors = [
        torch.randn(SEQUENCE, BATCH, heads, HEAD_DIM, device="cuda", generator=generator).to(dtype)
        for _ in range(3)
    ]
    grad_output = torch.randn(
        SEQUENCE, BATCH, heads * HEAD_DIM, device="cuda", generator=generator
    ).to(dtype)
    return tensors, grad_output


def _shard(tensor):
    """Contiguous sequence shard of this context-parallel rank."""
    cp_size = parallel_state.get_context_parallel_world_size()
    return tensor.chunk(cp_size, dim=0)[parallel_state.get_context_parallel_rank()]


def _run(attention, inputs, grad_output, mask, **kwargs):
    leaves = [t.detach().clone().requires_grad_() for t in inputs]
    output = attention(*leaves, mask, **kwargs)
    output.backward(grad_output)
    return [output.detach()] + [leaf.grad for leaf in leaves]


def _bits(tensor):
    integer = {2: torch.int16, 4: torch.int32}
    return tensor.detach().contiguous().view(integer[tensor.element_size()])


def _assert_bitwise(actual, expected, what):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, what
    mismatches = (_bits(actual) != _bits(expected)).sum().item()
    assert mismatches == 0, f"{what}: {mismatches} of {actual.numel()} elements differ"


def _te_attention(config, mask_type, cp_comm_type=None):
    return TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=mask_type,
        attention_type="self",
        cp_comm_type=cp_comm_type,
        pg_collection=_pg_collection(),
    )


class TestUlyssesAttention:
    def teardown_method(self, method):
        clear_key_padding_metadata_cache()
        Utils.destroy_model_parallel()

    def _initialize(self, tp, cp, monkeypatch):
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        Utils.initialize_model_parallel(tp, 1, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(123)

    @pytest.mark.parametrize("tp,cp", [(1, 2), (1, 4), (2, 2)])
    @pytest.mark.parametrize("mask_type", MASK_TYPES)
    def test_matches_full_sequence_attention(self, tp, cp, mask_type, monkeypatch):
        """Ulysses on contiguous shards equals TE attention over the full sequence, bitwise.

        Each head is computed by the same kernel on the same full-sequence inputs, so the
        all-to-all exchange must not change any bit of the output or of dq/dk/dv.
        """
        self._initialize(tp, cp, monkeypatch)
        attn_mask_type = AttnMaskType[mask_type]
        mask = _global_mask(mask_type)
        inputs, grad_output = _global_inputs(HEADS // tp)
        reference = _run(
            _te_attention(_config(tp, 1), attn_mask_type),
            inputs,
            grad_output,
            mask,
            attn_mask_type=attn_mask_type,
        )
        ulysses = _run(
            _te_attention(_config(tp, cp, cp_comm_type="ulysses"), attn_mask_type, "ulysses"),
            [_shard(t) for t in inputs],
            _shard(grad_output),
            mask,
            attn_mask_type=attn_mask_type,
        )
        for name, actual, expected in zip(("output", "dq", "dk", "dv"), ulysses, reference):
            expected = _shard(expected)
            if mask_type == "padding":
                # TE's padding also masks queries past each sample's length; compare valid rows.
                valid = ~_shard(mask[:, 0, 0].transpose(0, 1))
                actual, expected = actual[valid], expected[valid]
            _assert_bitwise(actual, expected, f"{mask_type} tp={tp} cp={cp} {name}")

    @pytest.mark.parametrize("cp", [2, 4])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_bitwise_equal_to_hand_written_ulysses(self, cp, dtype, monkeypatch):
        """Ulysses + key_padding equals the hand-written all-to-all + varlen attention."""
        self._initialize(1, cp, monkeypatch)
        mask = _global_mask("key_padding")
        inputs, grad_output = _global_inputs(HEADS, dtype)
        inputs, grad_output = [_shard(t) for t in inputs], _shard(grad_output)
        options = dict(bf16=dtype == torch.bfloat16, params_dtype=dtype)
        native = _run(
            _te_attention(
                _config(1, cp, cp_comm_type="ulysses", **options),
                AttnMaskType.key_padding,
                "ulysses",
            ),
            inputs,
            grad_output,
            mask,
            attn_mask_type=AttnMaskType.key_padding,
        )
        config = _config(1, cp, **options)
        reference = _run(
            DiTMaskedAttention(
                attention_type="self",
                config=config,
                pg_collection=_pg_collection(),
                layer_number=1,
                attn_mask_type=AttnMaskType.arbitrary,
            ),
            inputs,
            grad_output,
            mask,
            attn_mask_type=AttnMaskType.arbitrary,
        )
        for name, actual, expected in zip(("output", "dq", "dk", "dv"), native, reference):
            _assert_bitwise(actual, expected, f"{dtype} cp={cp} {name}")

    @pytest.mark.parametrize("cp", [2, 4])
    @pytest.mark.parametrize("mask_type", ["key_padding", "arbitrary", "no_mask"])
    def test_local_dot_product_attention(self, cp, mask_type, monkeypatch):
        self._initialize(1, cp, monkeypatch)
        attn_mask_type = AttnMaskType[mask_type]
        mask = _global_mask(mask_type)
        inputs, grad_output = _global_inputs(HEADS)

        def local_attention(config, cp_comm_type=None):
            return DotProductAttention(
                config,
                layer_number=1,
                attn_mask_type=attn_mask_type,
                attention_type="self",
                cp_comm_type=cp_comm_type,
                pg_collection=_pg_collection(),
            )

        reference = _run(
            local_attention(_config(1, 1)),
            inputs,
            grad_output,
            None if mask is None else mask.clone(),
            attn_mask_type=attn_mask_type,
        )
        ulysses = _run(
            local_attention(_config(1, cp, cp_comm_type="ulysses"), "ulysses"),
            [_shard(t) for t in inputs],
            _shard(grad_output),
            None if mask is None else mask.clone(),
            attn_mask_type=attn_mask_type,
        )
        for name, actual, expected in zip(("output", "dq", "dk", "dv"), ulysses, reference):
            _assert_bitwise(actual, _shard(expected), f"{mask_type} cp={cp} {name}")

    @pytest.mark.parametrize("cp", [2, 4])
    def test_self_attention_bitwise_equal_to_hand_written_dit_attention(self, cp, monkeypatch):
        """Stock SelfAttention + key_padding + ulysses + bf16 output gate == hand-written DiT."""
        self._initialize(1, cp, monkeypatch)
        options = dict(
            attention_output_gate=True,
            apply_rope_fusion=True,
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
        reference_config = _config(1, cp, **options)
        reference = DiTSelfAttention(
            config=reference_config,
            submodules=SelfAttentionSubmodules(core_attention=DiTMaskedAttention, **submodules),
            layer_number=1,
            attn_mask_type=AttnMaskType.arbitrary,
            pg_collection=_pg_collection(),
        )
        cp_group = parallel_state.get_context_parallel_group()
        for parameter in reference.parameters():
            if parameter.dim() == 1:  # q/k RMSNorm weights
                parameter.data.normal_(1.0, 0.1)
            else:
                parameter.data.normal_(0, 0.05)
            torch.distributed.broadcast(
                parameter.data, src=torch.distributed.get_global_rank(cp_group, 0), group=cp_group
            )
        native = SelfAttention(
            config=_config(
                1, cp, cp_comm_type="ulysses", attention_output_gate_in_fp32=False, **options
            ),
            submodules=SelfAttentionSubmodules(core_attention=TEDotProductAttention, **submodules),
            layer_number=1,
            attn_mask_type=AttnMaskType.key_padding,
            cp_comm_type="ulysses",
            pg_collection=_pg_collection(),
        )
        native.load_state_dict(reference.state_dict())

        generator = torch.Generator(device="cuda").manual_seed(5)
        hidden = torch.randn(
            SEQUENCE, BATCH, HEADS * HEAD_DIM, device="cuda", generator=generator
        ).to(torch.bfloat16)
        grad_output = torch.randn(hidden.shape, device="cuda", generator=generator).to(hidden.dtype)
        frequencies = torch.rand(SEQUENCE, 1, 1, HEAD_DIM, device="cuda", generator=generator) * 50
        mask = _global_mask("key_padding")

        results = []
        for module in (native, reference):
            module.zero_grad(set_to_none=True)
            leaf = _shard(hidden).detach().clone().requires_grad_()
            output, bias = module(leaf, mask, rotary_pos_emb=_shard(frequencies))
            assert bias is None
            output.backward(_shard(grad_output))
            results.append(
                [output.detach(), leaf.grad]
                + [p.grad for _, p in sorted(module.named_parameters())]
            )
        names = ["output", "dhidden"] + [name for name, _ in sorted(native.named_parameters())]
        for name, actual, expected in zip(names, *results):
            _assert_bitwise(actual, expected, f"cp={cp} {name}")

    def test_config_validation(self):
        with pytest.raises(ValueError, match="num_attention_heads"):
            _config(2, 4, hidden_size=768, num_attention_heads=12, cp_comm_type="ulysses")
        with pytest.raises(ValueError, match="num_query_groups"):
            _config(1, 4, num_query_groups=2, cp_comm_type="ulysses")
        with pytest.raises(ValueError, match="learnable"):
            _config(1, 2, cp_comm_type="ulysses", softmax_type="learnable")
        with pytest.raises(ValueError, match="qk_clip"):
            _config(1, 2, cp_comm_type="ulysses", log_max_attention_logit=True)
        # Per-layer lists are validated too; other CP types keep their behaviour.
        with pytest.raises(ValueError, match="num_attention_heads"):
            _config(1, 16, num_layers=2, cp_comm_type=["p2p", "ulysses"])
        _config(1, 4, cp_comm_type="ulysses")
        _config(1, 16, num_attention_heads=8, num_layers=2, cp_comm_type=["p2p", "a2a"])

    @pytest.mark.parametrize("mask_type", ["causal", "padding_causal", "causal_bottom_right"])
    def test_rejects_causal_masks(self, mask_type, monkeypatch):
        """Ulysses needs contiguous shards, but Megatron shards causal CP inputs zig-zag."""
        self._initialize(1, 2, monkeypatch)
        attn_mask_type = AttnMaskType[mask_type]
        config = _config(1, 2, cp_comm_type="ulysses")

        def local_attention(mask_type):
            return DotProductAttention(
                config,
                layer_number=1,
                attn_mask_type=mask_type,
                attention_type="self",
                cp_comm_type="ulysses",
                pg_collection=_pg_collection(),
            )

        with pytest.raises(ValueError, match="causal"):
            _te_attention(config, attn_mask_type, "ulysses")
        with pytest.raises(ValueError, match="causal"):
            local_attention(attn_mask_type)
        # Per-call causal masks on modules built for a non-causal mask type.
        inputs, _ = _global_inputs(HEADS)
        shards = [_shard(t) for t in inputs]
        for attention in (
            _te_attention(config, AttnMaskType.no_mask, "ulysses"),
            local_attention(AttnMaskType.no_mask),
        ):
            with pytest.raises(ValueError, match="causal"):
                attention(*shards, None, attn_mask_type=attn_mask_type)
        # Without context parallelism, cp_comm_type='ulysses' leaves causal attention alone.
        _te_attention(_config(1, 1, cp_comm_type="ulysses"), attn_mask_type, "ulysses")

    def test_rejects_cross_attention(self, monkeypatch):
        """Ulysses exchanges keys and values like the queries, which fits self-attention only."""
        self._initialize(1, 2, monkeypatch)

        def cross_attention(core_attention, cp):
            return core_attention(
                config=_config(1, cp, cp_comm_type="ulysses"),
                layer_number=1,
                attn_mask_type=AttnMaskType.no_mask,
                attention_type="cross",
                cp_comm_type="ulysses",
                pg_collection=_pg_collection(),
            )

        for core_attention in (TEDotProductAttention, DotProductAttention):
            with pytest.raises(ValueError, match="self-attention only"):
                cross_attention(core_attention, cp=2)
            # Without context parallelism, cross-attention is built as usual.
            cross_attention(core_attention, cp=1)

    def test_key_padding_rejects_te_context_parallelism(self, monkeypatch):
        """key_padding uses cp_comm_type='ulysses' for CP; TE's own CP cannot take the mask."""
        self._initialize(1, 2, monkeypatch)
        attention = _te_attention(
            _config(1, 2, cp_comm_type="p2p"), AttnMaskType.key_padding, "p2p"
        )
        inputs, _ = _global_inputs(HEADS)
        with pytest.raises(ValueError, match="ulysses"):
            attention(
                *[_shard(t) for t in inputs],
                _global_mask("key_padding"),
                attn_mask_type=AttnMaskType.key_padding,
            )

    def test_inner_kernel_heads(self, monkeypatch):
        self._initialize(2, 2, monkeypatch)
        config = _config(2, 2, cp_comm_type="ulysses")
        attention = _te_attention(config, AttnMaskType.key_padding, "ulysses")
        assert attention.ulysses_cp_group is parallel_state.get_context_parallel_group()
        assert attention.config.context_parallel_size == 1
        assert attention.config.num_attention_heads == HEADS // 2
        assert attention.num_attention_heads == HEADS // 2  # TE: global heads of the kernel
        assert config.num_attention_heads == HEADS and config.context_parallel_size == 2
        with pytest.raises(ValueError, match="packed"):
            from megatron.core.packed_seq_params import PackedSeqParams

            cu_seqlens = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
            query = torch.randn(8, HEADS // 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            attention(
                query,
                query,
                query,
                None,
                attn_mask_type=AttnMaskType.padding,
                packed_seq_params=PackedSeqParams(
                    qkv_format="thd", cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens
                ),
            )
        # The all-to-alls exchange only q/k/v, so an attention_bias is rejected.
        query = torch.randn(8, 1, HEADS // 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros(1, HEADS // 2, 16, 16, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="does not support attention_bias"):
            attention(
                query, query, query, None, attn_mask_type=AttnMaskType.no_mask, attention_bias=bias
            )
        # Every input of ulysses_attention needs a number of heads divisible by cp.
        query = torch.randn(8, 1, 3, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="3 heads cannot be split across 2 CP ranks"):
            ulysses_attention(
                attention._forward_local,
                query,
                query,
                query,
                attention.ulysses_cp_group,
                None,
                AttnMaskType.no_mask,
            )


def test_inner_config_validation():
    """ulysses_inner_config needs a context-parallel group that splits the heads evenly."""
    config = _config(1, 1)
    with pytest.raises(ValueError, match="context-parallel group"):
        ulysses_inner_config(config, None)
    three_ranks = types.SimpleNamespace(size=lambda: 3)
    with pytest.raises(ValueError, match="num_attention_heads"):
        ulysses_inner_config(config, three_ranks)
    inner = ulysses_inner_config(config, types.SimpleNamespace(size=lambda: 2))
    assert inner.num_attention_heads == HEADS // 2 and inner.context_parallel_size == 1
    assert config.num_attention_heads == HEADS
