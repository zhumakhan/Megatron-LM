# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AttnMaskType.key_padding: [b, 1, 1, sk] key masks with holes."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import utils as transformer_utils
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    attention_mask_func,
    clear_key_padding_metadata_cache,
    get_key_padding_metadata,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("transformer_engine")

from megatron.core.extensions.transformer_engine import TEDotProductAttention  # noqa: E402
from tests.unit_tests.transformer.dit_attention_parity_utils import (  # noqa: E402
    DiTMaskedAttention,
    dit_key_padding_mask,
)

HEADS, HEAD_DIM = 4, 64
REGISTERS, CAPTION_SLOTS, VIDEO_TOKENS = 4, 28, 48
SEQUENCE = REGISTERS + CAPTION_SLOTS + VIDEO_TOKENS
# [registers | caption | caption padding | video] with 0, 7, 20 and all 28 caption tokens valid,
# plus a sample without any valid key.
CAPTION_LENGTHS = [0, 7, 20, 28]


def _mask(device="cuda"):
    mask = dit_key_padding_mask(CAPTION_LENGTHS, REGISTERS, CAPTION_SLOTS, VIDEO_TOKENS, device)
    empty = torch.ones(1, 1, 1, SEQUENCE, dtype=torch.bool, device=device)
    return torch.cat((mask, empty))


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


def _pg_collection():
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])


def _te_attention(config, attn_mask_type=AttnMaskType.key_padding):
    return TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=attn_mask_type,
        attention_type="self",
        pg_collection=_pg_collection(),
    )


def _reference_attention(config):
    return DiTMaskedAttention(
        attention_type="self",
        config=config,
        pg_collection=_pg_collection(),
        layer_number=1,
        attn_mask_type=AttnMaskType.arbitrary,
    )


def _inputs(batch, dtype, seed=1234, length=SEQUENCE):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(length, batch, HEADS, HEAD_DIM, device="cuda", generator=generator)
        .to(dtype)
        .requires_grad_()
        for _ in range(3)
    ]


def _sdpa_reference(query, key, value, mask):
    """torch SDPA in fp64 with the key mask; samples without a valid key give zero rows."""
    q, k, v = (
        t.detach().double().permute(1, 2, 0, 3).requires_grad_() for t in (query, key, value)
    )
    # A fully masked row has no softmax: let those samples attend everywhere, then zero them.
    empty = mask.all(dim=-1, keepdim=True)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=~(mask & ~empty))
    output = output.masked_fill(empty, 0).permute(2, 0, 1, 3).flatten(2)
    return output, (q, k, v)


def _relative_error(actual, expected):
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def _bits(tensor):
    integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return tensor.detach().contiguous().view(integer[tensor.element_size()])


def _assert_bitwise(actual, expected, what):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, what
    mismatches = (_bits(actual) != _bits(expected)).sum().item()
    assert mismatches == 0, f"{what}: {mismatches} of {actual.numel()} elements differ"


def _run(attention, inputs, mask, grad_output, **kwargs):
    query, key, value = (t.detach().clone().requires_grad_() for t in inputs)
    output = attention(query, key, value, mask, **kwargs)
    output.backward(grad_output)
    return output.detach(), query.grad, key.grad, value.grad


class TestKeyPaddingAttention:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        clear_key_padding_metadata_cache()

    def teardown_method(self, method):
        clear_key_padding_metadata_cache()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_matches_sdpa_reference(self, dtype):
        """Varlen (bf16) and dense-fallback (fp32) paths against fp64 torch SDPA."""
        mask = _mask()
        inputs = _inputs(mask.shape[0], dtype)
        grad_output = torch.randn(
            SEQUENCE, mask.shape[0], HEADS * HEAD_DIM, device="cuda", dtype=dtype
        )
        attention = _te_attention(_config(bf16=dtype == torch.bfloat16, params_dtype=dtype))
        output, dq, dk, dv = _run(
            attention, inputs, mask, grad_output, attn_mask_type=AttnMaskType.key_padding
        )

        expected, leaves = _sdpa_reference(*inputs, mask)
        expected.backward(grad_output.double())
        # fp32 matmuls may run in TF32 (e.g. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 in NGC images).
        tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-3
        errors = [_relative_error(output, expected)] + [
            _relative_error(grad, leaf.grad.permute(2, 0, 1, 3))
            for grad, leaf in zip((dq, dk, dv), leaves)
        ]
        assert max(errors) < tolerance, f"{dtype}: relative errors out/dq/dk/dv {errors}"

        # The sample without a valid key gives zero rows and no gradient; masked keys get none.
        assert torch.count_nonzero(output[:, -1]) == 0
        for grad in (dq, dk, dv):
            assert torch.count_nonzero(grad[:, -1]) == 0
        masked_keys = mask[:-1, 0, 0].transpose(0, 1)
        assert torch.count_nonzero(dk[:, :-1][masked_keys]) == 0
        assert torch.count_nonzero(dv[:, :-1][masked_keys]) == 0

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_bitwise_equal_to_hand_written_varlen_attention(self, dtype, monkeypatch):
        """Output and gradients equal the hand-written TE varlen conversion bit for bit."""
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        mask = _mask()
        inputs = _inputs(mask.shape[0], dtype)
        grad_output = torch.randn(
            SEQUENCE, mask.shape[0], HEADS * HEAD_DIM, device="cuda", dtype=dtype
        )
        half = dtype != torch.float32
        config = _config(
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, params_dtype=dtype
        )
        native = _run(
            _te_attention(config),
            inputs,
            mask,
            grad_output,
            attn_mask_type=AttnMaskType.key_padding,
        )
        reference = _run(
            _reference_attention(
                _config(
                    bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, params_dtype=dtype
                )
            ),
            inputs,
            mask,
            grad_output,
            attn_mask_type=AttnMaskType.arbitrary,
        )
        for name, actual, expected in zip(("output", "dq", "dk", "dv"), native, reference):
            _assert_bitwise(actual, expected, f"{dtype} {name} (varlen={half})")

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_rejects_dense_masks(self, dtype):
        inputs = _inputs(2, dtype)
        attention = _te_attention(_config(bf16=dtype == torch.bfloat16, params_dtype=dtype))
        dense = torch.zeros(2, 1, SEQUENCE, SEQUENCE, dtype=torch.bool, device="cuda")
        with pytest.raises(ValueError, match=r"\[b, 1, 1, sk\]"):
            attention(*inputs, dense, attn_mask_type=AttnMaskType.key_padding)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_shared_mask_matches_per_sample_masks(self, dtype, monkeypatch):
        """A [1, 1, 1, sk] mask applies to every sample, exactly like its [b, 1, 1, sk] copy."""
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        batch = 4
        mask = _mask()[1:2]  # registers, 7 of 28 caption tokens, video
        per_sample = mask.expand(batch, -1, -1, -1).contiguous()
        inputs = _inputs(batch, dtype)
        grad_output = torch.randn(SEQUENCE, batch, HEADS * HEAD_DIM, device="cuda", dtype=dtype)
        attention = _te_attention(_config(bf16=dtype == torch.bfloat16, params_dtype=dtype))
        kwargs = dict(attn_mask_type=AttnMaskType.key_padding)
        shared = _run(attention, inputs, mask, grad_output, **kwargs)
        expected = _run(attention, inputs, per_sample, grad_output, **kwargs)
        for name, actual, reference in zip(("output", "dq", "dk", "dv"), shared, expected):
            _assert_bitwise(actual, reference, f"{dtype} {name}")
        reference, _ = _sdpa_reference(*inputs, per_sample)
        error = _relative_error(shared[0], reference)
        assert error < (1e-2 if dtype == torch.bfloat16 else 2e-3), f"relative error {error}"

    @pytest.mark.parametrize("backend", ["te_bf16", "te_fp32", "local"])
    def test_rejects_masks_of_other_batch_or_length(self, backend):
        """A mask must cover every key and have the batch size of the inputs (or 1)."""
        dtype = torch.float32 if backend == "te_fp32" else torch.bfloat16
        config = _config(bf16=dtype == torch.bfloat16, params_dtype=dtype)
        if backend == "local":
            attention = DotProductAttention(
                config,
                layer_number=1,
                attn_mask_type=AttnMaskType.key_padding,
                attention_type="self",
            )
        else:
            attention = _te_attention(config)
        inputs = _inputs(4, dtype)
        two_samples = torch.zeros(2, 1, 1, SEQUENCE, dtype=torch.bool, device="cuda")
        with pytest.raises(ValueError, match="batch size 2"):
            attention(*inputs, two_samples, attn_mask_type=AttnMaskType.key_padding)
        short = torch.zeros(4, 1, 1, SEQUENCE - 8, dtype=torch.bool, device="cuda")
        with pytest.raises(ValueError, match=f"covers {SEQUENCE - 8} keys"):
            attention(*inputs, short, attn_mask_type=AttnMaskType.key_padding)

    def test_rejects_sliding_window_attention(self):
        """The varlen call compacts keys, so a window would shift: key_padding rejects windows."""
        mask = _mask()
        inputs = _inputs(mask.shape[0], torch.bfloat16)
        windowed = _config(window_size=(8, 8))
        with pytest.raises(ValueError, match="sliding window"):
            _te_attention(windowed)
        with pytest.raises(ValueError, match="sliding window"):
            DotProductAttention(
                windowed,
                layer_number=1,
                attn_mask_type=AttnMaskType.key_padding,
                attention_type="self",
            )
        # Per-call key_padding masks on windowed layers built for another mask type.
        local = DotProductAttention(
            windowed, layer_number=1, attn_mask_type=AttnMaskType.arbitrary, attention_type="self"
        )
        for attention in (_te_attention(windowed, AttnMaskType.arbitrary), local):
            with pytest.raises(ValueError, match="sliding window"):
                attention(*inputs, mask, attn_mask_type=AttnMaskType.key_padding)
        # Layers without a window (window_attn_skip_freq) keep supporting key_padding.
        unwindowed_layer = TEDotProductAttention(
            config=_config(window_size=(8, 8), window_attn_skip_freq=2),
            layer_number=2,
            attn_mask_type=AttnMaskType.key_padding,
            attention_type="self",
            pg_collection=_pg_collection(),
        )
        output = unwindowed_layer(*inputs, mask, attn_mask_type=AttnMaskType.key_padding)
        expected, _ = _sdpa_reference(*inputs, mask)
        assert _relative_error(output, expected) < 1e-2

    def test_no_mask_runs_unmasked(self):
        inputs = _inputs(2, torch.bfloat16)
        attention = _te_attention(_config())
        output = attention(*inputs, None, attn_mask_type=AttnMaskType.key_padding)
        expected = attention(*inputs, None, attn_mask_type=AttnMaskType.no_mask)
        assert torch.equal(output, expected)

    def test_thd_maps_key_padding_and_arbitrary_to_padding(self, monkeypatch):
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        cu_seqlens = torch.tensor([0, 98, 275, 300], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=177,
            max_seqlen_kv=177,
        )
        inputs = [t[:, 0] for t in _inputs(1, torch.bfloat16, length=300)]
        grad_output = torch.randn(300, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        attention = _te_attention(_config())
        expected = _run(
            attention,
            inputs,
            None,
            grad_output,
            attn_mask_type=AttnMaskType.padding,
            packed_seq_params=packed_seq_params,
        )
        for mask_type in (AttnMaskType.key_padding, AttnMaskType.arbitrary):
            actual = _run(
                attention,
                inputs,
                None,
                grad_output,
                attn_mask_type=mask_type,
                packed_seq_params=packed_seq_params,
            )
            for name, a, e in zip(("output", "dq", "dk", "dv"), actual, expected):
                _assert_bitwise(a, e, f"{mask_type} {name}")
        with pytest.raises(ValueError, match="attention_mask=None"):
            attention(
                *inputs,
                torch.zeros(3, 1, 1, 177, dtype=torch.bool, device="cuda"),
                attn_mask_type=AttnMaskType.key_padding,
                packed_seq_params=packed_seq_params,
            )

    def test_metadata_cached_across_layers_and_checkpoint_recompute(self, monkeypatch):
        """One metadata build (one nonzero sync) serves every layer and the recompute."""
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        builds = []
        build = transformer_utils._build_key_padding_metadata
        monkeypatch.setattr(
            transformer_utils,
            "_build_key_padding_metadata",
            lambda *args: builds.append(args) or build(*args),
        )
        mask = _mask()
        inputs = _inputs(mask.shape[0], torch.bfloat16)
        grad_output = torch.randn(
            SEQUENCE, mask.shape[0], HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        layers = [_te_attention(_config()) for _ in range(3)]

        def block(query, key, value, attention_mask, checkpoint):
            for layer in layers:

                def run(query, key, value, attention_mask, layer=layer):
                    output = layer(
                        query, key, value, attention_mask, attn_mask_type=AttnMaskType.key_padding
                    )
                    return output.view_as(query)

                if checkpoint:
                    query = tensor_parallel.checkpoint(
                        run, False, query, key, value, attention_mask
                    )
                else:
                    query = run(query, key, value, attention_mask)
            return query.flatten(2)

        expected = _run(lambda *args: block(*args, checkpoint=False), inputs, mask, grad_output)
        assert len(builds) == 1
        clear_key_padding_metadata_cache()
        builds.clear()
        actual = _run(lambda *args: block(*args, checkpoint=True), inputs, mask, grad_output)
        assert len(builds) == 1, "recompute and later layers must reuse the cached metadata"
        for name, a, e in zip(("output", "dq", "dk", "dv"), actual, expected):
            _assert_bitwise(a, e, f"checkpointed {name}")

    def test_metadata_cache_invalidation(self, monkeypatch):
        builds = []
        build = transformer_utils._build_key_padding_metadata
        monkeypatch.setattr(
            transformer_utils,
            "_build_key_padding_metadata",
            lambda *args: builds.append(args) or build(*args),
        )
        mask = _mask()
        first = get_key_padding_metadata(mask, SEQUENCE)
        assert get_key_padding_metadata(mask.detach(), SEQUENCE) is first
        assert get_key_padding_metadata(mask.view_as(mask), SEQUENCE) is first
        assert len(builds) == 1
        # A different query length, an in-place update and a new tensor all miss.
        get_key_padding_metadata(mask, SEQUENCE // 2)
        assert len(builds) == 2
        mask[1, 0, 0, -1] = True
        updated = get_key_padding_metadata(mask, SEQUENCE)
        assert len(builds) == 3 and updated is not first
        assert updated.cu_seqlens_kv[2] - updated.cu_seqlens_kv[1] == (
            first.cu_seqlens_kv[2] - first.cu_seqlens_kv[1] - 1
        )
        get_key_padding_metadata(mask.clone(), SEQUENCE)
        assert len(builds) == 4

    @pytest.mark.parametrize("refresh", ["increment_version", "clear_cache"])
    def test_metadata_refreshed_after_untracked_mask_write(self, refresh):
        """A mask buffer refilled in place without a version bump (a collective, ``.data``).

        Such writes are invisible to the cache, like to autograd; increment_version or a cache
        clear after the write must give the metadata of the new mask contents.
        """
        mask = _mask()
        expected = transformer_utils._build_key_padding_metadata(mask, SEQUENCE)
        buffer = torch.zeros_like(mask)  # the previous step's mask: every key valid
        previous = get_key_padding_metadata(buffer, SEQUENCE)
        version = buffer._version
        # Refill the buffer in place: rank 0 through .data, the other ranks by an NCCL broadcast.
        if torch.distributed.get_rank() == 0:
            buffer.data.copy_(mask)
        torch.distributed.broadcast(buffer, src=0)
        assert torch.equal(buffer, mask)
        if refresh == "increment_version":
            torch.autograd.graph.increment_version(buffer)
            assert buffer._version > version
        else:
            clear_key_padding_metadata_cache()
        refreshed = get_key_padding_metadata(buffer, SEQUENCE)
        assert refreshed is not previous
        for name in ("sbhd_kv_index", "bshd_kv_index", "cu_seqlens_q", "cu_seqlens_kv", "empty"):
            assert torch.equal(getattr(refreshed, name), getattr(expected, name)), name

    def test_metadata_not_cached_in_inference_mode(self, monkeypatch):
        """Metadata built under inference_mode is not cached: a later training call can save it."""
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        builds = []
        build = transformer_utils._build_key_padding_metadata
        monkeypatch.setattr(
            transformer_utils,
            "_build_key_padding_metadata",
            lambda *args: builds.append(args) or build(*args),
        )
        mask = _mask()
        inputs = _inputs(mask.shape[0], torch.bfloat16)
        grad_output = torch.randn(
            SEQUENCE, mask.shape[0], HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        attention = _te_attention(_config())
        with torch.inference_mode():
            inference_output = attention(*inputs, mask, attn_mask_type=AttnMaskType.key_padding)
            attention(*inputs, mask, attn_mask_type=AttnMaskType.key_padding)
        assert len(builds) == 2
        # Training forward/backward with the same (normal) mask tensor.
        actual = _run(attention, inputs, mask, grad_output, attn_mask_type=AttnMaskType.key_padding)
        assert len(builds) == 3
        clear_key_padding_metadata_cache()
        expected = _run(
            attention, inputs, mask, grad_output, attn_mask_type=AttnMaskType.key_padding
        )
        for name, a, e in zip(("output", "dq", "dk", "dv"), actual, expected):
            _assert_bitwise(a, e, f"after inference_mode: {name}")
        _assert_bitwise(inference_output.clone(), expected[0], "inference_mode output")

    def test_metadata_layout(self):
        mask = _mask()
        metadata = get_key_padding_metadata(mask, SEQUENCE)
        valid = ~mask[:, 0, 0]
        lengths = valid.sum(-1)
        lengths[-1] = 1  # a sample without valid keys keeps key 0
        assert metadata.cu_seqlens_kv.tolist() == [0] + lengths.cumsum(0).tolist()
        assert metadata.cu_seqlens_q.tolist() == [i * SEQUENCE for i in range(mask.shape[0] + 1)]
        assert metadata.empty.flatten().tolist() == [False] * 4 + [True]
        keys = torch.arange(SEQUENCE * mask.shape[0], device="cuda").view(SEQUENCE, -1)
        valid[-1, 0] = True
        sbhd = transformer_utils.key_padding_pack(keys[..., None, None], metadata.sbhd_kv_index)
        assert torch.equal(sbhd.flatten(), keys.transpose(0, 1)[valid])
        keys = keys.transpose(0, 1).contiguous()
        bshd = transformer_utils.key_padding_pack(keys[..., None, None], metadata.bshd_kv_index)
        assert torch.equal(bshd.flatten(), keys[valid])

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_bshd_matches_sbhd(self, dtype, monkeypatch):
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        mask = _mask()
        inputs = _inputs(mask.shape[0], dtype)
        grad_output = torch.randn(
            SEQUENCE, mask.shape[0], HEADS * HEAD_DIM, device="cuda", dtype=dtype
        )
        config = _config(bf16=dtype == torch.bfloat16, params_dtype=dtype)
        expected = _run(
            _te_attention(config),
            inputs,
            mask,
            grad_output,
            attn_mask_type=AttnMaskType.key_padding,
        )
        attention = _te_attention(config)
        attention.qkv_format = "bshd"
        actual = _run(
            attention,
            [t.transpose(0, 1).contiguous() for t in inputs],
            mask,
            grad_output.transpose(0, 1).contiguous(),
            attn_mask_type=AttnMaskType.key_padding,
        )
        for name, a, e in zip(("output", "dq", "dk", "dv"), actual, expected):
            _assert_bitwise(a.transpose(0, 1), e, f"bshd {dtype} {name}")

    def test_local_dot_product_attention(self):
        """The local path accepts key_padding: broadcast mask, zero rows without valid keys."""
        mask = _mask()
        inputs = _inputs(mask.shape[0], torch.bfloat16)
        attention = DotProductAttention(
            _config(),
            layer_number=1,
            attn_mask_type=AttnMaskType.key_padding,
            attention_type="self",
        )
        output = attention(*inputs, mask.clone(), attn_mask_type=AttnMaskType.key_padding)
        expected, _ = _sdpa_reference(*inputs, mask)
        error = _relative_error(output, expected)
        assert error < 2e-2, f"relative error {error}"
        assert torch.count_nonzero(output[:, -1]) == 0

    def test_fused_scale_mask_softmax_accepts_key_padding(self):
        scores = torch.randn(2, HEADS, 16, 16, device="cuda", dtype=torch.bfloat16)
        mask = torch.rand(2, 1, 1, 16, device="cuda") < 0.3
        mask = mask.expand(-1, -1, 16, -1).contiguous()
        outputs = []
        for mask_type in (AttnMaskType.key_padding, AttnMaskType.arbitrary):
            softmax = FusedScaleMaskSoftmax(
                input_in_fp16=False,
                input_in_bf16=True,
                attn_mask_type=mask_type,
                scaled_masked_softmax_fusion=False,
                mask_func=attention_mask_func,
                softmax_in_fp32=True,
                scale=None,
            )
            outputs.append(softmax(scores.clone(), mask))
        assert torch.equal(outputs[0], outputs[1])
