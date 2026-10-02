# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for `moe_seq_aux_loss_normalization='per_sequence'`.

The reference functions below are verbatim copies of a downstream implementation of the
per-sequence balance loss (a dense variant on valid-token-compacted inputs and a packed variant),
which the native option has to reproduce up to fp32 rounding.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    get_default_pg_collection,
    per_sequence_load_balancing_loss_func,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

NUM_EXPERTS = 16
TOPK = 4


# ------------------------------------------------------------------------------------------------
# Reference implementation (verbatim copy, only the helper names are prefixed).
# ------------------------------------------------------------------------------------------------
def _ref_sum_tokens(value, group):
    if group is None:
        return value
    return reduce_from_tensor_model_parallel_region(value, group=group)


def _ref_sequence_balance_loss(scores, valid, top_k, group=None):
    valid = valid.to(scores.dtype)
    tokens = _ref_sum_tokens(valid.sum(dim=1), group).clamp_min(1)
    experts = scores.shape[-1]
    unbiased = scores.topk(top_k, dim=-1).indices
    assignments = scores.new_zeros(scores.shape[0], experts)
    assignments.scatter_add_(
        1,
        unbiased.reshape(scores.shape[0], -1),
        valid[..., None].expand(-1, -1, top_k).reshape(scores.shape[0], -1),
    )
    assignments = _ref_sum_tokens(assignments, group)
    frequency = experts * assignments / (top_k * tokens[:, None])
    probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    probability_sum = _ref_sum_tokens((probabilities * valid[..., None]).sum(dim=1), group)
    return (frequency.detach() * (probability_sum / tokens[:, None])).sum(dim=-1).mean()


def _ref_router_scores(logits, score):
    if score == "sqrtsoftplus":
        return torch.nn.functional.softplus(logits).sqrt()
    return logits.sigmoid()


def _ref_packed_balance_loss(scores, sample_ids, lengths, top_k):
    batch, experts = lengths.numel(), scores.shape[-1]
    choices = scores.topk(top_k, dim=-1).indices
    positions = sample_ids[:, None] * experts + choices
    counts = (
        scores.new_zeros(batch * experts)
        .scatter_add(0, positions.flatten(), scores.new_ones(positions.numel()))
        .view(batch, experts)
    )
    probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    sums = scores.new_zeros(batch, experts).index_add(0, sample_ids, probabilities)
    lengths = lengths.clamp_min(1)[:, None]
    return ((experts * counts / (top_k * lengths)).detach() * (sums / lengths)).sum(-1).mean()


# ------------------------------------------------------------------------------------------------
# Helpers.
# ------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _deterministic_algorithms(enabled=True):
    """Make the floating-point index_add accumulation reproducible (or not) within the block."""
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _assert_ulp_close(actual, expected, what, max_ulps=8):
    """fp32 values agree up to a few units in the last place of the largest magnitude."""
    actual, expected = actual.detach().float(), expected.detach().float()
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    if expected.numel() == 0:
        return
    scale = expected.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
    tolerance = max_ulps * torch.finfo(torch.float32).eps * scale
    diff = (actual - expected).abs().max()
    assert diff <= tolerance, f"{what}: max |diff| {diff.item():.3e} > {tolerance.item():.3e}"


def _make_router(score_function="sigmoid", coeff=1.0, normalization="per_sequence", **kwargs):
    tp = parallel_state.get_tensor_model_parallel_world_size()
    cp = parallel_state.get_context_parallel_world_size()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=8,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=coeff,
        moe_seq_aux_loss_normalization=normalization,
        moe_router_score_function=score_function,
        moe_router_dtype="fp32",
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=tp > 1,
        **kwargs,
    )
    router = TopKRouter(config=config, pg_collection=get_default_pg_collection()).cuda()
    router.set_layer_number(1)
    return router


def _aux_value_and_grad(router, logits, **routing_kwargs):
    """Run the router on `logits` and return the logged aux loss and its gradient w.r.t. logits.

    The main-loss gradient is zero, so the gradient is that of the aux loss alone (with a loss
    scale of 1).
    """
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    tracker = get_moe_metrics_tracker()
    tracker.clear()
    logits = logits.detach().clone().requires_grad_(True)
    probs, _ = router.routing(logits, **routing_kwargs)
    (probs.sum() * 0).backward()
    entry = tracker.metrics["seq_load_balancing_loss"]
    assert entry.reduce_group is None, "the per-sequence loss is already reduced over TP x CP"
    value = entry.values[0].clone()
    tracker.clear()
    return value, logits.grad


def _global_batch(seq_length, batch, seed, empty_rank=None):
    """Logits and validity of a global [seq_length, batch] batch, identical on all ranks.

    Padding is non-uniform with holes, sample 1 has no valid token, and `empty_rank` (a TP x CP
    rank) gets a shard without any valid token.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        seq_length, batch, NUM_EXPERTS, device="cuda", generator=generator, dtype=torch.float32
    )
    keep = torch.rand(seq_length, batch, device="cuda", generator=generator)
    valid = keep < torch.linspace(0.3, 0.9, batch, device="cuda")
    valid[:, 1] = False
    if empty_rank is not None:
        shard = seq_length // parallel_state.get_tensor_and_context_parallel_world_size()
        valid[empty_rank * shard : (empty_rank + 1) * shard] = False
    return logits, valid


def _local_shard(tensor):
    """The sequence shard of this TP x CP rank."""
    size = parallel_state.get_tensor_and_context_parallel_world_size()
    rank = parallel_state.get_tensor_and_context_parallel_rank()
    return tensor.chunk(size, dim=0)[rank].contiguous()


class TestPerSequenceSeqAuxLoss:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _init(self, tp_size=1, cp_size=1):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
    @pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 2), (8, 1), (1, 4)])
    def test_dense_padded_layout_matches_reference(self, score_function, tp_size, cp_size):
        """[seq, batch] router input with a padding mask, sequences from the batch layout."""
        self._init(tp_size, cp_size)
        router = _make_router(score_function)
        group = parallel_state.get_tensor_and_context_parallel_group()
        seq_length, batch = 8 * tp_size * cp_size, 5
        logits, valid = _global_batch(seq_length, batch, seed=11)
        logits, valid = _local_shard(logits), _local_shard(valid)

        value, grad = _aux_value_and_grad(router, logits, padding_mask=~valid)

        reference_logits = logits.detach().clone().requires_grad_(True)
        selected = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        flat_logits = reference_logits.reshape(-1, NUM_EXPERTS)
        scores = _ref_router_scores(flat_logits[selected], score_function)
        full_scores = scores.new_zeros(valid.numel(), NUM_EXPERTS).index_copy(0, selected, scores)
        reference = _ref_sequence_balance_loss(
            full_scores.reshape(*valid.shape, -1).transpose(0, 1), valid.T, TOPK, group
        )
        reference.backward()

        _assert_ulp_close(value, reference, "loss")
        _assert_ulp_close(grad, reference_logits.grad, "d loss / d logits")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
    @pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 2), (8, 1)])
    def test_compacted_tokens_with_sequence_ids_match_reference(
        self, score_function, tp_size, cp_size
    ):
        """[num_valid, 1] router input with explicit ids; the last TP x CP rank has no token."""
        self._init(tp_size, cp_size)
        router = _make_router(score_function)
        group = parallel_state.get_tensor_and_context_parallel_group()
        size = parallel_state.get_tensor_and_context_parallel_world_size()
        seq_length, batch = 8 * size, 6
        logits, valid = _global_batch(
            seq_length, batch, seed=17, empty_rank=size - 1 if size > 1 else None
        )
        logits, valid = _local_shard(logits), _local_shard(valid)
        selected = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        compacted = logits.reshape(-1, NUM_EXPERTS)[selected].unsqueeze(1)
        if size > 1 and parallel_state.get_tensor_and_context_parallel_rank() == size - 1:
            assert compacted.shape[0] == 0

        value, grad = _aux_value_and_grad(
            router, compacted, sequence_ids=selected % batch, num_sequences=batch
        )

        reference_logits = compacted.detach().clone().requires_grad_(True)
        scores = _ref_router_scores(reference_logits[:, 0], score_function)
        full_scores = scores.new_zeros(valid.numel(), NUM_EXPERTS).index_copy(0, selected, scores)
        reference = _ref_sequence_balance_loss(
            full_scores.reshape(*valid.shape, -1).transpose(0, 1), valid.T, TOPK, group
        )
        reference.backward()

        _assert_ulp_close(value, reference, "loss")
        _assert_ulp_close(grad, reference_logits.grad, "d loss / d logits")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
    @pytest.mark.parametrize("deterministic", [False, True])
    def test_packed_sequences_match_reference(self, score_function, deterministic):
        """Packed (THD) router input; sequence ids are the sample index of every token."""
        self._init()
        router = _make_router(score_function)
        generator = torch.Generator(device="cuda").manual_seed(5)
        lengths = torch.tensor([37, 1, 120, 64, 9], device="cuda")
        sample_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device="cuda"), lengths)
        logits = torch.randn(int(lengths.sum()), 1, NUM_EXPERTS, device="cuda", generator=generator)

        with _deterministic_algorithms(deterministic):
            value, grad = _aux_value_and_grad(
                router, logits, sequence_ids=sample_ids, num_sequences=lengths.numel()
            )
            reference_logits = logits.detach().clone().requires_grad_(True)
            reference = _ref_packed_balance_loss(
                _ref_router_scores(reference_logits[:, 0], score_function),
                sample_ids,
                lengths.float(),
                TOPK,
            )
            reference.backward()

        if deterministic:
            # Both sides accumulate with the same deterministic index_add.
            assert torch.equal(value, reference.detach()), (value, reference)
            assert torch.equal(grad, reference_logits.grad)
        _assert_ulp_close(value, reference, "loss")
        _assert_ulp_close(grad, reference_logits.grad, "d loss / d logits")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_default_sequences_are_the_batch_entries(self):
        """Without sequence_ids, token i of the flattened [seq, batch] input is in sequence
        i % batch. int32 ids work like int64 ids."""
        self._init(2, 2)
        router = _make_router()
        seq_length, batch = 32, 3
        logits, valid = _global_batch(seq_length, batch, seed=23)
        logits, valid = _local_shard(logits), _local_shard(valid)
        ids = torch.arange(valid.numel(), device="cuda") % batch
        with _deterministic_algorithms():
            implicit = _aux_value_and_grad(router, logits, padding_mask=~valid)
            for dtype in (torch.int64, torch.int32):
                explicit = _aux_value_and_grad(
                    router,
                    logits,
                    padding_mask=~valid,
                    sequence_ids=ids.to(dtype),
                    num_sequences=batch,
                )
                assert torch.equal(implicit[0], explicit[0]), dtype
                assert torch.equal(implicit[1], explicit[1]), dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 2)])
    def test_matches_batch_mean_without_padding(self, tp_size, cp_size):
        """Without padding all sequences have the same length and both normalizations agree."""
        self._init(tp_size, cp_size)
        group = parallel_state.get_tensor_and_context_parallel_group()
        seq_length, batch = 16 * tp_size * cp_size, 4
        logits, _ = _global_batch(seq_length, batch, seed=29)
        logits = _local_shard(logits)

        per_sequence = _aux_value_and_grad(_make_router("sigmoid"), logits)
        batch_mean_router = _make_router("sigmoid", normalization="batch_mean")
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        tracker = get_moe_metrics_tracker()
        tracker.clear()
        batch_mean_logits = logits.detach().clone().requires_grad_(True)
        probs, _ = batch_mean_router.routing(batch_mean_logits)
        (probs.sum() * 0).backward()
        # The batch-mean loss is a per-rank partial that the tracker sums over TP x CP.
        batch_mean_value = tracker.metrics["seq_load_balancing_loss"].values[0].clone()
        batch_mean_value = reduce_from_tensor_model_parallel_region(batch_mean_value, group)
        tracker.clear()

        _assert_ulp_close(per_sequence[0], batch_mean_value, "loss", max_ulps=64)
        _assert_ulp_close(per_sequence[1], batch_mean_logits.grad, "gradient", max_ulps=64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_coefficient_and_per_token_loss_scaling(self, monkeypatch):
        """The coefficient scales the gradient, and per-token-loss scaling uses the exact local
        number of valid tokens."""
        self._init(2, 1)
        seq_length, batch = 32, 3
        logits, valid = _global_batch(seq_length, batch, seed=31)
        logits, valid = _local_shard(logits), _local_shard(valid)
        _, unit_grad = _aux_value_and_grad(_make_router(coeff=1.0), logits, padding_mask=~valid)

        router = _make_router(coeff=0.25, calculate_per_token_loss=True)
        counts = []
        attach = router.attach_and_log_load_balancing_loss

        def spy(*args, **kwargs):
            counts.append(kwargs["valid_token_count"])
            return attach(*args, **kwargs)

        monkeypatch.setattr(router, "attach_and_log_load_balancing_loss", spy)
        value, grad = _aux_value_and_grad(router, logits, padding_mask=~valid)
        local_valid = int(valid.sum())
        assert len(counts) == 1 and int(counts[0]) == local_valid
        expected = unit_grad * 0.25 * local_valid * router.tp_cp_group.size()
        _assert_ulp_close(grad, expected, "scaled gradient", max_ulps=64)
        assert value > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_grad_and_eval_skip_the_loss(self):
        self._init()
        router = _make_router()
        logits = torch.randn(8, 2, NUM_EXPERTS, device="cuda")
        tracker = get_moe_metrics_tracker()
        tracker.clear()
        with torch.no_grad():
            router.routing(logits)
        router.eval()
        router.routing(logits)
        entry = tracker.metrics.get("seq_load_balancing_loss")
        assert entry is None or not entry.values.any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_argument_validation(self):
        self._init()
        router = _make_router()
        logits = torch.randn(8, 2, NUM_EXPERTS, device="cuda", requires_grad=True)
        ids = torch.zeros(16, dtype=torch.long, device="cuda")
        with pytest.raises(ValueError, match="num_sequences is required"):
            router.routing(logits, sequence_ids=ids)
        with pytest.raises(ValueError, match="must match the batch size"):
            router.routing(logits, num_sequences=3)
        with pytest.raises(ValueError, match="sequence_ids"):
            router.routing(logits, sequence_ids=ids[:5], num_sequences=1)
        # Batch-first ids, laid out like the padding mask of MoELayer.forward, have the right
        # number of elements but not the flattened [seq_length, bsz] order.
        with pytest.raises(ValueError, match=r"flattened token order.*shape \(2, 8\)"):
            router.routing(logits, sequence_ids=ids.view(2, 8), num_sequences=1)
        with pytest.raises(ValueError, match="dtype torch.float32"):
            router.routing(logits, sequence_ids=ids.float(), num_sequences=1)
        with pytest.raises(ValueError, match="moe_seq_aux_loss_normalization"):
            _make_router(normalization="per_token")


class TestPerSequenceSeqAuxLossInMoELayer:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size,ep_size,shared_expert", [(1, 4, True), (2, 2, False)])
    def test_compacted_tokens_with_an_empty_rank(self, tp_size, ep_size, shared_expert):
        """MoELayer on valid-token-compacted [num_valid, 1, hidden] inputs with sequence ids,
        where rank 3 has no token: the forward pass, the aux-loss reductions, the empty expert
        all-to-all and the backward pass complete, and every parameter gets a gradient."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=64 if shared_expert else None,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=0.01,
            moe_seq_aux_loss_normalization="per_sequence",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            use_cpu_initialization=True,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=8, moe_grouped_gemm=True
            ).mlp
        )
        layer = MoELayer(config, submodules, layer_number=1).cuda()
        rank = torch.distributed.get_rank()
        num_tokens = 0 if rank == 3 else 5 + 3 * rank
        generator = torch.Generator(device="cuda").manual_seed(rank)
        hidden = torch.randn(num_tokens, 1, 64, device="cuda", generator=generator).bfloat16()
        router_input = torch.randn(num_tokens, 1, 64, device="cuda", generator=generator)
        hidden.requires_grad_(True)
        router_input = router_input.bfloat16().requires_grad_(True)
        sequence_ids = torch.arange(num_tokens, device="cuda") % 4

        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        get_moe_metrics_tracker().clear()
        output, _ = layer(
            hidden, router_input=router_input, sequence_ids=sequence_ids, num_sequences=4
        )
        assert output.shape == hidden.shape
        output.float().sum().backward()
        assert hidden.grad.shape == hidden.shape
        assert router_input.grad.shape == router_input.shape
        for name, param in layer.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
        value = get_moe_metrics_tracker().metrics["seq_load_balancing_loss"].values[0]
        # Without TP the empty rank holds no token of its sequences, so its loss is 0.
        assert value > 0 or (tp_size == 1 and num_tokens == 0)
        # Every rank of a TP x CP group reports the loss of the same complete sequences.
        values = [torch.zeros_like(value) for _ in range(tp_size)]
        torch.distributed.all_gather(
            values, value, group=parallel_state.get_tensor_and_context_parallel_group()
        )
        assert all(torch.equal(v, value) for v in values)
        get_moe_metrics_tracker().clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_loss_function_excludes_padding_and_counts_exactly():
    """Padding tokens neither count nor receive gradient; the counts are exact integers."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    probs = torch.rand(10, NUM_EXPERTS, device="cuda", generator=generator).requires_grad_(True)
    top_indices = torch.rand(10, NUM_EXPERTS, device="cuda", generator=generator).topk(TOPK).indices
    sequence_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0, 1, 2], device="cuda")
    padding_mask = torch.tensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 1], device="cuda").bool()
    keep = ~padding_mask
    reference_probs = probs.detach()[keep].clone().requires_grad_(True)
    with _deterministic_algorithms():
        loss, local_num_tokens = per_sequence_load_balancing_loss_func(
            probs, top_indices, sequence_ids, 3, TOPK, NUM_EXPERTS, 1.0, padding_mask=padding_mask
        )
        loss.backward()
        reference, _ = per_sequence_load_balancing_loss_func(
            reference_probs, top_indices[keep], sequence_ids[keep], 3, TOPK, NUM_EXPERTS, 1.0
        )
        reference.backward()
    assert local_num_tokens.dtype == torch.int64 and int(local_num_tokens) == 6
    assert not probs.grad[padding_mask].any()
    assert probs.grad[keep].abs().sum() > 0
    assert torch.equal(loss, reference)
    assert torch.equal(probs.grad[keep], reference_probs.grad)


def test_loss_function_checks_the_range_of_sequence_ids():
    """The ids of valid tokens must lie in [0, num_sequences); padding ids are not used.

    The check is an asynchronous device-side assertion, which leaves a CUDA context unusable
    when it fires, so the out-of-range cases run on the CPU, where it raises a RuntimeError.
    """
    generator = torch.Generator().manual_seed(5)
    probs = torch.rand(6, NUM_EXPERTS, generator=generator)
    top_indices = torch.rand(6, NUM_EXPERTS, generator=generator).topk(TOPK).indices
    padding_mask = torch.tensor([False, False, False, False, False, True])

    def loss(ids, mask=None):
        ids = torch.tensor(ids)
        value, _ = per_sequence_load_balancing_loss_func(
            probs, top_indices, ids, 3, TOPK, NUM_EXPERTS, 1.0, padding_mask=mask
        )
        return value

    expected = loss([0, 1, 2, 2, 1, 0], padding_mask)
    for padding_id in (-1, 3, 99):
        assert torch.equal(loss([0, 1, 2, 2, 1, padding_id], padding_mask), expected)
    # With a padding mask, a valid token with the id num_sequences (3) would otherwise be
    # dropped like padding without an error.
    for ids in ([0, 1, 3, 2, 1, 0], [0, -1, 2, 2, 1, 0]):
        for mask in (None, padding_mask):
            with pytest.raises(RuntimeError, match="sequence_ids"):
                loss(ids, mask)
