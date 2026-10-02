# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for router and MoE robustness fixes.

- Router gating with input and weight dtypes that differ.
- Router gating, the router, its z-loss and the shared experts on 0 tokens.
- Top-k probabilities that do not depend on the grad mode.
- The expert bias staying fp32 under module dtype casts.
- The global token counts that the expert-bias update keeps for metrics.

`_ReferenceRouter` is a verbatim copy of a downstream router that these fixes let `TopKRouter`
replace bit for bit.
"""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias,
    reset_model_temporary_tensors,
)
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import (
    get_default_pg_collection,
    router_gating_linear,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

HIDDEN = 256
NUM_EXPERTS = 64
TOPK = 8


# ------------------------------------------------------------------------------------------------
# Reference router (verbatim copy; configuration read from TransformerConfig).
# ------------------------------------------------------------------------------------------------
def _ref_router_scores(logits, score):
    if score == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    return logits.sigmoid()


class _ReferenceRouter(Router):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.score = config.moe_router_score_function
        self.route_scale = config.moe_router_topk_scaling_factor
        self.register_buffer(
            "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32, device="cuda")
        )

    def _apply(self, function, recurse=True):
        super()._apply(function, recurse)
        self.expert_bias = self.expert_bias.float()
        return self

    def routing(self, logits):
        scores = _ref_router_scores(logits.reshape(-1, self.num_experts), self.score)
        indices = (scores + self.expert_bias).topk(self.config.moe_router_topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.route_scale
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)
        probabilities = weights.new_zeros(scores.shape).scatter(1, indices, weights)
        return probabilities, routing_map

    def forward(self, hidden_states, padding_mask=None, input_ids=None):
        length = hidden_states.shape[0]
        if length == 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, 1))
        hidden_states = hidden_states.to(self.weight.dtype)
        return self.routing(self.gating(hidden_states)[:length])


# ------------------------------------------------------------------------------------------------
# Helpers.
# ------------------------------------------------------------------------------------------------
def _config(params_dtype=torch.bfloat16, **overrides):
    tp = parallel_state.get_tensor_model_parallel_world_size()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=2.5,
        moe_router_dtype="fp32",
        use_cpu_initialization=True,
        bf16=params_dtype == torch.bfloat16,
        params_dtype=params_dtype,
        add_bias_linear=False,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
    )
    return dataclasses.replace(config, **overrides)


def _router(config, cls=TopKRouter):
    router = cls(config=config, pg_collection=get_default_pg_collection()).cuda()
    router.set_layer_number(1)
    return router


def _run_router(router, hidden, grad_probs=None):
    """Return probs, routing map and, with grad_probs, the input and weight gradients."""
    hidden = hidden.detach().clone().requires_grad_(grad_probs is not None)
    router.weight.grad = None
    probs, routing_map = router(hidden)
    if grad_probs is None:
        return probs.detach(), routing_map, None, None
    probs.backward(grad_probs)
    return probs.detach(), routing_map, hidden.grad, router.weight.grad.clone()


class TestRouterNativeFixes:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
    @pytest.mark.parametrize(
        "input_dtype,params_dtype",
        [
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.bfloat16, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("grad", [True, False])
    @pytest.mark.parametrize("num_tokens", [4096, 0])
    def test_topk_router_matches_reference_router(
        self, score_function, input_dtype, params_dtype, grad, num_tokens
    ):
        """Mixed-dtype gating casts the input to the weight dtype first, the top-k is sorted
        without grad as well, and 0 tokens work without padding; all bitwise as the reference."""
        config = _config(params_dtype, moe_router_score_function=score_function)
        router = _router(config)
        reference = _router(config, _ReferenceRouter)
        with torch.no_grad():
            reference.weight.copy_(router.weight)
            router.expert_bias.copy_(torch.randn_like(router.expert_bias) * 0.05)
            reference.expert_bias.copy_(router.expert_bias)
        hidden = torch.randn(num_tokens, 1, HIDDEN, device="cuda").to(input_dtype)
        grad_probs = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") if grad else None

        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            actual = _run_router(router, hidden, grad_probs)
            expected = _run_router(reference, hidden, grad_probs)
        assert actual[0].shape == (num_tokens, NUM_EXPERTS)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        if grad:
            assert actual[2].dtype == input_dtype
            assert torch.equal(actual[2], expected[2])
            assert torch.equal(actual[3], expected[3])
            if num_tokens == 0:
                assert not actual[3].any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gating_matches_pre_cast_input(self):
        """Mixed dtypes give exactly the gating of the input cast to the weight dtype; fp64
        routing keeps the input untouched."""
        for input_dtype, params_dtype in [
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ]:
            router = _router(_config(params_dtype))
            hidden = torch.randn(64, 2, HIDDEN, device="cuda").to(input_dtype).requires_grad_()
            cast = hidden.detach().to(params_dtype).requires_grad_()
            logits = router.gating(hidden)
            expected = router_gating_linear(cast, router.weight, None, torch.float32)
            assert torch.equal(logits, expected)
            grad = torch.randn_like(logits)
            logits.backward(grad)
            expected.backward(grad)
            assert hidden.grad.dtype == input_dtype
            assert torch.equal(hidden.grad, cast.grad.to(input_dtype))

        router = _router(_config(torch.bfloat16, moe_router_dtype="fp64"))
        hidden = torch.randn(64, 2, HIDDEN, device="cuda")
        logits = router.gating(hidden)
        weight = router.weight.double()
        assert logits.dtype == torch.float64
        torch.testing.assert_close(logits, hidden.double() @ weight.t())
        assert not torch.allclose(logits, hidden.bfloat16().double() @ weight.t())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("shape", [(0, 1, HIDDEN), (0, HIDDEN)])
    @pytest.mark.parametrize("router_dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_gating_linear_on_zero_tokens(self, shape, router_dtype, with_bias):
        """Forward and backward on 0 tokens; the weight gradient is zero and its hooks fire."""
        weight = torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        weight.requires_grad_(True)
        bias = None
        if with_bias:
            bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
            bias.requires_grad_(True)
        fired = []
        weight.register_post_accumulate_grad_hook(lambda param: fired.append(param.grad.clone()))
        hidden = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        logits = router_gating_linear(hidden, weight, bias, router_dtype)
        assert logits.shape == (*shape[:-1], NUM_EXPERTS) and logits.dtype == router_dtype
        logits.sum().backward()
        assert hidden.grad.shape == hidden.shape
        assert len(fired) == 1 and torch.equal(fired[0], torch.zeros_like(weight))
        if with_bias:
            assert torch.equal(bias.grad, torch.zeros_like(bias))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
    def test_topk_probs_do_not_depend_on_grad_mode(self, score_function):
        logits = torch.randn(32768, NUM_EXPERTS, device="cuda") * 2
        expert_bias = torch.randn(NUM_EXPERTS, device="cuda") * 0.05
        kwargs = dict(
            topk=TOPK,
            score_function=score_function,
            expert_bias=expert_bias,
            scaling_factor=2.5,
            dense_output=True,
        )
        with torch.enable_grad():
            probs, indices = topk_routing_with_score_function(logits, **kwargs)
        with torch.no_grad():
            no_grad_probs, no_grad_indices = topk_routing_with_score_function(logits, **kwargs)
        assert torch.equal(no_grad_indices, indices)
        assert torch.equal(no_grad_probs, probs)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_topk_is_unsorted_only_in_inference_mode(self, monkeypatch):
        sorted_flags = []
        topk = torch.topk

        def spy(*args, **kwargs):
            sorted_flags.append(kwargs.get("sorted", True))
            return topk(*args, **kwargs)

        monkeypatch.setattr(torch, "topk", spy)
        logits = torch.randn(16, NUM_EXPERTS, device="cuda")
        with torch.no_grad():
            topk_routing_with_score_function(logits, TOPK, score_function="sigmoid")
            with InferenceMode.active():
                topk_routing_with_score_function(logits, TOPK, score_function="sigmoid")
        topk_routing_with_score_function(logits, TOPK, score_function="sigmoid")
        assert sorted_flags == [True, False, True]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("load_balancing_type", ["none", "seq_aux_loss"])
    def test_topk_router_on_zero_tokens(self, load_balancing_type):
        overrides = dict(moe_router_load_balancing_type=load_balancing_type)
        if load_balancing_type == "seq_aux_loss":
            overrides.update(moe_aux_loss_coeff=0.01, moe_seq_aux_loss_normalization="per_sequence")
        config = _config(**overrides)
        router = _router(config)
        hidden = torch.empty(0, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_(True)
        probs, routing_map = router(hidden)
        assert probs.shape == routing_map.shape == (0, NUM_EXPERTS)
        probs.sum().backward()
        assert torch.equal(router.weight.grad, torch.zeros_like(router.weight))
        assert hidden.grad.shape == hidden.shape
        assert not router.local_tokens_per_expert.any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("with_padding_mask", [False, True])
    def test_z_loss_on_zero_tokens(self, with_padding_mask):
        """The z-loss of 0 tokens is 0 rather than NaN, so that a router without tokens logs a
        finite value; the z-loss of tokens is unchanged."""
        padding_mask = None
        if with_padding_mask:
            padding_mask = torch.zeros(0, dtype=torch.bool, device="cuda")
        logits = torch.randn(0, NUM_EXPERTS, device="cuda", requires_grad=True)
        z_loss = z_loss_func(logits, 1e-3, padding_mask=padding_mask)
        assert torch.equal(z_loss, torch.zeros_like(z_loss))
        z_loss.backward()
        assert logits.grad.shape == logits.shape
        logits = torch.randn(64, NUM_EXPERTS, device="cuda")
        expected = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * 1e-3
        assert torch.equal(z_loss_func(logits, 1e-3), expected)

        router = _router(_config(moe_z_loss_coeff=1e-3))
        tracker = get_moe_metrics_tracker()
        tracker.clear()
        hidden = torch.empty(0, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_(True)
        router_padding_mask = None if padding_mask is None else padding_mask.view(0, 1)
        probs, _ = router(hidden, padding_mask=router_padding_mask)
        probs.sum().backward()
        logged = tracker.metrics["z_loss"].values
        assert torch.equal(logged, torch.zeros_like(logged))
        assert torch.equal(router.weight.grad, torch.zeros_like(router.weight))
        tracker.clear()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_tokens", [0, 7])
    def test_shared_expert_on_zero_tokens(self, num_tokens):
        """The fused SwiGLU of the shared expert accepts 3D inputs with 0 tokens."""
        config = _config(
            moe_shared_expert_intermediate_size=128,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=NUM_EXPERTS, moe_grouped_gemm=True
            ).mlp
        )
        shared_expert = submodules.shared_experts(
            config=config, pg_collection=get_default_pg_collection(), gate=False
        ).cuda()
        hidden = torch.randn(num_tokens, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_(True)
        output = shared_expert(hidden)
        assert output.shape == hidden.shape
        output.sum().backward()
        assert hidden.grad.shape == hidden.shape
        for name, param in shared_expert.named_parameters():
            assert param.grad is not None, name
            if num_tokens == 0:
                assert not param.grad.any(), name

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expert_bias_stays_fp32_under_module_casts(self):
        router = _router(_config())
        assert router.expert_bias.dtype == torch.float32
        router.to(torch.bfloat16)
        assert router.expert_bias.dtype == torch.float32
        assert router.weight.dtype == torch.bfloat16
        assert router.local_tokens_per_expert.dtype == torch.int64
        assert router.last_global_tokens_per_expert.dtype == torch.int64
        # Values written after the cast keep full precision (e.g. copying a teacher's weights).
        values = torch.rand(NUM_EXPERTS, device="cuda", dtype=torch.float64).float() * 1e-3 + 1
        router.expert_bias.copy_(values)
        assert torch.equal(router.expert_bias, values)
        router.half()
        assert router.expert_bias.dtype == torch.float32
        wrapped = Float16Module(_config(), _router(_config()))
        assert wrapped.module.expert_bias.dtype == torch.float32

        quantile_router = _router(
            _config(
                moe_router_load_balancing_type="quantile_balancing",
                moe_router_enable_expert_bias=False,
            )
        )
        quantile_router.to(torch.bfloat16)
        assert quantile_router.qb_beta.dtype == torch.float32
        assert quantile_router.qb_beta_accum.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_last_global_tokens_per_expert_buffer(self):
        router = _router(_config())
        assert "last_global_tokens_per_expert" not in router.state_dict()
        assert torch.equal(
            router.last_global_tokens_per_expert,
            torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda"),
        )
        router = _router(_config(moe_router_enable_expert_bias=False))
        assert router.last_global_tokens_per_expert is None


class TestLastGlobalTokensPerExpert:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_update_keeps_the_reduced_counts(self, tp_size):
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = _config(moe_router_bias_update_rate=1e-3)
        routers = torch.nn.ModuleList([_router(config), _router(config)])
        frozen = _router(config)
        frozen.eval()
        generator = torch.Generator(device="cuda").manual_seed(torch.distributed.get_rank())
        for router in [*routers, frozen]:
            for _ in range(2):
                hidden = torch.randn(32, 2, HIDDEN, device="cuda", generator=generator)
                router(hidden.bfloat16())
        group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
        expected = [router.local_tokens_per_expert.clone() for router in routers]
        for counts in expected:
            assert counts.sum() == 2 * 32 * 2 * TOPK
            torch.distributed.all_reduce(counts, group=group)
        biases = [router.expert_bias.clone() for router in routers]

        _update_router_expert_bias([routers], config, tp_dp_cp_group=group)
        for router, counts, bias in zip(routers, expected, biases):
            assert torch.equal(router.last_global_tokens_per_expert, counts)
            # The local counters are reset only afterwards.
            assert not torch.equal(router.local_tokens_per_expert, counts)
            direction = torch.sign(counts.sum() - counts * NUM_EXPERTS).float()
            assert torch.equal(router.expert_bias, bias + direction * 1e-3)
        assert not frozen.last_global_tokens_per_expert.any()

        reset_model_temporary_tensors(config, [routers])
        for router, counts in zip(routers, expected):
            assert not router.local_tokens_per_expert.any()
            assert torch.equal(router.last_global_tokens_per_expert, counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("num_tokens", [0, 5])
@pytest.mark.parametrize("weighted", [False, True])
def test_fused_swiglu_3d_inputs(num_tokens, weighted):
    """3D inputs (including 0 tokens) match the 2D path."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    inputs = torch.randn(num_tokens, 2, 64, device="cuda", generator=generator).bfloat16()
    weights = torch.rand(num_tokens * 2, 1, device="cuda", generator=generator)

    def run(x):
        x = x.detach().clone().requires_grad_(True)
        if weighted:
            output = weighted_bias_swiglu_impl(x, None, weights)
        else:
            output = bias_swiglu_impl(x, None)
        output.backward(torch.ones_like(output))
        return output.detach(), x.grad

    output, grad = run(inputs)
    flat_output, flat_grad = run(inputs.reshape(-1, 64))
    assert output.shape == (num_tokens, 2, 32)
    assert torch.equal(output.reshape(-1, 32), flat_output)
    assert torch.equal(grad.reshape(-1, 64), flat_grad)
