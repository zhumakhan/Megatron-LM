# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the `router_input`, `sequence_ids` and `num_sequences` arguments of MoELayer."""

import contextlib
import dataclasses

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe import moe_layer as moe_layer_module
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64


class _SideChannelMoELayer(MoELayer):
    """Reference: swaps the router input in through an attribute, as downstream code did before
    `router_input` existed (verbatim `route` override)."""

    router_tokens = None

    def route(self, hidden_states, padding_mask=None, *args, **kwargs):
        tokens, self.router_tokens = self.router_tokens, None
        return super().route(
            hidden_states if tokens is None else tokens, padding_mask, *args, **kwargs
        )


@contextlib.contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _config(**overrides):
    tp = parallel_state.get_tensor_model_parallel_world_size()
    ep = parallel_state.get_expert_model_parallel_world_size()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=64,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_seq_aux_loss_normalization="per_sequence",
        moe_z_loss_coeff=1e-3,
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
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        sequence_parallel=tp > 1,
        use_cpu_initialization=True,
    )
    return dataclasses.replace(config, **overrides)


def _layer(config, cls=MoELayer, source=None, strict=True):
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
        ).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    layer = cls(config, submodules, layer_number=1).cuda()
    if source is None:
        with torch.no_grad():
            layer.router.expert_bias.copy_(torch.randn_like(layer.router.expert_bias) * 0.05)
    else:
        layer.load_state_dict(source.state_dict(), strict=strict)
    layer.train()
    return layer


def _inputs(seed, seq_length=16, batch=3):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (seq_length, batch, HIDDEN)
    hidden = torch.randn(*shape, device="cuda", generator=generator).bfloat16()
    router_input = torch.randn(*shape, device="cuda", generator=generator).bfloat16()
    grad_output = torch.randn(*shape, device="cuda", generator=generator).bfloat16()
    return hidden, router_input, grad_output


def _run(layer, hidden, router_input, grad_output, side_channel=False, **kwargs):
    """Forward and backward; returns output, input grads, parameter grads and router state."""
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    get_moe_metrics_tracker().clear()
    hidden = hidden.detach().clone().requires_grad_(True)
    router_input = router_input.detach().clone().requires_grad_(True)
    if side_channel:
        layer.router_tokens = router_input
        output, _ = layer(hidden, **kwargs)
    else:
        output, _ = layer(hidden, router_input=router_input, **kwargs)
    output.backward(grad_output)
    grads = {name: p.grad.clone() for name, p in layer.named_parameters() if p.grad is not None}
    aux = get_moe_metrics_tracker().metrics["seq_load_balancing_loss"].values.clone()
    state = {"local_tokens_per_expert": layer.router.local_tokens_per_expert.clone(), "aux": aux}
    layer.zero_grad(set_to_none=True)
    layer.router.local_tokens_per_expert.zero_()
    return output.detach(), hidden.grad, router_input.grad, grads, state


def _assert_runs_equal(actual, expected):
    output, hidden_grad, router_grad, grads, state = actual
    ref_output, ref_hidden_grad, ref_router_grad, ref_grads, ref_state = expected
    assert torch.equal(output, ref_output)
    assert torch.equal(hidden_grad, ref_hidden_grad)
    assert torch.equal(router_grad, ref_router_grad)
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        assert torch.equal(grads[name], ref_grads[name]), name
    for name in state:
        assert torch.equal(state[name], ref_state[name]), name


class TestMoELayerRouterInput:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _init(self, tp_size, ep_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    def test_router_input_matches_side_channel_reference(self, tp_size, ep_size):
        """The router (gating, z-loss, aux loss, expert-bias counters) reads router_input; the
        shared experts, dispatch and experts read hidden_states."""
        self._init(tp_size, ep_size)
        config = _config()
        layer = _layer(config)
        reference = _layer(config, _SideChannelMoELayer, source=layer)
        seed = parallel_state.get_tensor_model_parallel_rank()
        hidden, router_input, grad_output = _inputs(seed=seed)
        with _deterministic_algorithms():
            actual = _run(layer, hidden, router_input, grad_output)
            expected = _run(reference, hidden, router_input, grad_output, side_channel=True)
            default = _run(layer, hidden, hidden, grad_output)
        _assert_runs_equal(actual, expected)
        assert actual[4]["aux"].abs().sum() > 0
        assert actual[2].abs().sum() > 0, "router_input received no gradient"
        # Routing on router_input differs from routing on hidden_states.
        assert not torch.equal(actual[0], default[0])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "fp8",
        [
            None,
            pytest.param(
                "hybrid",
                marks=pytest.mark.skipif(
                    not is_te_min_version("1.7.0.dev0"), reason="te_checkpoint needs TE >= 1.7"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("with_sequence_ids", [False, True])
    def test_router_input_under_moe_recompute(
        self, tp_size, ep_size, fp8, with_sequence_ids, monkeypatch
    ):
        """Under moe_layer_recompute, router_input is an explicit checkpoint input and the
        recomputed layer matches the layer without recompute bit for bit."""
        self._init(tp_size, ep_size)
        # The fp8 config only selects te_checkpoint; without fp8_autocast the GEMMs run in bf16.
        config = _config(fp8=fp8, moe_grouped_gemm=fp8 is None)
        layer = _layer(config)
        recomputed = _layer(
            dataclasses.replace(
                config, recompute_granularity="selective", recompute_modules=["moe"]
            ),
            source=layer,
        )
        assert recomputed.moe_layer_recompute

        checkpoint_args = []
        name = "te_checkpoint" if fp8 else None
        if fp8:
            original = moe_layer_module.te_checkpoint

            def spy(function, distribute, rng_tracker, group, *args, **kwargs):
                checkpoint_args.append(args)
                return original(function, distribute, rng_tracker, group, *args, **kwargs)

            monkeypatch.setattr(moe_layer_module, name, spy)
        else:
            original = tensor_parallel.checkpoint

            def spy(function, distribute, *args):
                checkpoint_args.append(args)
                return original(function, distribute, *args)

            monkeypatch.setattr(tensor_parallel, "checkpoint", spy)

        hidden, router_input, grad_output = _inputs(seed=7)
        kwargs = {}
        if with_sequence_ids:
            # Two sequences per batch entry, numbered globally.
            positions = torch.arange(hidden.shape[0] * hidden.shape[1], device="cuda")
            sequence_ids = positions % hidden.shape[1] * 2 + (positions >= positions.numel() // 2)
            kwargs = dict(sequence_ids=sequence_ids, num_sequences=2 * hidden.shape[1])
        with _deterministic_algorithms():
            expected = _run(layer, hidden, router_input, grad_output, **kwargs)
            actual = _run(recomputed, hidden, router_input, grad_output, **kwargs)
        assert len(checkpoint_args) == 1
        # hidden_states, intermediate_tensors, padding_mask, router_input
        assert len(checkpoint_args[0]) == 4 and checkpoint_args[0][3] is not None
        assert checkpoint_args[0][3].shape == router_input.shape
        _assert_runs_equal(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sequence_ids_reach_the_router(self, monkeypatch):
        """sequence_ids and num_sequences are forwarded to the router and from there to its
        `routing` only when given."""
        self._init(1, 1)
        layer = _layer(_config())
        calls = {"forward": [], "routing": []}
        forward, routing = TopKRouter.forward, TopKRouter.routing

        def forward_spy(router, *args, **kwargs):
            calls["forward"].append(kwargs)
            return forward(router, *args, **kwargs)

        def routing_spy(router, *args, **kwargs):
            calls["routing"].append(kwargs)
            return routing(router, *args, **kwargs)

        monkeypatch.setattr(TopKRouter, "forward", forward_spy)
        monkeypatch.setattr(TopKRouter, "routing", routing_spy)
        hidden, router_input, _ = _inputs(seed=3, seq_length=12, batch=1)
        ids = torch.arange(12, device="cuda") // 4
        layer(hidden, router_input=router_input, sequence_ids=ids, num_sequences=3)
        layer(hidden)
        for kwargs in (calls["forward"][0], calls["routing"][0]):
            assert kwargs["sequence_ids"] is ids and kwargs["num_sequences"] == 3
        for kwargs in (calls["forward"][1], calls["routing"][1]):
            assert "sequence_ids" not in kwargs and "num_sequences" not in kwargs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("override", ["forward", "routing", "routing_attribute"])
    def test_routers_without_sequence_arguments_still_work(self, override):
        """Routers whose `forward` or `routing` has the original signature, in a subclass or as
        an instance attribute (as tests and tools patch it), keep working when no ids are
        given, also with the per-sequence seq_aux_loss."""

        class LegacyForwardRouter(TopKRouter):
            def forward(self, input, padding_mask=None, input_ids=None):
                return super().forward(input, padding_mask, input_ids)

        class LegacyRoutingRouter(TopKRouter):
            def routing(self, logits, padding_mask=None, input_ids=None):
                return super().routing(logits, padding_mask=padding_mask, input_ids=input_ids)

        self._init(1, 1)
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=8, moe_grouped_gemm=True
            ).mlp
        )
        router_cls = {"forward": LegacyForwardRouter, "routing": LegacyRoutingRouter}
        submodules = dataclasses.replace(submodules, router=router_cls.get(override, TopKRouter))
        layer = MoELayer(_config(), submodules, layer_number=1).cuda()
        if override == "routing_attribute":
            routing = layer.router.routing
            layer.router.routing = lambda logits, padding_mask=None, input_ids=None: routing(
                logits, padding_mask=padding_mask, input_ids=input_ids
            )
        hidden, router_input, _ = _inputs(seed=5)
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        get_moe_metrics_tracker().clear()
        output, _ = layer(hidden, router_input=router_input)
        assert output.shape == hidden.shape
        assert get_moe_metrics_tracker().metrics["seq_load_balancing_loss"].values.sum() > 0
        probs, routing_map = layer.router(router_input)
        assert probs.shape == routing_map.shape == (hidden.shape[0] * hidden.shape[1], 8)
        get_moe_metrics_tracker().clear()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_input_shape_is_validated(self):
        self._init(1, 1)
        layer = _layer(_config())
        hidden, router_input, _ = _inputs(seed=9)
        with pytest.raises(ValueError, match="leading shape"):
            layer(hidden, router_input=router_input[:-1])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compacted_tokens_with_the_shared_expert_outside_the_layer(self):
        """With TP > 1 and sequence parallelism, the shared expert of the layer all-gathers
        equal-size shards, so compacted inputs whose token count differs between the TP ranks
        (one rank has none here) need the shared expert outside of the layer, as documented in
        MoELayer.forward: `shared_experts_compute` returns None and `shared_experts` runs on the
        uncompacted shard. The routed part equals that of a layer without a shared expert."""

        class ExternalSharedExpertMoELayer(MoELayer):
            def shared_experts_compute(self, hidden_states):
                return None

        self._init(2, 2)
        config = _config()
        assert config.sequence_parallel and config.moe_shared_expert_intermediate_size
        layer = _layer(config, ExternalSharedExpertMoELayer)
        reference = _layer(
            dataclasses.replace(config, moe_shared_expert_intermediate_size=None),
            source=layer,
            strict=False,
        )
        assert reference.shared_experts is None
        extra_keys = set(layer.state_dict()) - set(reference.state_dict())
        assert extra_keys and all(key.startswith("shared_experts.") for key in extra_keys)

        rank = torch.distributed.get_rank()
        seq_length, batch = 8, 2
        hidden, router_input, grad_output = _inputs(seed=rank, seq_length=seq_length, batch=batch)
        generator = torch.Generator(device="cuda").manual_seed(rank)
        num_valid = (11, 0, 5, 16, 9, 3, 7, 12)[rank]
        order = torch.randperm(seq_length * batch, device="cuda", generator=generator)
        valid = (order < num_valid).view(seq_length, batch)
        selected = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        shared_experts = layer.shared_experts

        def run(moe_layer):
            MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
            get_moe_metrics_tracker().clear()
            x = hidden.detach().clone().requires_grad_(True)
            r = router_input.detach().clone().requires_grad_(True)
            routed, _ = moe_layer(
                x.reshape(-1, HIDDEN)[selected].unsqueeze(1),
                router_input=r.reshape(-1, HIDDEN)[selected].unsqueeze(1),
                sequence_ids=selected % batch,
                num_sequences=batch,
            )
            assert routed.shape == (num_valid, 1, HIDDEN)
            output = torch.zeros_like(x).reshape(-1, HIDDEN)
            output = output.index_copy(0, selected, routed[:, 0]).view_as(x)
            output = output + shared_experts(x).masked_fill(~valid[..., None], 0)
            output.backward(grad_output)
            grads = {
                name: param.grad.clone()
                for name, param in moe_layer.named_parameters()
                if param.grad is not None and not name.startswith("shared_experts.")
            }
            shared_grads = {name: p.grad.clone() for name, p in shared_experts.named_parameters()}
            aux = get_moe_metrics_tracker().metrics["seq_load_balancing_loss"].values.clone()
            moe_layer.zero_grad(set_to_none=True)
            shared_experts.zero_grad(set_to_none=True)
            return output.detach(), x.grad, r.grad, grads, shared_grads, aux

        with _deterministic_algorithms():
            expected = run(reference)
            actual = run(layer)
        for name, value, reference_value in zip(
            ("output", "hidden grad", "router_input grad"), actual[:3], expected[:3]
        ):
            assert torch.equal(value, reference_value), name
        assert "router.weight" in actual[3]
        for grads, reference_grads in zip(actual[3:5], expected[3:5]):
            assert grads.keys() == reference_grads.keys()
            for name in grads:
                assert torch.isfinite(grads[name]).all(), name
                assert torch.equal(grads[name], reference_grads[name]), name
        # Every TP pair has valid tokens, so every shard of the shared expert gets a gradient.
        assert all(grad.any() for grad in actual[4].values())
        assert torch.equal(actual[5], expected[5]) and actual[5].sum() > 0
        get_moe_metrics_tracker().clear()
