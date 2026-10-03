# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Extra layer keyword arguments and early exit in ``TransformerBlock.forward``.

``layer_kwargs`` passes extra keyword arguments, e.g. conditioning tensors shared by all layers,
to every layer call; ``exit_after_layer`` stops the block after one layer. Both have to give the
same results in the eager layer loop and under full activation recompute, where the tensors of
``layer_kwargs`` are explicit checkpoint inputs and values without tensors reach the layers as the
caller's objects.
"""

import dataclasses
from contextlib import nullcontext
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch.utils._pytree import register_dataclass

from megatron.core import recompute
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer, TransformerLayer
from megatron.core.utils import make_viewless_tensor
from tests.unit_tests.test_utilities import Utils

NUM_LAYERS = 4
SEQ_LEN = 32
MICRO_BATCH = 2
HIDDEN = 64
CONDITIONING = 16


class _ConditionedLayer(TransformerLayer):
    """TransformerLayer whose residual branches are modulated by per-sample conditioning.

    The conditioning arrives as the extra keyword argument ``conditioning``, a nested mapping
    with a time embedding shared by all layers ([b, CONDITIONING]), a list holding a boolean
    token mask ([s, b], masked tokens skip the MLP branch) and a Python float.
    """

    def __init__(self, config, submodules, layer_number=1, **kwargs):
        super().__init__(config, submodules, layer_number=layer_number, **kwargs)
        self.modulation = torch.nn.Linear(CONDITIONING, 6 * config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, *, conditioning, **kwargs):
        # The time embedding is read once per layer, so that its gradient is one term per layer.
        modulation = self.modulation(F.silu(conditioning["time"]))
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        normed = self.input_layernorm(hidden_states) * (1 + scale1) + shift1
        attention, bias = self.self_attention(normed, attention_mask=attention_mask)
        if bias is not None:
            attention = attention + bias
        hidden_states = hidden_states + gate1 * attention
        normed = self.pre_mlp_layernorm(hidden_states) * (1 + scale2) + shift2
        mlp, bias = self.mlp(normed)
        if bias is not None:
            mlp = mlp + bias
        (token_mask,) = conditioning["masks"]
        mlp = mlp * (~token_mask)[..., None] * conditioning["mlp_scale"]
        hidden_states = hidden_states + gate2 * mlp
        return make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True), None


class _PlainConditionedLayer(MegatronModule, BaseTransformerLayer):
    """A layer that is not a TransformerLayer, conditioned the same way."""

    def __init__(self, config, layer_number=1, **kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.modulation = torch.nn.Linear(CONDITIONING, config.hidden_size)

    def forward(self, hidden_states, *, conditioning, **kwargs):
        gate = self.modulation(F.silu(conditioning["time"]))
        (token_mask,) = conditioning["masks"]
        update = torch.tanh(self.linear(hidden_states)) * (~token_mask)[..., None]
        return hidden_states + gate * update * conditioning["mlp_scale"], None


@dataclasses.dataclass
class _DataclassConditioning:
    """Conditioning tensors in a dataclass that ``torch.utils._pytree`` does not flatten."""

    time: torch.Tensor
    token_mask: torch.Tensor


@dataclasses.dataclass
class _RegisteredDataclassConditioning(_DataclassConditioning):
    """The same conditioning in a dataclass registered with ``torch.utils._pytree``."""


register_dataclass(_RegisteredDataclassConditioning)


class _DataclassConditionedLayer(MegatronModule, BaseTransformerLayer):
    """A layer conditioned like _PlainConditionedLayer, through a dataclass."""

    def __init__(self, config, layer_number=1, **kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.modulation = torch.nn.Linear(CONDITIONING, config.hidden_size)

    def forward(self, hidden_states, *, conditioning, **kwargs):
        gate = self.modulation(F.silu(conditioning.time))
        update = torch.tanh(self.linear(hidden_states)) * (~conditioning.token_mask)[..., None]
        return hidden_states + gate * update, None


class _SideChannelLayer(MegatronModule, BaseTransformerLayer):
    """A layer that reports a statistic of its output through containers in layer_kwargs.

    ``statistics`` is a list at the top level of layer_kwargs; ``conditioning`` is a dict that
    holds a tensor (``scale``) next to the dict ``statistics_by_layer``.
    """

    def __init__(self, config, layer_number=1, **kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, *, statistics, conditioning, **kwargs):
        output = hidden_states + conditioning["scale"] * torch.tanh(self.linear(hidden_states))
        statistic = output.detach().float().pow(2).mean()
        statistics.append(statistic)
        conditioning["statistics_by_layer"].setdefault(self.layer_number, []).append(statistic)
        return output, None


class _BlockArgumentsLayer(MegatronModule, BaseTransformerLayer):
    """A layer that is not a TransformerLayer and uses context, attention_bias and padding_mask.

    ``received`` records which of them every call got.
    """

    def __init__(self, config, layer_number=1, **kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.received = []

    def forward(
        self, hidden_states, *, context=None, attention_bias=None, padding_mask=None, **kwargs
    ):
        arguments = dict(context=context, attention_bias=attention_bias, padding_mask=padding_mask)
        self.received.append(tuple(name for name, value in arguments.items() if value is not None))
        update = self.linear(hidden_states if context is None else hidden_states + context)
        if attention_bias is not None:
            update = update + attention_bias
        update = torch.tanh(update)
        if padding_mask is not None:
            # [b, s], True marks the padding tokens, which keep their hidden states.
            update = update * (~padding_mask).transpose(0, 1)[..., None]
        return hidden_states + update, context


def _layer_spec(layer_class):
    if layer_class is _ConditionedLayer:
        return ModuleSpec(
            module=_ConditionedLayer,
            submodules=get_gpt_layer_with_transformer_engine_spec().submodules,
        )
    return ModuleSpec(module=layer_class)


def _build_block(layer_class=_ConditionedLayer, recompute_method=None, **config_kwargs):
    # Same seeds for every build, so that blocks of one test share their weights. Parameters
    # are initialized on the CPU, so the CPU generator is seeded too.
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    if recompute_method is not None:
        config_kwargs.setdefault("recompute_granularity", "full")
        config_kwargs.setdefault("recompute_method", recompute_method)
        config_kwargs.setdefault(
            "recompute_num_layers", NUM_LAYERS if recompute_method == "block" else 2
        )
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        **config_kwargs,
    )
    block = TransformerBlock(config, _layer_spec(layer_class)).cuda()
    block.train()
    return block


def _count_forward_calls(block):
    """Count the forward calls of every layer, recomputes included."""
    calls = [0] * len(block.layers)

    def counter(index):
        def hook(module, args, kwargs):
            calls[index] += 1

        return hook

    handles = [
        layer.register_forward_pre_hook(counter(index), with_kwargs=True)
        for index, layer in enumerate(block.layers)
    ]
    return calls, handles


def _conditioning_dict(time_embedding, token_mask):
    """layer_kwargs for _ConditionedLayer and _PlainConditionedLayer."""
    return {"conditioning": {"time": time_embedding, "masks": [token_mask], "mlp_scale": 0.5}}


def _run(block, autocast=False, make_layer_kwargs=_conditioning_dict, **forward_kwargs):
    """One forward and backward pass of ``block`` with conditioning passed as layer_kwargs.

    ``make_layer_kwargs`` builds layer_kwargs from the time embedding and the token mask.
    Returns the output, the gradients of the input, of the shared time embedding, of the time
    embedder weight and of every block parameter, and the forward calls of every layer.
    """
    generator = torch.Generator(device="cuda").manual_seed(7)
    hidden_states = torch.randn(
        (SEQ_LEN, MICRO_BATCH, HIDDEN), device="cuda", generator=generator
    ).requires_grad_(True)
    timesteps = torch.rand((MICRO_BATCH, 1), device="cuda", generator=generator)
    embedder_weight = torch.randn(
        (CONDITIONING, 1), device="cuda", generator=generator
    ).requires_grad_(True)
    token_mask = torch.rand((SEQ_LEN, MICRO_BATCH), device="cuda", generator=generator) < 0.25

    calls, handles = _count_forward_calls(block)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16) if autocast else nullcontext():
            # The time embedding is not a leaf: its gradient has to flow on to the embedder.
            time_embedding = F.silu(F.linear(timesteps, embedder_weight))
            time_embedding.retain_grad()
            layer_kwargs = make_layer_kwargs(time_embedding, token_mask)
            output = block(
                hidden_states=hidden_states,
                attention_mask=None,
                layer_kwargs=layer_kwargs,
                **forward_kwargs,
            )
        loss_input = output[1][-1] if isinstance(output, tuple) else output
        # The backward pass runs outside autocast, as in training loops.
        loss_input.float().pow(2).mean().backward()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    grads = {
        name: param.grad.detach().clone()
        for name, param in block.named_parameters()
        if param.grad is not None
    }
    block.zero_grad(set_to_none=True)
    return dict(
        output=output,
        input_grad=hidden_states.grad,
        time_embedding=time_embedding,
        time_embedding_grad=time_embedding.grad,
        embedder_grad=embedder_weight.grad,
        param_grads=grads,
        calls=calls,
    )


def _assert_bitwise_equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_equal_up_to_summation_order(actual, expected, autocast):
    """Compare gradients of a tensor that several layers of one checkpoint chunk share.

    A checkpoint chunk of several layers adds up its layers' gradient contributions before the
    outer graph adds up the chunks, so only the summation order differs from the eager loop.
    """
    tolerance = (3e-2 if autocast else 1e-5) * expected.abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


def _assert_same_grads(result, reference, layers_per_chunk=1, autocast=False):
    _assert_bitwise_equal(result["input_grad"], reference["input_grad"])
    assert result["param_grads"].keys() == reference["param_grads"].keys()
    for name, grad in reference["param_grads"].items():
        torch.testing.assert_close(
            result["param_grads"][name], grad, rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )
    # The time embedding is shared by all layers.
    for key in ("time_embedding_grad", "embedder_grad"):
        if reference[key] is None:
            assert result[key] is None, key
        elif layers_per_chunk == 1:
            _assert_bitwise_equal(result[key], reference[key])
        else:
            _assert_equal_up_to_summation_order(result[key], reference[key], autocast)


LAYER_CLASSES = [pytest.param(_PlainConditionedLayer, id="plain-layer")]
if HAVE_TE:
    LAYER_CLASSES.insert(0, pytest.param(_ConditionedLayer, id="transformer-layer"))


class TestTransformerBlockLayerKwargs:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layer_class", LAYER_CLASSES)
    @pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "autocast-bf16"])
    @pytest.mark.parametrize("recompute_method", ["block", "uniform"])
    def test_recompute_matches_eager(self, layer_class, autocast, recompute_method):
        reference = _run(_build_block(layer_class), autocast=autocast)
        assert reference["calls"] == [1] * NUM_LAYERS
        # The conditioning reached every layer and its gradient reached the shared time
        # embedding and the embedder weight in front of it.
        assert reference["time_embedding_grad"].abs().sum() > 0
        assert reference["embedder_grad"].abs().sum() > 0

        block = _build_block(layer_class, recompute_method)
        result = _run(block, autocast=autocast)
        # Every layer is checkpointed, so it runs again in the backward pass.
        assert result["calls"] == [2] * NUM_LAYERS
        # Recompute replays the same kernels on the same inputs, under the same autocast.
        _assert_bitwise_equal(result["output"], reference["output"])
        layers_per_chunk = 1 if recompute_method == "block" else block.config.recompute_num_layers
        _assert_same_grads(result, reference, layers_per_chunk, autocast)

    @pytest.mark.parametrize("layer_class", LAYER_CLASSES)
    @pytest.mark.parametrize("recompute_method", [None, "block", "uniform"])
    @pytest.mark.parametrize("exit_after_layer", [1, 2, NUM_LAYERS - 1])
    def test_exit_after_layer(self, layer_class, recompute_method, exit_after_layer):
        # Reference: a full eager pass, with the loss taken on the extracted output of the
        # exit layer. Layers behind it run but get no gradient.
        reference = _run(_build_block(layer_class), extract_layer_indices={exit_after_layer})
        full_output, (intermediate,) = reference["output"]
        if exit_after_layer == NUM_LAYERS - 1:
            # The full pass applies the final layernorm; the early exit must not.
            assert not torch.equal(full_output, intermediate)

        block = _build_block(layer_class, recompute_method)
        result = _run(block, exit_after_layer=exit_after_layer)
        _assert_bitwise_equal(result["output"], intermediate)
        layers_per_chunk = 1 if recompute_method != "uniform" else block.config.recompute_num_layers
        _assert_same_grads(result, reference, layers_per_chunk)
        # Layers behind the exit layer do not run; under recompute the others run twice.
        runs = 1 if recompute_method is None else 2
        assert result["calls"] == [runs] * (exit_after_layer + 1) + [0] * (
            NUM_LAYERS - exit_after_layer - 1
        )

        # extract_layer_indices is handled as usual for the layers that run.
        result = _run(block, exit_after_layer=exit_after_layer, extract_layer_indices={0, 1})
        output, intermediates = result["output"]
        _assert_bitwise_equal(output, intermediate)
        if recompute_method == "uniform":
            # Uniform recompute extracts at chunk ends only: layer 1 ends the chunk [0, 2).
            assert len(intermediates) == 1
        else:
            assert len(intermediates) == 2
        if exit_after_layer == 1:
            _assert_bitwise_equal(intermediates[-1], intermediate)

    @pytest.mark.parametrize("recompute_method", [None, "block"])
    def test_invalid_arguments(self, recompute_method):
        block = _build_block(_PlainConditionedLayer, recompute_method)
        hidden_states = torch.zeros((SEQ_LEN, MICRO_BATCH, HIDDEN), device="cuda")
        for exit_after_layer in (-1, NUM_LAYERS):
            with pytest.raises(ValueError, match="is not a layer of this block"):
                block(hidden_states, None, exit_after_layer=exit_after_layer)
        with pytest.raises(ValueError, match="lie after exit_after_layer"):
            block(hidden_states, None, exit_after_layer=1, extract_layer_indices={2})
        with pytest.raises(ValueError, match="must not contain"):
            block(hidden_states, None, layer_kwargs={"attention_mask": None})

    @pytest.mark.parametrize(
        "recompute_method, recompute_num_layers, calls",
        [
            (None, None, [1] * NUM_LAYERS),
            ("block", NUM_LAYERS, [2] * NUM_LAYERS),
            ("block", 2, [2, 2] + [1] * (NUM_LAYERS - 2)),
            ("uniform", 2, [2] * NUM_LAYERS),
        ],
        ids=["eager", "block", "partial-block", "uniform"],
    )
    def test_side_channels(self, recompute_method, recompute_num_layers, calls):
        # Values without tensors, e.g. empty containers that collect per-layer statistics, reach
        # the layers as the caller's objects, also inside a dict that holds a tensor, which full
        # recompute rebuilds around the checkpoint inputs.
        def run(block):
            generator = torch.Generator(device="cuda").manual_seed(7)
            hidden_states = torch.randn(
                (SEQ_LEN, MICRO_BATCH, HIDDEN), device="cuda", generator=generator
            ).requires_grad_(True)
            scale = torch.rand(
                (MICRO_BATCH, HIDDEN), device="cuda", generator=generator
            ).requires_grad_(True)
            statistics, statistics_by_layer = [], {}
            layer_kwargs = {
                "statistics": statistics,
                "conditioning": {"scale": scale, "statistics_by_layer": statistics_by_layer},
            }
            output = block(
                hidden_states=hidden_states, attention_mask=None, layer_kwargs=layer_kwargs
            )
            forward_counts = [len(statistics_by_layer.get(n, [])) for n in range(1, NUM_LAYERS + 1)]
            assert len(statistics) == NUM_LAYERS
            assert forward_counts == [1] * NUM_LAYERS
            output.float().pow(2).mean().backward()
            return output, scale.grad, statistics, statistics_by_layer

        reference_output, _, _, reference_statistics = run(_build_block(_SideChannelLayer))
        config_kwargs = {}
        if recompute_num_layers is not None:
            config_kwargs["recompute_num_layers"] = recompute_num_layers
        block = _build_block(_SideChannelLayer, recompute_method, **config_kwargs)
        output, scale_grad, statistics, statistics_by_layer = run(block)

        _assert_bitwise_equal(output, reference_output)
        # The tensor inside the rebuilt dict gets gradients.
        assert scale_grad is not None and scale_grad.abs().sum() > 0
        # A checkpointed layer adds its statistic once more when it is recomputed.
        counts = [len(statistics_by_layer[n]) for n in range(1, NUM_LAYERS + 1)]
        assert counts == calls
        assert len(statistics) == sum(calls)
        for layer_number, layer_statistics in statistics_by_layer.items():
            (expected,) = reference_statistics[layer_number]
            for statistic in layer_statistics:
                _assert_bitwise_equal(statistic, expected)

    @pytest.mark.parametrize("recompute_method", ["block", "uniform"])
    def test_dataclass_values(self, recompute_method):
        def unregistered(time_embedding, token_mask):
            return {"conditioning": _DataclassConditioning(time_embedding, token_mask)}

        def unregistered_without_grad(time_embedding, token_mask):
            return {"conditioning": _DataclassConditioning(time_embedding.detach(), token_mask)}

        def registered(time_embedding, token_mask):
            return {"conditioning": _RegisteredDataclassConditioning(time_embedding, token_mask)}

        block = _build_block(_DataclassConditionedLayer, recompute_method)
        layers_per_chunk = 1 if recompute_method == "block" else block.config.recompute_num_layers

        # pytree does not flatten an unregistered dataclass, so the recompute would close over
        # a tensor in it that requires grad. The eager loop takes it, full recompute rejects it.
        reference = _run(_build_block(_DataclassConditionedLayer), make_layer_kwargs=unregistered)
        assert reference["embedder_grad"].abs().sum() > 0
        with pytest.raises(ValueError, match="register_dataclass"):
            _run(block, make_layer_kwargs=unregistered)

        # Closed-over tensors that need no gradients, e.g. masks, are fine.
        reference = _run(
            _build_block(_DataclassConditionedLayer), make_layer_kwargs=unregistered_without_grad
        )
        result = _run(block, make_layer_kwargs=unregistered_without_grad)
        assert result["embedder_grad"] is None
        _assert_bitwise_equal(result["output"], reference["output"])
        _assert_same_grads(result, reference, layers_per_chunk)

        # A registered dataclass is flattened: its tensors are checkpoint inputs.
        reference = _run(_build_block(_DataclassConditionedLayer), make_layer_kwargs=registered)
        result = _run(block, make_layer_kwargs=registered)
        assert result["embedder_grad"].abs().sum() > 0
        _assert_bitwise_equal(result["output"], reference["output"])
        _assert_same_grads(result, reference, layers_per_chunk)

    @pytest.mark.parametrize("recompute_method", ["block", "uniform"])
    def test_custom_layer_gets_block_arguments(self, recompute_method):
        # A BaseTransformerLayer that is not a TransformerLayer is called like a TransformerLayer
        # under full recompute, as in the eager loop: it gets context, attention_bias and
        # padding_mask.
        def run(block):
            generator = torch.Generator(device="cuda").manual_seed(7)
            shape = (SEQ_LEN, MICRO_BATCH, HIDDEN)
            hidden_states = torch.randn(shape, device="cuda", generator=generator)
            context = torch.randn(shape, device="cuda", generator=generator)
            attention_bias = torch.randn((HIDDEN,), device="cuda", generator=generator)
            padding_mask = torch.rand((MICRO_BATCH, SEQ_LEN), device="cuda", generator=generator)
            padding_mask = padding_mask < 0.25
            hidden_states.requires_grad_(True)
            context.requires_grad_(True)
            output = block(
                hidden_states=hidden_states,
                attention_mask=None,
                context=context,
                attention_bias=attention_bias,
                padding_mask=padding_mask,
            )
            output.float().pow(2).mean().backward()
            grads = {name: param.grad for name, param in block.named_parameters()}
            received = [layer.received for layer in block.layers]
            return output, hidden_states.grad, context.grad, grads, received

        everything = ("context", "attention_bias", "padding_mask")
        ref_output, ref_input_grad, ref_context_grad, ref_grads, ref_received = run(
            _build_block(_BlockArgumentsLayer)
        )
        assert ref_received == [[everything]] * NUM_LAYERS
        output, input_grad, context_grad, grads, received = run(
            _build_block(_BlockArgumentsLayer, recompute_method)
        )
        # Both the forward call and the recompute get the arguments.
        assert received == [[everything, everything]] * NUM_LAYERS
        _assert_bitwise_equal(output, ref_output)
        _assert_bitwise_equal(input_grad, ref_input_grad)
        for name, grad in ref_grads.items():
            _assert_bitwise_equal(grads[name], grad)
        # Every layer reads the context, so only the summation order of its gradient differs.
        _assert_equal_up_to_summation_order(context_grad, ref_context_grad, autocast=False)

    @pytest.mark.skipif(
        not HAVE_TE or recompute.te_mark_not_offload is None,
        reason="CPU offloading on the recompute path needs Transformer Engine's offload API",
    )
    @pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "autocast-bf16"])
    def test_cpu_offloading_with_layer_kwargs(self, autocast):
        offload_kwargs = dict(
            cpu_offloading=True, cpu_offloading_num_layers=3, cpu_offloading_prefetch_num_layers=1
        )
        reference = _run(_build_block(recompute_method="block"), autocast=autocast)
        block = _build_block(recompute_method="block", **offload_kwargs)
        controller = block.offload_manual_controller
        assert controller is not None
        with (
            mock.patch.object(
                controller, "start_offload_layer", wraps=controller.start_offload_layer
            ) as start_offload,
            mock.patch.object(
                recompute, "te_mark_not_offload", wraps=recompute.te_mark_not_offload
            ) as mark_not_offload,
        ):
            result = _run(block, autocast=autocast)
        assert [c.args[0] for c in start_offload.call_args_list] == [0, 1, 2]
        # Offloading only moves bytes, so the results are bitwise identical.
        _assert_bitwise_equal(result["output"], reference["output"])
        _assert_same_grads(result, reference)
        # The time embedding is shared by every layer: it is marked to stay on the GPU.
        marked = [tensor for call in mark_not_offload.call_args_list for tensor in call.args]
        assert any(tensor is result["time_embedding"] for tensor in marked)

    @pytest.mark.skipif(not HAVE_TE, reason="CPU offloading needs Transformer Engine")
    @pytest.mark.parametrize("recompute_method", [None, "block"])
    def test_cpu_offloading_skipped_on_early_exit(self, recompute_method):
        offload_kwargs = dict(
            cpu_offloading=True, cpu_offloading_num_layers=2, cpu_offloading_prefetch_num_layers=1
        )
        reference = _run(_build_block(recompute_method=recompute_method))
        block = _build_block(recompute_method=recompute_method, **offload_kwargs)
        # A pass that exits early leaves Transformer Engine's offload context alone ...
        with (
            mock.patch.object(block, "offload_context") as offload_context,
            mock.patch.object(block, "group_prefetch_offload_commit_async") as offload_commit,
        ):
            _run(block, exit_after_layer=1)
        offload_context.__enter__.assert_not_called()
        offload_commit.assert_not_called()
        # ... so the next full pass still offloads layer by layer and matches the baseline.
        result = _run(block)
        _assert_bitwise_equal(result["output"], reference["output"])
        _assert_same_grads(result, reference)
