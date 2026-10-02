# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Extra layer keyword arguments and early exit in ``TransformerBlock.forward``.

``layer_kwargs`` passes extra keyword arguments, e.g. conditioning tensors shared by all layers,
to every layer call; ``exit_after_layer`` stops the block after one layer. Both have to give the
same results in the eager layer loop and under full activation recompute, where the tensors of
``layer_kwargs`` are explicit checkpoint inputs.
"""

from contextlib import nullcontext
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

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


def _run(block, autocast=False, **forward_kwargs):
    """One forward and backward pass of ``block`` with conditioning passed as layer_kwargs.

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
            layer_kwargs = {
                "conditioning": {"time": time_embedding, "masks": [token_mask], "mlp_scale": 0.5}
            }
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
        if layers_per_chunk == 1:
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

    @pytest.mark.skipif(not HAVE_TE, reason="CPU offloading needs Transformer Engine")
    @pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "autocast-bf16"])
    def test_cpu_offloading_with_layer_kwargs(self, autocast):
        offload_kwargs = dict(
            cpu_offloading=True, cpu_offloading_num_layers=3, cpu_offloading_prefetch_num_layers=1
        )
        reference = _run(_build_block(recompute_method="block"), autocast=autocast)
        block = _build_block(recompute_method="block", **offload_kwargs)
        controller = block.offload_manual_controller
        assert controller is not None
        with mock.patch.object(
            controller, "start_offload_layer", wraps=controller.start_offload_layer
        ) as start_offload:
            result = _run(block, autocast=autocast)
        assert [c.args[0] for c in start_offload.call_args_list] == [0, 1, 2]
        # Offloading only moves bytes, so the results are bitwise identical.
        _assert_bitwise_equal(result["output"], reference["output"])
        _assert_same_grads(result, reference)
        # The time embedding is shared by every layer: it stays on the GPU.
        assert getattr(result["time_embedding"], "_TE_do_not_offload", False)

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
