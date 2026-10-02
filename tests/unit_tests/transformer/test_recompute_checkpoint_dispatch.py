# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""recompute_granularity='full' has to checkpoint, and the offload path has to keep the
per-layer shared inputs (including hash-routing token IDs) resident.

Runs on the CPU with toy layers: torch.utils.checkpoint stands in for
tensor_parallel.checkpoint, which needs CUDA RNG state.
"""

import types
from unittest import mock

import pytest
import torch
import torch.utils.checkpoint

from megatron.core import recompute
from megatron.core.transformer.transformer_layer import TransformerLayer

HIDDEN = 8
SEQ = 4
BATCH = 2


class _ToyLayer(TransformerLayer):
    """TransformerLayer stand-in that records how often, and with which grad mode, it runs."""

    def __init__(self, layer_number: int, is_hash_layer: bool = False):
        torch.nn.Module.__init__(self)  # skip TransformerLayer.__init__ (needs a full config)
        self.layer_number = layer_number
        self.weight = torch.nn.Parameter(torch.full((HIDDEN,), 1.0 + 0.1 * layer_number))
        self.mlp = types.SimpleNamespace(router=types.SimpleNamespace(is_hash_layer=is_hash_layer))
        self.grad_modes = []
        self.input_ids_seen = []

    def __call__(self, *args, **kwargs):  # bypass the CUDA-graph dispatch in __call__
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, hidden_states, context=None, input_ids=None, **kwargs):
        self.grad_modes.append(torch.is_grad_enabled())
        self.input_ids_seen.append(input_ids)
        return torch.tanh(hidden_states * self.weight), context


def _block(num_layers, recompute_method, recompute_num_layers, hash_layers=(), **config):
    layers = [_ToyLayer(i + 1, is_hash_layer=i in hash_layers) for i in range(num_layers)]
    cfg = dict(
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        distribute_saved_activations=False,
        fp8=None,
        fp4=None,
        cpu_offloading=False,
        cpu_offloading_num_layers=0,
        cpu_offloading_prefetch_num_layers=1,
    )
    cfg.update(config)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(**cfg),
        layers=layers,
        num_layers_per_pipeline_rank=num_layers,
        pg_collection=types.SimpleNamespace(tp=None),
    )


def _run(block, input_ids=None):
    """One forward/backward through checkpointed_forward; returns the checkpoint call count."""
    calls = []

    def fake_tp_checkpoint(function, distribute_saved_activations, *args):
        calls.append(len(args))
        return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=True)

    hidden_states = torch.randn(SEQ, BATCH, HIDDEN, requires_grad=True)
    kwargs = {} if input_ids is None else {"input_ids": input_ids}
    with mock.patch.object(recompute.tensor_parallel, "checkpoint", fake_tp_checkpoint):
        out = recompute.checkpointed_forward(
            block,
            hidden_states=hidden_states,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            attention_bias=None,
            packed_seq_params=None,
            use_inner_quantization_context=False,
            **kwargs,
        )
        out.float().pow(2).sum().backward()
    assert hidden_states.grad is not None
    return len(calls)


class TestFullRecomputeCheckpoints:
    """Without FP8/FP4, the recompute path checkpoints through tensor_parallel.checkpoint."""

    def test_block_method_checkpoints_first_layers_without_fp8(self):
        block = _block(4, "block", 2)
        assert _run(block) == 2
        for layer in block.layers[:2]:
            # No-grad forward inside the checkpoint, then the recompute in backward.
            assert layer.grad_modes == [False, True]
        for layer in block.layers[2:]:
            assert layer.grad_modes == [True]

    def test_uniform_method_checkpoints_every_chunk_without_fp8(self):
        block = _block(4, "uniform", 2)
        assert _run(block) == 2
        for layer in block.layers:
            assert layer.grad_modes == [False, True]

    def test_hash_routing_ids_reach_layers_through_checkpoint(self):
        block = _block(2, "block", 2, hash_layers=(0,))
        input_ids = torch.arange(SEQ * BATCH).reshape(BATCH, SEQ)
        assert _run(block, input_ids=input_ids) == 2
        assert all(ids is not None for ids in block.layers[0].input_ids_seen)
        assert block.layers[1].input_ids_seen == [None, None]


@pytest.mark.skipif(
    recompute.te_mark_not_offload is None, reason="needs Transformer Engine's offload API"
)
class TestRecomputeOffloadKeepsSharedInputsResident:
    """The CPU-offload path of full recompute keeps the inputs shared by all layers resident."""

    def test_hash_routing_ids_are_not_offloaded(self):
        block = _block(2, "block", 2, hash_layers=(0,), cpu_offloading=True)
        input_ids = torch.arange(SEQ * BATCH).reshape(BATCH, SEQ)
        saved_ids_marks = []

        def pack(tensor):
            # What TE's offload hook checks before copying a saved tensor to the CPU.
            if tensor is input_ids:
                saved_ids_marks.append(getattr(tensor, "_TE_do_not_offload", False))
            return tensor

        block.offload_context = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        block.group_prefetch_offload_commit_async = lambda tensor: tensor
        block.offload_manual_controller = None

        assert _run(block, input_ids=input_ids) == 2
        # Saved once per checkpointed layer, and marked resident every time.
        assert saved_ids_marks == [True, True]
