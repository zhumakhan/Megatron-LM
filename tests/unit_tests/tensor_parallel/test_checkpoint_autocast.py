# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Activation checkpoints recompute under the torch autocast state of the forward pass.

The backward pass usually runs outside the forward pass's ``torch.autocast`` region. Unless
the checkpoint replays the recorded autocast state, the recompute runs ``linear`` in the input
dtype instead of the autocast dtype (and ``rms_norm`` in the input dtype instead of fp32), so
the gradients differ from a run without checkpointing.
"""

from functools import partial

import pytest
import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    model_parallel_cuda_manual_seed,
)
from tests.unit_tests.test_utilities import Utils

TOKENS = 48
HIDDEN = 64


def _rms_norm_linear(x, weight, dtypes=None):
    """Under CUDA autocast, rms_norm runs in fp32 and linear in the autocast dtype."""
    normed = F.rms_norm(x, (x.shape[-1],), eps=1e-5)
    output = F.linear(normed, weight)
    if dtypes is not None:
        dtypes.append((normed.dtype, output.dtype))
    return output


def _inputs(device="cuda"):
    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.randn((TOKENS, HIDDEN), device=device, generator=generator)
    weight = torch.randn((HIDDEN, HIDDEN), device=device, generator=generator) / HIDDEN**0.5
    return x.requires_grad_(True), weight.requires_grad_(True)


def _loss(output):
    return output.float().pow(2).sum()


def _reference(device="cuda", dtype=None):
    """Gradients without checkpointing, under autocast to ``dtype`` (None: no autocast)."""
    x, weight = _inputs(device)
    with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
        output = _rms_norm_linear(x, weight)
    _loss(output).backward()
    return output.detach(), x.grad, weight.grad


def _with_forward_autocast(function):
    """Wrap ``function`` so that every call re-enters the CUDA autocast state of this moment.

    This is what callers had to do by hand while the checkpoints did not replay autocast.
    """
    autocast_state = dict(
        device_type="cuda",
        enabled=torch.is_autocast_enabled("cuda"),
        dtype=torch.get_autocast_dtype("cuda"),
        cache_enabled=torch.is_autocast_cache_enabled(),
    )

    def wrapped(*args):
        with torch.autocast(**autocast_state):
            return function(*args)

    return wrapped


def _assert_bitwise_equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestCheckpointAutocastReplay:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_checkpoint(self, dtype):
        ref_output, ref_x_grad, ref_weight_grad = _reference(dtype=dtype)

        x, weight = _inputs()
        dtypes = []
        with torch.autocast("cuda", dtype=dtype):
            output = tensor_parallel.checkpoint(
                partial(_rms_norm_linear, dtypes=dtypes), False, x, weight
            )
        _loss(output).backward()

        _assert_bitwise_equal(output, ref_output)
        _assert_bitwise_equal(x.grad, ref_x_grad)
        _assert_bitwise_equal(weight.grad, ref_weight_grad)
        # The forward pass and the recompute both ran under autocast.
        assert dtypes == [(torch.float32, dtype)] * 2

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_checkpoint_without_output(self, dtype):
        def forward(x, weight, checkpoint=None, dtypes=None):
            with torch.autocast("cuda", dtype=dtype):
                if checkpoint is None:
                    output = _rms_norm_linear(x, weight)
                else:
                    output = checkpoint.checkpoint(
                        partial(_rms_norm_linear, dtypes=dtypes), x, weight
                    )
                # The product saves the checkpoint output, which is discarded and restored.
                result = output * output
            if checkpoint is not None:
                checkpoint.discard_output_and_register_recompute(result)
            return result

        x_ref, weight_ref = _inputs()
        ref_result = forward(x_ref, weight_ref)
        _loss(ref_result).backward()

        x, weight = _inputs()
        dtypes = []
        result = forward(x, weight, CheckpointWithoutOutput(), dtypes)
        _loss(result).backward()

        _assert_bitwise_equal(result, ref_result)
        _assert_bitwise_equal(x.grad, x_ref.grad)
        _assert_bitwise_equal(weight.grad, weight_ref.grad)
        assert dtypes == [(torch.float32, dtype)] * 2

    def test_checkpoint_matches_explicit_autocast_wrapper(self):
        x_wrapped, weight_wrapped = _inputs()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wrapped = _with_forward_autocast(_rms_norm_linear)
            output_wrapped = tensor_parallel.checkpoint(wrapped, False, x_wrapped, weight_wrapped)
        _loss(output_wrapped).backward()

        x, weight = _inputs()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = tensor_parallel.checkpoint(_rms_norm_linear, False, x, weight)
        _loss(output).backward()

        _assert_bitwise_equal(output, output_wrapped)
        _assert_bitwise_equal(x.grad, x_wrapped.grad)
        _assert_bitwise_equal(weight.grad, weight_wrapped.grad)

    def test_checkpoint_replays_cpu_autocast(self):
        ref_output, ref_x_grad, ref_weight_grad = _reference(device="cpu", dtype=torch.bfloat16)

        x, weight = _inputs(device="cpu")
        dtypes = []
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = tensor_parallel.checkpoint(
                partial(_rms_norm_linear, dtypes=dtypes), False, x, weight
            )
        _loss(output).backward()

        _assert_bitwise_equal(output, ref_output)
        _assert_bitwise_equal(x.grad, ref_x_grad)
        _assert_bitwise_equal(weight.grad, ref_weight_grad)
        assert [output_dtype for _, output_dtype in dtypes] == [torch.bfloat16] * 2

    def test_recompute_replays_disabled_autocast(self):
        # The forward pass runs without autocast and the backward pass inside an autocast
        # region, which autograd propagates to the backward pass: the recompute still runs
        # without autocast, like the forward pass. (The backward kernels themselves run under
        # the caller's autocast here, so the gradients are not compared.)
        x, weight = _inputs()
        dtypes = []
        output = tensor_parallel.checkpoint(
            partial(_rms_norm_linear, dtypes=dtypes), False, x, weight
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _loss(output).backward()

        assert dtypes == [(torch.float32, torch.float32)] * 2

    def test_no_autocast_context_without_autocast(self, monkeypatch):
        entered = []
        autocast = torch.autocast

        def recording_autocast(*args, **kwargs):
            entered.append((args, kwargs))
            return autocast(*args, **kwargs)

        ref_output, ref_x_grad, ref_weight_grad = _reference(dtype=None)
        monkeypatch.setattr(torch, "autocast", recording_autocast)

        x, weight = _inputs()
        output = tensor_parallel.checkpoint(_rms_norm_linear, False, x, weight)
        _loss(output).backward()

        # With autocast off in both passes, the replay does not touch autocast at all.
        assert entered == []
        _assert_bitwise_equal(output, ref_output)
        _assert_bitwise_equal(x.grad, ref_x_grad)
        _assert_bitwise_equal(weight.grad, ref_weight_grad)
