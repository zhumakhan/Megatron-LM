# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the MoonEP flex-dispatcher backend.

Kernel level (one GPU, no ``moonep`` needed): ``GroupedMMLiveGrad``, the grouped GEMM over the
``[2*epn, out, in]`` pool (this rank's experts, then its prefetch slots) whose backward
overwrites one fp32 ``[2*epn, out, in]`` target with the bf16 dW of every group. Checked against
a per-group fp32 reference on the own-expert and the slot rows of the target (empty groups give
exact zeros, rows past the last segment are never read, the target's old contents do not
matter), without a target (frozen experts: dX only, no dW GEMM), and replayed under stream
contention.

Module level (``moonep`` and >= 2 sm_90+ GPUs of one NVLink domain): an ``MoELayer`` on MoonEP,
with balanced and skewed routing so the weight-prefetch and grad-reduce kernels run, with and
without an activation checkpoint around the layer as block recompute places it (whose backward
replays the forward's plan). The expert weights are ordinary parameters, so the replays reset
gradients the standard way (``param.grad = None``) and compare the output, the input and router
gradients and every expert weight's gradient (``linear_fc{1,2}.weight{i}``).
"""

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moonep_backend.autograd import GroupedMMLiveGrad
from tests.unit_tests.determinism.kernels.harness import bytes_equal, seeded
from tests.unit_tests.determinism.utils import RacingStreams
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="needs an sm_90+ GPU (torch._grouped_mm)",
)

# EPN experts on this "rank" and as many prefetch slots: groups [0, EPN) and [EPN, 2*EPN).
EPN, IN, OUT = 4, 256, 384
# Rows per group, with an empty own expert and an empty slot, as on a MoonEP rank.
SIZES = [256, 0, 128, 384] + [128, 0, 256, 128]
TAIL = 64  # rows past the last segment: uninitialized in MoonEP, NaN here


def _grouped_mm_inputs():
    seeded()
    offs = torch.tensor(SIZES, device="cuda").cumsum(0).to(torch.int32)
    rows = int(offs[-1])
    x = torch.randn(rows + TAIL, IN, device="cuda", dtype=torch.bfloat16)
    x[rows:] = float("nan")
    weight = torch.randn(2 * EPN, OUT, IN, device="cuda", dtype=torch.bfloat16) / IN**0.5
    grad = torch.randn(rows + TAIL, OUT, device="cuda", dtype=torch.bfloat16)
    grad[rows:] = 0
    return x, weight, grad, offs, rows


def _grouped_mm_run(x, weight, grad, offs, rows, fill=float("nan")):
    """Forward and backward of one projection; returns ``(out, dx, target)``.

    The fp32 dW target starts out filled with ``fill``, which the backward must overwrite
    entirely; ``fill=None`` passes no target, as for frozen experts.
    """
    target = None
    if fill is not None:
        target = torch.full((2 * EPN, OUT, IN), fill, device="cuda", dtype=torch.float32)
    inp = x.clone().requires_grad_(True)
    out = GroupedMMLiveGrad.apply(inp, weight, offs, target)
    out.backward(grad)
    torch.cuda.synchronize()
    return out.detach()[:rows], inp.grad[:rows], target


def _check_out_and_dx(x, weight, grad, offs, out, dx):
    """Compare the forward and dX with a per-group fp32 reference."""
    start = 0
    for group, end in enumerate(offs.tolist()):
        w = weight[group].float()
        xs, gs = x[start:end].float(), grad[start:end].float()
        torch.testing.assert_close(out[start:end].float(), xs @ w.T, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(dx[start:end].float(), gs @ w, rtol=2e-2, atol=2e-2)
        start = end
    assert not torch.isnan(out).any() and not torch.isnan(dx).any()


def test_grouped_mm_live_grad_matches_reference():
    x, weight, grad, offs, rows = _grouped_mm_inputs()
    out, dx, dw = _grouped_mm_run(x, weight, grad, offs, rows)
    _check_out_and_dx(x, weight, grad, offs, out, dx)
    ref = torch.zeros(2 * EPN, OUT, IN, device="cuda", dtype=torch.float32)
    start = 0
    for group, end in enumerate(offs.tolist()):
        ref[group] = grad[start:end].float().T @ x[start:end].float()
        start = end
    # The target holds this rank's experts' dW in rows [0, EPN), its slots' dW after them; no
    # row keeps the NaN it started with.
    torch.testing.assert_close(dw[:EPN], ref[:EPN], rtol=2e-2, atol=5e-2)  # own experts
    torch.testing.assert_close(dw[EPN:], ref[EPN:], rtol=2e-2, atol=5e-2)  # prefetch slots
    for group, size in enumerate(SIZES):
        if size == 0:
            assert torch.count_nonzero(dw[group]) == 0, f"empty group {group} got a dW"


def test_grouped_mm_live_grad_without_target(monkeypatch):
    """Frozen experts pass no target: the backward runs no dW GEMM, and dX is bit for bit that
    of a run with a target."""
    inputs = _grouped_mm_inputs()
    operand_dims = []
    grouped_mm = torch._grouped_mm

    def recording_grouped_mm(a, b, *args, **kwargs):
        operand_dims.append((a.dim(), b.dim()))
        return grouped_mm(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "_grouped_mm", recording_grouped_mm)
    out_ref, dx_ref, _ = _grouped_mm_run(*inputs)
    with_target = list(operand_dims)
    operand_dims.clear()
    out, dx, target = _grouped_mm_run(*inputs, fill=None)
    # The forward and dX multiply by the 3-D pool; the dW GEMM groups two 2-D operands along K.
    assert with_target.count((2, 2)) == 1, f"grouped GEMMs with a target: {with_target}"
    assert operand_dims.count((2, 2)) == 0, f"grouped GEMMs without a target: {operand_dims}"
    assert target is None
    assert bytes_equal(out, out_ref), "output differs from the run with a target"
    assert bytes_equal(dx, dx_ref), "dX differs from the run with a target"
    x, weight, grad, offs, _ = inputs
    _check_out_and_dx(x, weight, grad, offs, out, dx)


def test_grouped_mm_live_grad_replays():
    inputs = _grouped_mm_inputs()
    reference = _grouped_mm_run(*inputs)
    for replay in range(1, 4):
        # Every replay starts the target from other contents, which the backward overwrites.
        with RacingStreams():
            result = _grouped_mm_run(*inputs, fill=float(replay))
        for name, got, want in zip(("out", "dx", "dw"), result, reference):
            assert bytes_equal(got, want), f"GroupedMMLiveGrad replay {replay}: {name} differs"


class TestMoonEPLayerReplay:
    def setup_method(self, method):
        from tests.unit_tests.transformer.moe.moonep_utils import MOONEP_SKIP_REASON

        if MOONEP_SKIP_REASON is not None:
            pytest.skip(MOONEP_SKIP_REASON)
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        from tests.unit_tests.transformer.moe.moonep_utils import (
            MOONEP_SKIP_REASON,
            teardown_moonep,
        )

        if MOONEP_SKIP_REASON is None:
            teardown_moonep()

    @pytest.mark.parametrize("recompute", [False, True], ids=["plain", "recompute"])
    @pytest.mark.parametrize("skewed", [False, True], ids=["balanced", "skewed"])
    def test_moe_layer_replays(self, skewed, recompute):
        from megatron.core import tensor_parallel
        from tests.unit_tests.transformer.moe.moonep_utils import (
            HIDDEN,
            NUM_EXPERTS,
            RankChecks,
            build_moe_layer,
            moonep_config,
        )

        seq, mbs = 256, 2
        # Block recompute checkpoints every layer of its window; the backward replays the
        # checkpointed forward's plan.
        recompute_kwargs = (
            dict(recompute_granularity="full", recompute_method="block", recompute_num_layers=1)
            if recompute
            else {}
        )
        layer = build_moe_layer(moonep_config(Utils.world_size, seq * mbs, **recompute_kwargs))
        rank = torch.distributed.get_rank()
        gen = torch.Generator(device="cuda").manual_seed(4321 + rank)
        x = torch.randn(seq, mbs, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen)
        grad = torch.randn(seq, mbs, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen)
        if skewed:
            # All tokens pick experts of the upper half of the EP ranks, so MoonEP copies them to
            # the idle ranks (prefetch) and reduces their grads back (reduce_grad).
            hot = torch.arange(NUM_EXPERTS, device="cuda") >= NUM_EXPERTS // 2
            direction = torch.ones(HIDDEN, device="cuda") / HIDDEN**0.5
            with torch.no_grad():
                weight = layer.router.weight
                weight.mul_(0.01)
                weight.add_(torch.where(hot, 1.0, -1.0)[:, None] * direction.to(weight.dtype))
            x = x + 4.0 * direction.to(x.dtype)

        def forward(inp):
            if recompute:
                return tensor_parallel.checkpoint(lambda h: layer(h)[0], False, inp)
            return layer(inp)[0]

        def run():
            # The expert weights are ordinary parameters: drop every gradient the standard way.
            for param in layer.parameters():
                param.grad = None
            inp = x.clone().requires_grad_(True)
            out = forward(inp)
            out.backward(grad)
            torch.cuda.synchronize()
            # dx, the router grad and every linear_fc{1,2}.weight{i} grad (None if missing).
            grads = {"dx": inp.grad, "router": layer.router.weight.grad}
            grads.update((name, p.grad) for name, p in layer.experts.named_parameters())
            snapshot = {name: g if g is None else g.detach().clone() for name, g in grads.items()}
            snapshot["out"] = out.detach()
            return snapshot

        checks = RankChecks()
        reference = run()
        plan = layer.token_dispatcher._comm_manager.last_plan
        copied = torch.tensor(int((plan.experts_to_copy >= 0).any()), device="cuda")
        torch.distributed.all_reduce(copied)
        checks.check(not skewed or int(copied) > 0, "skewed routing copied no expert")
        num_expert_grads = sum(name.startswith("linear_fc") for name in reference)
        expected = 2 * layer.experts.num_local_experts
        checks.check(num_expert_grads == expected, f"{num_expert_grads} expert weight grads")
        for replay in range(1, 4):
            with RacingStreams():
                result = run()
            for name, want in reference.items():
                got = result[name]
                same = got is not None and want is not None and bytes_equal(got, want)
                checks.check(same, f"MoonEP MoELayer replay {replay}: {name}")
        checks.assert_all()
