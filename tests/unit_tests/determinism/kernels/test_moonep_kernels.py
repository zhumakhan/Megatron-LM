# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the MoonEP flex-dispatcher backend.

Kernel level (one GPU, no ``moonep`` needed): ``GroupedMMLiveGrad``, the grouped GEMM over the
composite ``[E+B, out, in]`` pool that computes dW over the live groups only and adds it into
fp32 targets -- checked against a per-group fp32 reference (empty groups give exact zeros, rows
past the last segment are never read) and replayed under stream contention. Module level
(``moonep`` and >= 2 sm_90+ GPUs of one NVLink domain): an ``MoELayer`` on MoonEP, with
balanced and skewed routing so the weight-prefetch and grad-reduce kernels run, replayed with
``zero_moe_grads()`` between replays (never ``zero_grad(set_to_none=True)``, which would drop
the masters' persistent grad aliases).
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

# E experts, B prefetch slots; this "rank" owns experts [LO, HI).
E, B, LO, HI, IN, OUT = 16, 4, 4, 8, 256, 384
# Rows per group; every group outside [LO, HI) and [E, E+B) is empty, as on a MoonEP rank.
SIZES = [0] * LO + [256, 0, 128, 384] + [0] * (E - HI) + [128, 0, 256, 128]
TAIL = 64  # rows past the last segment: uninitialized in MoonEP, NaN here


def _grouped_mm_inputs():
    seeded()
    offs = torch.tensor(SIZES, device="cuda").cumsum(0).to(torch.int32)
    rows = int(offs[-1])
    x = torch.randn(rows + TAIL, IN, device="cuda", dtype=torch.bfloat16)
    x[rows:] = float("nan")
    weight = torch.randn(E + B, OUT, IN, device="cuda", dtype=torch.bfloat16) / IN**0.5
    grad = torch.randn(rows + TAIL, OUT, device="cuda", dtype=torch.bfloat16)
    grad[rows:] = 0
    offs_live = torch.cat((offs[LO:HI], offs[E:]))
    return x, weight, grad, offs, offs_live, rows


def _grouped_mm_run(x, weight, grad, offs, offs_live, rows):
    own = torch.zeros(HI - LO, OUT, IN, device="cuda", dtype=torch.float32)
    slots = torch.zeros(B, OUT, IN, device="cuda", dtype=torch.float32)
    inp = x.clone().requires_grad_(True)
    out = GroupedMMLiveGrad.apply(inp, weight, offs, offs_live, (own, slots))
    out.backward(grad)
    torch.cuda.synchronize()
    return out.detach()[:rows], inp.grad[:rows], own, slots


def test_grouped_mm_live_grad_matches_reference():
    x, weight, grad, offs, offs_live, rows = _grouped_mm_inputs()
    out, dx, own, slots = _grouped_mm_run(x, weight, grad, offs, offs_live, rows)
    live = list(range(LO, HI)) + list(range(E, E + B))
    dw = torch.cat((own, slots))
    start = 0
    for group, size in enumerate(SIZES):
        end = start + size
        w = weight[group].float()
        if size:
            xs, gs = x[start:end].float(), grad[start:end].float()
            torch.testing.assert_close(out[start:end].float(), xs @ w.T, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(dx[start:end].float(), gs @ w, rtol=2e-2, atol=2e-2)
        if group in live:
            i = live.index(group)
            if size:
                ref = grad[start:end].float().T @ x[start:end].float()
                torch.testing.assert_close(dw[i], ref, rtol=2e-2, atol=5e-2)
            else:
                assert torch.count_nonzero(dw[i]) == 0, f"empty live group {group} got a dW"
        start = end
    assert not torch.isnan(out).any() and not torch.isnan(dx).any() and not torch.isnan(dw).any()


def test_grouped_mm_live_grad_replays():
    inputs = _grouped_mm_inputs()
    reference = _grouped_mm_run(*inputs)
    for replay in range(1, 4):
        with RacingStreams():
            result = _grouped_mm_run(*inputs)
        for name, got, want in zip(("out", "dx", "dw_own", "dw_slots"), result, reference):
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

    @pytest.mark.parametrize("skewed", [False, True], ids=["balanced", "skewed"])
    def test_moe_layer_replays(self, skewed):
        from tests.unit_tests.transformer.moe.moonep_utils import (
            HIDDEN,
            NUM_EXPERTS,
            RankChecks,
            build_moe_layer,
            moonep_config,
        )

        seq, mbs = 256, 2
        layer = build_moe_layer(moonep_config(Utils.world_size, seq * mbs))
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

        def run():
            layer.zero_moe_grads()
            layer.router.weight.grad = None
            inp = x.clone().requires_grad_(True)
            out, _ = layer(inp)
            out.backward(grad)
            torch.cuda.synchronize()
            snapshot = {"out": out.detach(), "dx": inp.grad, "router": layer.router.weight.grad}
            for param, name in zip(
                layer.experts.moonep_master_parameters(), ("gate", "up", "down")
            ):
                snapshot[f"dw_{name}"] = param.grad.detach().clone()
            return snapshot

        checks = RankChecks()
        reference = run()
        plan = layer.token_dispatcher._comm_manager.last_plan
        copied = torch.tensor(int((plan.experts_to_copy >= 0).any()), device="cuda")
        torch.distributed.all_reduce(copied)
        checks.check(not skewed or int(copied) > 0, "skewed routing copied no expert")
        for replay in range(1, 4):
            with RacingStreams():
                result = run()
            for name, want in reference.items():
                checks.check(
                    bytes_equal(result[name], want), f"MoonEP MoELayer replay {replay}: {name}"
                )
        checks.assert_all()
