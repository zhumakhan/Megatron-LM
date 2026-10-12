# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend with a different token count on every EP rank.

Each rank passes its own ``s`` tokens and the EP group's total as ``MoELayer.forward(...,
total_num_tokens=...)``; MoonEP then sizes every rank's receive slots from the total instead of
from ``R * s``. The alltoall dispatcher with SequentialMLP experts accepts a different count on
every rank as is and is the reference, as in ``test_moonep_layer.py``. The counts include the
capacity and a single token, and between micro-batches they change on some ranks only, which
the first-use barriers of the fresh and the replayed dispatch must not depend on.

Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>`` and without
``-x``: checks are collected and raised on every rank together (see ``RankChecks``).
"""

import logging
from contextlib import contextmanager
from unittest import mock

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.moe.moonep_backend import manager as moonep_manager
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import (
    HIDDEN,
    NUM_EXPERTS,
    RankChecks,
    alltoall_config,
    build_moe_layer,
    copy_moonep_layer,
    copy_sequential_to_moonep,
    moonep_config,
    moonep_expert_grads,
    requires_moonep,
    sequential_expert_grads,
    teardown_moonep,
)

pytestmark = [pytest.mark.internal, requires_moonep]

MAX_TOKENS = 256
TOL = 1e-2
BLOCK = dict(recompute_granularity="full", recompute_method="block", recompute_num_layers=1)
# Tokens on each EP rank (cycled beyond 8 ranks): the capacity, a single token, uneven counts.
COUNTS = (MAX_TOKENS, 1, 129, 77, 200, 3, 255, 64)
# The next micro-batch: even ranks keep their count, odd ranks get one they have not used yet.
NEXT_COUNTS = (MAX_TOKENS, 20, 129, 40, 200, 34, 255, 205)


def _tokens(counts):
    """This rank's token count and the EP group's (here WORLD's) total, summed as a caller
    would."""
    s = counts[torch.distributed.get_rank() % len(counts)]
    total = torch.tensor([s], device="cuda")
    torch.distributed.all_reduce(total)
    return s, int(total.item())


def _inputs(s, seed_offset=0):
    rank = torch.distributed.get_rank()
    gen = torch.Generator(device="cuda").manual_seed(3000 + 17 * seed_offset + rank)
    x = torch.randn(s, 1, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    grad = torch.randn(s, 1, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    return x, grad


def _run(layer, x, grad_out, **kwargs):
    inp = x.clone().requires_grad_(True)
    out, _ = layer(inp, **kwargs)
    out.backward(grad_out)
    return out.detach(), inp.grad.detach()


def _run_checkpointed(layer, x, grad_out, total):
    """The layer under an activation checkpoint, as block recompute runs every layer."""
    inp = x.clone().requires_grad_(True)
    out = tensor_parallel.checkpoint(lambda h: layer(h, total_num_tokens=total)[0], False, inp)
    out.backward(grad_out)
    return out.detach(), inp.grad.detach()


@torch.no_grad()
def _skew_routing(*layers):
    """Route every token to the experts of the upper half of the EP ranks (prefetch runs)."""
    hot = torch.arange(NUM_EXPERTS, device="cuda") >= NUM_EXPERTS // 2
    direction = torch.ones(HIDDEN, device="cuda") / HIDDEN**0.5
    for layer in layers:
        weight = layer.router.weight
        weight.mul_(0.01)
        weight.add_(torch.where(hot, 1.0, -1.0)[:, None] * direction.to(weight.dtype))
    return direction


def _check_grads_against_reference(checks, moon, base):
    """Router and expert weight gradients of a MoonEP layer against its alltoall reference."""
    checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
    for i, (got, want) in enumerate(zip(moonep_expert_grads(moon), sequential_expert_grads(base))):
        for name, g, w in zip(("gate", "up", "down"), got, want):
            if w.norm() == 0:
                checks.check(torch.count_nonzero(g) == 0, f"expert {i} {name}: expected zero grad")
            else:
                checks.close(f"expert {i} {name} grad", g, w, TOL)


@contextmanager
def _dispatches(layer):
    """The Buffer's dispatch calls: a fresh one carries no ``plan``, a replay carries the plan
    and the tokens, the backward of combine the plan and the output gradient only."""
    buffer = layer.token_dispatcher._comm_manager.buffer
    with mock.patch.object(buffer, "dispatch", wraps=buffer.dispatch) as dispatch:
        yield dispatch.call_args_list


class TestMoonEPPerRankTokens:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(123)
        self.ep = Utils.world_size

    def teardown_method(self, method):
        teardown_moonep()

    def _pair(self):
        base = build_moe_layer(alltoall_config(self.ep))
        moon = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        copy_sequential_to_moonep(base, moon)
        return base, moon

    @pytest.mark.parametrize("skewed", [False, True], ids=["balanced", "skewed"])
    def test_parity_vs_alltoall(self, skewed, caplog):
        base, moon = self._pair()
        s, total = _tokens(COUNTS)
        x, grad = _inputs(s)
        if skewed:
            x = x + 4.0 * _skew_routing(base, moon).to(x.dtype)
        caplog.set_level(logging.WARNING)
        out_b, dx_b = _run(base, x, grad)
        with _dispatches(moon) as calls:
            out_m, dx_m = _run(moon, x, grad, total_num_tokens=total)
        checks = RankChecks()
        checks.close("output", out_m, out_b, TOL)
        checks.close("input grad", dx_m, dx_b, TOL)
        _check_grads_against_reference(checks, moon, base)
        fresh = [c for c in calls if c.kwargs.get("plan") is None]
        checks.check(len(fresh) == 1, f"{len(fresh)} fresh dispatches, expected 1")
        got = [c.kwargs.get("total_num_tokens") for c in fresh]
        checks.check(got == [total], f"MoonEP got total_num_tokens {got}, expected {total}")
        manager = moon.token_dispatcher._comm_manager
        plan, nvs = manager.last_plan, int(manager.buffer._require_ctx()["NvS"])
        checks.check(plan.num_tokens == s, f"plan.num_tokens={plan.num_tokens}, s={s}")
        # The receive slots follow the total, so every rank gets the same step-sized view.
        nvs_s = torch.tensor([plan.nvs_s], device="cuda")
        every = [torch.empty_like(nvs_s) for _ in range(self.ep)]
        torch.distributed.all_gather(every, nvs_s)
        every = [int(t) for t in every]
        checks.check(len(set(every)) == 1, f"plan.nvs_s differs between ranks: {every}")
        checks.check(plan.nvs_s <= nvs, f"plan.nvs_s={plan.nvs_s} exceeds NvS={nvs}")
        if skewed:
            checks.check(bool((plan.experts_to_copy >= 0).any()), "no expert was copied")
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and "MoonEP" in r.getMessage()
        ]
        checks.check(not warnings, f"warnings: {warnings}")
        checks.check(not moon.experts.moonep_state.fwd_outstanding, "forward still outstanding")
        checks.assert_all()

    def test_counts_change_between_micro_batches(self):
        # Odd ranks get a count they have not used yet while even ranks repeat theirs: a first-use
        # barrier keyed by the local count would hold the odd ranks only and hang.
        base, moon = self._pair()
        checks = RankChecks()
        for step, counts in enumerate((COUNTS, NEXT_COUNTS, COUNTS)):
            s, total = _tokens(counts)
            x, grad = _inputs(s, seed_offset=10 + step)
            out_b, dx_b = _run(base, x, grad)
            out_m, dx_m = _run(moon, x, grad, total_num_tokens=total)
            checks.close(f"micro-batch {step} output", out_m, out_b, TOL)
            checks.close(f"micro-batch {step} input grad", dx_m, dx_b, TOL)
        # Both layers accumulated the gradients of all three micro-batches.
        _check_grads_against_reference(checks, moon, base)
        checks.assert_all()

    def test_block_recompute_replays_the_plan(self):
        # The recompute in the backward reuses its forward's plan, which carries this rank's s and
        # the step-sized views. It must match a recompute that plans again, bit for bit, also
        # when the counts change on some ranks only (the replay's first-use barrier).
        replayed = build_moe_layer(moonep_config(self.ep, MAX_TOKENS, **BLOCK))
        reference = build_moe_layer(moonep_config(self.ep, MAX_TOKENS, **BLOCK))
        copy_moonep_layer(replayed, reference)
        checks = RankChecks()
        for step, counts in enumerate((COUNTS, NEXT_COUNTS)):
            s, total = _tokens(counts)
            x, grad = _inputs(s, seed_offset=20 + step)
            with mock.patch.object(moonep_manager, "is_checkpointing", lambda: False):
                out_ref, dx_ref = _run_checkpointed(reference, x, grad, total)
            with _dispatches(replayed) as calls:
                out, dx = _run_checkpointed(replayed, x, grad, total)
            replays = [c for c in calls if c.kwargs.get("plan") is not None and len(c.args) > 1]
            checks.check(len(replays) == 1, f"micro-batch {step}: {len(replays)} replays")
            checks.equal(f"micro-batch {step} output", out, out_ref)
            checks.equal(f"micro-batch {step} input grad", dx, dx_ref)
        reference_params = dict(reference.named_parameters())
        for name, param in replayed.named_parameters():
            checks.equal(f"{name} grad", param.grad, reference_params[name].grad)
        checks.assert_all()
