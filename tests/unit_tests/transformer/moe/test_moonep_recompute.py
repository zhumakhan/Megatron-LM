# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend under full block-wise activation recompute, with and without CPU offloading.

Full recompute with ``recompute_method='block'`` is the only MoE activation recompute MoonEP
supports. The first forward of a checkpointed MoE layer plans, stages its expert weights in the
pools shared by every MoE layer, pushes the copies peers need and records its plan; the
recompute in the backward dispatches with that plan instead of planning again, and pushes only
if another layer used the pools in between. Replaying the plan only removes work, so every test
compares against the same configuration recomputed without the replay (the recompute plans and
pushes again) and expects bit-identical outputs and gradients. CPU offloading only moves
bytes, so its reference is the same run without offloading. A recompute replays only when its
forward is the layer's one checkpointed forward still waiting for a recompute; with more
waiting it plans again.

The layer-level tests checkpoint one MoE layer the way block recompute checkpoints every layer
of its window: ``tensor_parallel.checkpoint`` around the layer (``megatron/core/recompute.py``).

A recompute is not compared bitwise with a run without recompute: the checkpoint's first
forward runs without grad, and the no-grad forward of a layer already differs from the grad
forward in the last bits (for every dispatcher), so the loss computed from it does too.

Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>`` and without
``-x``: checks are collected and raised on every rank together (see ``RankChecks``).
"""

import logging
from contextlib import contextmanager
from unittest import mock

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.moonep_backend import manager as moonep_manager
from megatron.core.transformer.moe.moonep_backend.grads import finalize_moonep_step
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import (
    HIDDEN,
    NUM_EXPERTS,
    RankChecks,
    build_moe_layer,
    copy_moonep_layer,
    moonep_calls,
    moonep_config,
    requires_moonep,
    teardown_moonep,
)
from tests.unit_tests.transformer.moe.test_moonep_training import (
    build_model,
    forward_backward,
    gpt_config,
    make_batches,
    moonep_warnings,
)

pytestmark = [pytest.mark.internal, requires_moonep]

SEQ, MBS = 128, 2
MAX_TOKENS = SEQ * MBS
BLOCK = dict(recompute_granularity="full", recompute_method="block", recompute_num_layers=1)


@contextmanager
def without_plan_replay():
    """Recompute without replaying the plan: the recompute plans and pushes again."""
    with mock.patch.object(moonep_manager, "is_checkpointing", lambda: False):
        yield


def _buffer(layer):
    return layer.token_dispatcher._comm_manager.buffer


def _inputs(seed_offset=0):
    rank = torch.distributed.get_rank()
    gen = torch.Generator(device="cuda").manual_seed(2000 + 17 * seed_offset + rank)
    x = torch.randn(SEQ, MBS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    grad = torch.randn(SEQ, MBS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    return x, grad


def _forward(layer, hidden_states):
    """The layer's output; in training under an activation checkpoint, as block recompute runs
    every layer of its window."""
    if not layer.training:
        return layer(hidden_states)[0]
    return tensor_parallel.checkpoint(lambda h: layer(h)[0], False, hidden_states)


def _run(layer, x, grad_out):
    inp = x.clone().requires_grad_(True)
    out = _forward(layer, inp)
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


def _check_same_grads(checks, got_layer, want_layer, what):
    """Router and expert weight gradients, bit for bit."""
    want_params = dict(want_layer.named_parameters())
    for name, param in got_layer.named_parameters():
        checks.equal(f"{what}: {name} grad", param.grad, want_params[name].grad)


def _run_schedule(layer, schedule):
    """Run micro-batch ``i``'s forward (``Fi``), its backward (``Bi``) or a training-mode
    forward under ``torch.no_grad()`` that never gets a backward (``Gi``) in schedule order.
    Returns each micro-batch's output and input grad (None for ``Gi``)."""
    waiting, results = {}, {}
    for step in schedule.split():
        kind, i = step[0], int(step[1:])
        x, grad = _inputs(seed_offset=20 + i)
        if kind == "F":
            inp = x.clone().requires_grad_(True)
            out = _forward(layer, inp)
            waiting[i] = (inp, out, grad)
        elif kind == "G":
            with torch.no_grad():
                out = _forward(layer, x)
            results[i] = (out, None)
        else:
            inp, out, grad = waiting.pop(i)
            out.backward(grad)
            results[i] = (out.detach(), inp.grad.detach())
    return results


def _expected_calls(schedule, replays):
    """Calls of one layer alone in the pools. Every forward and every recompute that does not
    replay plans and pushes its weights; a replay finds its forward's copies still in the pools
    (nothing else used them since) and the backward finds the recompute's, so neither pushes.
    Every backward reduces the expert gradients once."""
    steps = schedule.split()
    forwards = sum(step[0] in "FG" for step in steps)
    backwards = sum(step[0] == "B" for step in steps)
    fresh = forwards + backwards - replays
    return {
        "fresh": fresh,
        "replay": replays,
        "combine_backward": backwards,
        "prefetch": fresh,
        "reduce": backwards,
    }


class TestMoonEPLayerRecompute:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(123)
        self.ep = Utils.world_size

    def teardown_method(self, method):
        teardown_moonep()

    def _pair(self):
        """Two layers of a block-recompute config with identical weights."""
        replayed = build_moe_layer(moonep_config(self.ep, MAX_TOKENS, **BLOCK))
        reference = build_moe_layer(moonep_config(self.ep, MAX_TOKENS, **BLOCK))
        copy_moonep_layer(replayed, reference)
        return replayed, reference

    @pytest.mark.parametrize("skewed", [False, True], ids=["balanced", "skewed"])
    def test_replay_matches_recompute_without_replay(self, skewed):
        replayed, reference = self._pair()
        x, grad = _inputs()
        if skewed:
            x = x + 4.0 * _skew_routing(replayed, reference).to(x.dtype)
        with without_plan_replay(), moonep_calls(_buffer(reference)) as ref_calls:
            out_ref, dx_ref = _run(reference, x, grad)
        manager = replayed.token_dispatcher._comm_manager
        with moonep_calls(manager.buffer) as calls:
            out, dx = _run(replayed, x, grad)
        checks = RankChecks()
        checks.equal("output", out, out_ref)
        checks.equal("input grad", dx, dx_ref)
        _check_same_grads(checks, replayed, reference, "1 micro-batch")
        # Without the replay the recompute plans and pushes a second time. With it, it finds the
        # checkpointed forward's copies still in the pools and pushes nothing.
        expected_ref = {"fresh": 2, "replay": 0, "combine_backward": 1, "prefetch": 2, "reduce": 1}
        expected = {"fresh": 1, "replay": 1, "combine_backward": 1, "prefetch": 1, "reduce": 1}
        checks.check(ref_calls == expected_ref, f"reference calls {ref_calls}")
        checks.check(calls == expected, f"MoonEP calls {calls}, expected {expected}")
        state = replayed.experts.moonep_state
        checks.check(state.recompute_tokens == SEQ * MBS, f"{state.recompute_tokens} tokens")
        checks.check(int(state.recompute_mismatches) == 0, "the recompute routed differently")
        checks.check(manager._checkpointed is None, "the record outlived its recompute")
        checks.check(state.pending_checkpoints == 0, "a checkpointed forward still waits")
        checks.check(not state.fwd_outstanding, "forward still outstanding")
        if skewed:
            copied = (manager.last_plan.experts_to_copy >= 0).any()
            checks.check(bool(copied), "skewed routing copied no expert")
        checks.assert_all()

    def test_accumulation_matches_recompute_without_replay(self):
        replayed, reference = self._pair()
        for micro_batch in range(3):
            x, grad = _inputs(seed_offset=10 + micro_batch)
            with without_plan_replay():
                out_ref, dx_ref = _run(reference, x, grad)
            out, dx = _run(replayed, x, grad)
        checks = RankChecks()
        checks.equal("last output", out, out_ref)
        checks.equal("last input grad", dx, dx_ref)
        _check_same_grads(checks, replayed, reference, "3 micro-batches")
        replayed_tokens = replayed.experts.moonep_state.recompute_tokens
        checks.check(replayed_tokens == 3 * SEQ * MBS, f"{replayed_tokens} replayed tokens")
        checks.assert_all()

    def test_eval_forward_before_recompute_drops_the_record(self):
        """An eval-mode (not checkpointed) forward of the layer between a checkpointed forward
        and its recompute pushes over the recorded copies, so it drops the record and the
        recompute plans and pushes again."""
        replayed, reference = self._pair()
        x, grad = _inputs()
        other, _ = _inputs(seed_offset=3)

        def run(layer):
            inp = x.clone().requires_grad_(True)
            out = _forward(layer, inp)
            layer.eval()
            with torch.no_grad():
                layer(other)
            layer.train()
            out.backward(grad)
            return out.detach(), inp.grad.detach()

        with without_plan_replay():
            out_ref, dx_ref = run(reference)
        with moonep_calls(_buffer(replayed)) as calls:
            out, dx = run(replayed)
        checks = RankChecks()
        checks.equal("output", out, out_ref)
        checks.equal("input grad", dx, dx_ref)
        _check_same_grads(checks, replayed, reference, "forward in between")
        expected = {"fresh": 3, "replay": 0, "combine_backward": 1, "prefetch": 3, "reduce": 1}
        checks.check(calls == expected, f"MoonEP calls {calls}, expected {expected}")
        checks.assert_all()

    @pytest.mark.parametrize(
        "schedule, replays, still_waiting",
        [
            ("F1 F2 B1 B2", 0, 0),
            ("F1 F2 B2 B1", 0, 0),
            ("F1 F2 B1 F3 B2 B3", 0, 0),
            ("F1 G2 B1", 0, 1),
            ("F1 F2 B1 B2 F3 B3", 1, 0),
        ],
        ids=["fifo", "lifo", "lookahead", "no-grad-forward-between", "overlap-then-alone"],
    )
    def test_several_waiting_checkpoints_plan_afresh(self, schedule, replays, still_waiting):
        """With two checkpointed forwards of the layer waiting for their recomputes, the record
        may be the other forward's, so the recomputes plan and push again. A recompute replays
        again once its forward is the only one waiting. Results match the recompute without
        replay either way."""
        replayed, reference = self._pair()
        with without_plan_replay():
            want = _run_schedule(reference, schedule)
        with moonep_calls(_buffer(replayed)) as calls:
            got = _run_schedule(replayed, schedule)
        checks = RankChecks()
        for i, (out_ref, dx_ref) in want.items():
            checks.equal(f"micro-batch {i} output", got[i][0], out_ref)
            if dx_ref is not None:
                checks.equal(f"micro-batch {i} input grad", got[i][1], dx_ref)
        _check_same_grads(checks, replayed, reference, schedule)
        expected = _expected_calls(schedule, replays)
        checks.check(calls == expected, f"MoonEP calls {calls}, expected {expected}")
        waiting = replayed.experts.moonep_state.pending_checkpoints
        checks.check(waiting == still_waiting, f"{waiting} checkpointed forwards still wait")
        checks.assert_all()

    def test_finalize_forgets_a_forward_without_backward(self):
        """A training-mode forward under no_grad is checkpointed but never recomputed, so later
        recomputes plan and push again until finalize_model_grads, where every backward of the
        step has run, forgets it."""
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS, **BLOCK))
        state = layer.experts.moonep_state
        _run_schedule(layer, "F1 G2 B1")
        checks = RankChecks()
        checks.check(state.pending_checkpoints == 1, f"{state.pending_checkpoints} waiting")
        with moonep_calls(_buffer(layer)) as calls:
            _run_schedule(layer, "F3 B3")
        expected = _expected_calls("F3 B3", replays=0)
        checks.check(calls == expected, f"before finalize: calls {calls}, expected {expected}")
        finalize_moonep_step([layer])
        checks.check(state.pending_checkpoints == 0, f"{state.pending_checkpoints} waiting")
        with moonep_calls(_buffer(layer)) as calls:
            _run_schedule(layer, "F4 B4")
        expected = _expected_calls("F4 B4", replays=1)
        checks.check(calls == expected, f"after finalize: calls {calls}, expected {expected}")
        checks.assert_all()

    def test_stuck_checkpoint_flag_does_not_replay_outside_backward(self):
        """Megatron sets and clears its checkpoint flag without try/finally, so an exception
        inside a checkpoint leaves it set. A no-grad forward then counts as a checkpoint's first
        forward and records its plan, but a later forward outside any backward is still no
        recompute: it plans and pushes, with the results of a run without the stuck flag."""
        replayed, reference = self._pair()
        replayed.eval()
        reference.eval()
        x, grad = _inputs()
        other, _ = _inputs(seed_offset=3)
        out_ref, dx_ref = _run(reference, x, grad)
        with mock.patch.object(moonep_manager, "is_checkpointing", lambda: True):
            with torch.no_grad():
                _forward(replayed, other)
            recorded = replayed.token_dispatcher._comm_manager._checkpointed is not None
            with moonep_calls(_buffer(replayed)) as calls:
                out, dx = _run(replayed, x, grad)
        checks = RankChecks()
        checks.check(recorded, "the no-grad forward recorded no plan")
        checks.equal("output", out, out_ref)
        checks.equal("input grad", dx, dx_ref)
        _check_same_grads(checks, replayed, reference, "stuck checkpoint flag")
        expected = {"fresh": 1, "replay": 0, "combine_backward": 1, "prefetch": 1, "reduce": 1}
        checks.check(calls == expected, f"MoonEP calls {calls}, expected {expected}")
        checks.assert_all()

    @pytest.mark.parametrize("flipped", [1, SEQ * MBS], ids=["one-token", "every-token"])
    def test_routing_mismatch_is_reported(self, flipped, caplog):
        """A recompute that sends tokens to other experts than its forward is counted on the
        device. finalize (finalize_moonep_step) warns when more than 1% of the tokens
        differ, which means the recompute did not reproduce its forward; a near tie flipped
        by the no-grad/grad last-bit differences of a single token stays quiet."""
        replayed, _ = self._pair()
        replayed.set_layer_number(5)
        x, grad = _inputs()
        inp = x.clone().requires_grad_(True)
        out = _forward(replayed, inp)
        record = replayed.token_dispatcher._comm_manager._checkpointed
        # Move the first `flipped` tokens' recorded experts to other experts.
        indices = record.token_indices.clone()
        indices[:flipped] = (indices[:flipped] + 1) % NUM_EXPERTS
        record.token_indices = indices
        out.backward(grad)
        state = replayed.experts.moonep_state
        mismatches = int(state.recompute_mismatches)
        caplog.set_level(logging.WARNING)
        finalize_moonep_step([replayed])
        notices = [r.getMessage() for r in moonep_warnings(caplog, "other experts")]
        checks = RankChecks()
        checks.check(mismatches == flipped, f"{mismatches} mismatched tokens counted")
        if flipped == SEQ * MBS:
            checks.check(len(notices) == 1, f"notices: {notices}")
            expected_text = f"sent {flipped} of {SEQ * MBS} tokens"
            checks.check(
                all("MoE layer 5 (rank" in n and expected_text in n for n in notices),
                f"notice text: {notices}",
            )
        else:
            checks.check(notices == [], f"notices: {notices}")
        checks.check(int(state.recompute_mismatches) == 0, "counter not reset")
        checks.check(state.recompute_tokens == 0, "token count not reset")
        checks.assert_all()


def _model_grads(model):
    """fp32 grads by name: DDP's main grads, MoonEP expert weights included."""
    return {
        name: param.main_grad.detach().clone()
        for name, param in model.named_parameters()
        if getattr(param, "main_grad", None) is not None
    }


@torch.no_grad()
def _copy_gpt(src_gpt, dst_gpt):
    dst_params = dict(dst_gpt.named_parameters())
    for name, param in src_gpt.named_parameters():
        dst_params[name].copy_(param)
    torch.cuda.synchronize()
    torch.distributed.barrier()


NUM_LAYERS = 4
FULL_BLOCK_2 = dict(recompute_granularity="full", recompute_method="block", recompute_num_layers=2)
MODEL_VARIANTS = {
    "full-block": FULL_BLOCK_2,
    # Offload window inside the checkpointed window: only checkpoint inputs move.
    "full-block-offload": dict(
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=3,
        cpu_offloading=True,
        cpu_offloading_num_layers=3,
    ),
    # Offload window past the checkpointed window: layer 2 offloads its MoonEP activations.
    "full-block-offload-past": dict(
        FULL_BLOCK_2,
        cpu_offloading=True,
        cpu_offloading_num_layers=3,
        cpu_offloading_prefetch_num_layers=2,
    ),
}
CHECKPOINTED_MOE_LAYERS = {"full-block": 2, "full-block-offload": 3, "full-block-offload-past": 2}


class TestMoonEPModelRecompute:
    def setup_method(self, method):
        self.ep = Utils.world_size
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=self.ep)

    def teardown_method(self, method):
        destroy_moe_metrics_tracker()
        teardown_moonep()

    def test_block_recompute_matches_no_recompute(self, caplog):
        """Full block recompute changes neither the loss nor any gradient, bit for bit. The
        checkpointed first forward runs without grad and its recompute with grad, and both
        compute exactly what a forward without recompute computes; this needs the router to
        sort its top-k in every training forward (an unsorted no-grad top-k rounds the routing
        probabilities differently)."""
        reference, reference_gpt = build_model(
            gpt_config(self.ep, "moonep", num_layers=NUM_LAYERS), seed=1234, distopt=False
        )
        model, gpt = build_model(
            gpt_config(self.ep, "moonep", num_layers=NUM_LAYERS, **FULL_BLOCK_2),
            seed=1234,
            distopt=False,
        )
        _copy_gpt(reference_gpt, gpt)
        batches = make_batches(2, seed=7)
        caplog.set_level(logging.WARNING)
        reference_loss = forward_backward(reference, batches)
        loss = forward_backward(model, batches)
        checks = RankChecks()
        checks.check(loss == reference_loss, f"loss {loss!r}, without recompute {reference_loss!r}")
        got, want = _model_grads(model), _model_grads(reference)
        checks.check(got.keys() == want.keys() and len(got) > 0, "different parameters")
        for name in want:
            checks.equal(f"grad of {name}", got[name], want[name])
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()

    @pytest.mark.parametrize("variant", list(MODEL_VARIANTS))
    def test_matches_reference(self, variant, caplog):
        """Same loss and gradients as the variant without CPU offloading and without the plan
        replay. The MoE layers share one pool set, so per micro-batch every forward pushes and
        every backward but the last forward's layer pushes again; a checkpointed layer's
        recompute push replaces its backward one."""
        overrides = MODEL_VARIANTS[variant]
        if overrides.get("cpu_offloading") and not HAVE_TE:
            pytest.skip("CPU offloading needs Transformer Engine")
        reference_overrides = {k: v for k, v in overrides.items() if "cpu_offloading" not in k}
        reference, reference_gpt = build_model(
            gpt_config(self.ep, "moonep", num_layers=NUM_LAYERS, **reference_overrides),
            seed=1234,
            distopt=False,
        )
        model, gpt = build_model(
            gpt_config(self.ep, "moonep", num_layers=NUM_LAYERS, **overrides),
            seed=1234,
            distopt=False,
        )
        _copy_gpt(reference_gpt, gpt)
        batches = make_batches(2, seed=5)
        caplog.set_level(logging.WARNING)
        with without_plan_replay():
            reference_loss = forward_backward(reference, batches)
        with moonep_calls(_buffer(gpt.decoder.layers[0].mlp)) as calls:
            loss = forward_backward(model, batches)
        checks = RankChecks()
        checks.check(loss == reference_loss, f"loss {loss!r} vs {reference_loss!r}")
        got, want = _model_grads(model), _model_grads(reference)
        checks.check(got.keys() == want.keys() and len(got) > 0, "different parameters")
        for name in want:
            checks.equal(f"grad of {name}", got[name], want[name])
        # Per micro-batch: every MoE layer plans once (checkpointed ones replay that plan in
        # their recompute), pushes 2 * layers - 1 times in all, and reduces once.
        expected = {
            "fresh": 2 * NUM_LAYERS,
            "replay": 2 * CHECKPOINTED_MOE_LAYERS[variant],
            "combine_backward": 2 * NUM_LAYERS,
            "prefetch": 2 * (2 * NUM_LAYERS - 1),
            "reduce": 2 * NUM_LAYERS,
        }
        checks.check(calls == expected, f"MoonEP calls {calls}, expected {expected}")
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()
