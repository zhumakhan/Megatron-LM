# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend: process-global Buffer and expert pools, expert weights and the MoE layer.

Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>`` and without
``-x``: checks are collected and raised on every rank together (see ``RankChecks``).
"""

import logging

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moonep_backend import iter_moonep_experts, uses_moonep
from megatron.core.transformer.moe.moonep_backend.grads import finalize_moonep_step
from megatron.core.transformer.moe.moonep_backend.storage import MoonEPExpertPools
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import (
    FFN,
    HIDDEN,
    NUM_EXPERTS,
    TOPK,
    RankChecks,
    alltoall_config,
    build_moe_layer,
    copy_moonep_layer,
    copy_sequential_to_moonep,
    moonep_calls,
    moonep_config,
    moonep_expert_grads,
    requires_moonep,
    sequential_expert_grads,
    teardown_moonep,
)

pytestmark = [pytest.mark.internal, requires_moonep]

SEQ, MBS = 128, 2
MAX_TOKENS = SEQ * MBS
TOL = 1e-2


def _inputs(seed_offset=0):
    rank = torch.distributed.get_rank()
    gen = torch.Generator(device="cuda").manual_seed(1000 + 17 * seed_offset + rank)
    x = torch.randn(SEQ, MBS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    grad = torch.randn(SEQ, MBS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    return x, grad


def _run(layer, x, grad_out):
    inp = x.clone().requires_grad_(True)
    out, _ = layer(inp)
    out.backward(grad_out)
    return out.detach(), inp.grad.detach()


def _moonep_warnings(caplog):
    return [
        r for r in caplog.records if r.levelno >= logging.WARNING and "MoonEP" in r.getMessage()
    ]


def _check_expert_grads(checks, moon, base, tol=TOL):
    for i, (got, want) in enumerate(zip(moonep_expert_grads(moon), sequential_expert_grads(base))):
        for name, g, w in zip(("gate", "up", "down"), got, want):
            if w.norm() == 0:
                checks.check(torch.count_nonzero(g) == 0, f"expert {i} {name}: expected zero grad")
            else:
                checks.close(f"expert {i} {name} grad", g, w, tol)


def _skew_routing(*layers):
    """Route every token to the experts of the upper half of the EP ranks, so MoonEP copies
    them to the idle ranks (prefetch) and reduces their grads back (reduce_grad)."""
    hot = torch.arange(NUM_EXPERTS, device="cuda") >= NUM_EXPERTS // 2
    direction = torch.ones(HIDDEN, device="cuda") / HIDDEN**0.5
    with torch.no_grad():
        for layer in layers:
            w = layer.router.weight
            w.mul_(0.01)
            w.add_(torch.where(hot, 1.0, -1.0)[:, None] * direction.to(w.dtype))
    return direction


def _layer_grads(layer):
    """Every gradient of a MoonEP layer by parameter name (router and expert weights)."""
    return {name: param.grad.detach().clone() for name, param in layer.named_parameters()}


class TestMoonEPLayer:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(123)
        self.ep = Utils.world_size

    def teardown_method(self, method):
        InferenceMode.unset_active()
        teardown_moonep()

    def _pair(self, **overrides):
        base = build_moe_layer(alltoall_config(self.ep, **overrides))
        moon = build_moe_layer(moonep_config(self.ep, MAX_TOKENS, **overrides))
        copy_sequential_to_moonep(base, moon)
        return base, moon

    def _moonep_layers(self, count, seed=7):
        """``count`` MoonEP layers with distinct weights (they share one pool set)."""
        layers = []
        for i in range(count):
            _set_random_seed(seed + i)
            layers.append(build_moe_layer(moonep_config(self.ep, MAX_TOKENS), layer_number=i + 1))
        return layers

    # ---- Buffer and pools ---------------------------------------------------------------------

    def test_buffer_cache(self):
        from megatron.core.transformer.moe.moonep_backend.buffer import (
            destroy_moonep_buffer,
            get_moonep_buffer,
            moonep_buffer_is_live,
            release_moonep_buffer,
        )

        group = torch.distributed.new_group(list(range(self.ep)))
        kwargs = dict(
            max_tokens_per_rank=MAX_TOKENS,
            hidden_size=HIDDEN,
            topk=TOPK,
            num_experts=NUM_EXPERTS,
            num_sms=None,
            token_padding=128,
        )
        checks = RankChecks()
        buffer = get_moonep_buffer(group, **kwargs)
        checks.check(get_moonep_buffer(group, **kwargs) is buffer, "same key, new Buffer")
        with pytest.raises(RuntimeError, match="must share one moonep.Buffer"):
            get_moonep_buffer(group, **{**kwargs, "topk": TOPK + 1})
        destroy_moonep_buffer()
        checks.check(not moonep_buffer_is_live() and buffer.destroyed, "destroy left it live")
        again = get_moonep_buffer(group, **kwargs)
        checks.check(again is not buffer and moonep_buffer_is_live(), "re-create failed")
        release_moonep_buffer()  # non-collective; the caller keeps ownership of `again`
        checks.check(not moonep_buffer_is_live(), "release left it live")
        again.destroy()
        checks.assert_all()

    def test_layers_share_one_pool_set(self):
        layers = self._moonep_layers(3)
        pools = layers[0].experts.pools
        checks = RankChecks()
        checks.check(isinstance(pools, MoonEPExpertPools), "pools type")
        checks.check(all(layer.experts.pools is pools for layer in layers), "pools not shared")
        epn, rank = pools.epn, layers[0].experts.ep_group.rank()
        shapes = {"gate": (FFN, HIDDEN), "up": (FFN, HIDDEN), "down": (HIDDEN, FFN)}
        for name, inner in shapes.items():
            weights, grads = pools.weights(name), pools.grads(name)
            checks.check(weights.shape == (2 * epn, *inner), f"{name}: compute view shape")
            checks.check(weights.dtype == torch.bfloat16, f"{name}: compute view dtype")
            checks.check(grads.shape == (2 * epn, *inner), f"{name}: grad view shape")
            checks.check(grads.dtype == torch.float32, f"{name}: grad view dtype")
            prefetch = pools.pools.prefetch_buffer(name)
            checks.check(prefetch.shape == (self.ep, epn, *inner), f"{name}: prefetch pool")
            # This rank's slots are the second half of its compute view and chunk `rank` of the
            # all-rank pool: write through the former, read every chunk through the latter.
            weights[epn:].fill_(rank + 1)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            for r in range(self.ep):
                ok = bool((prefetch[r] == r + 1).all())
                checks.check(ok, f"{name}: prefetch pool chunk {r} is not rank {r}'s slots")
            torch.cuda.synchronize()
            torch.distributed.barrier()
        for layer in layers:
            experts = layer.experts
            for name, param in experts.named_parameters():
                checks.check(param.dtype == torch.bfloat16, f"{name}: dtype {param.dtype}")
                checks.check(param.requires_grad, f"{name}: not trainable")
                checks.check(param.allreduce is False, f"{name}: not marked as expert param")
            names = sorted(n for n, _ in experts.named_parameters())
            want = sorted(
                f"linear_fc{k}.weight{i}" for k in (1, 2) for i in range(experts.num_local_experts)
            )
            checks.check(names == want, f"parameter names {names}")
            checks.check(list(iter_moonep_experts(layer)) == [experts], "iter_moonep_experts")
            checks.check(uses_moonep(layer.config), "uses_moonep")
        checks.assert_all()

    def test_float16_module_casts_weights_like_any_linear(self):
        config = moonep_config(self.ep, MAX_TOKENS, params_dtype=torch.float32)
        layer = build_moe_layer(config)
        wrapped = Float16Module(config, layer)
        checks = RankChecks()
        for name, param in layer.experts.named_parameters():
            checks.check(param.dtype == torch.bfloat16, f"{name} not converted: {param.dtype}")
        x, grad = _inputs()
        out, _ = wrapped(x.clone().requires_grad_(True))
        out.float().backward(grad.float())
        for name, param in layer.experts.named_parameters():
            ok = param.grad is not None and param.grad.dtype == torch.bfloat16
            checks.check(ok, f"{name}: no bf16 grad")
        checks.assert_all()

    # ---- MoE layer ---------------------------------------------------------------------------

    def test_parity_vs_alltoall(self, caplog):
        base, moon = self._pair()
        x, grad = _inputs()
        caplog.set_level(logging.WARNING)
        out_b, dx_b = _run(base, x, grad)
        with moonep_calls(moon.token_dispatcher._comm_manager.buffer) as calls:
            out_m, dx_m = _run(moon, x, grad)
        checks = RankChecks()
        checks.close("output", out_m, out_b, TOL)
        checks.close("input grad", dx_m, dx_b, TOL)
        checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
        _check_expert_grads(checks, moon, base)
        # One layer: its backward finds its own copies in the pools and pushes nothing.
        expected = {"fresh": 1, "replay": 0, "combine_backward": 1, "prefetch": 1, "reduce": 1}
        checks.check(calls == expected, f"MoonEP calls {calls}, expected {expected}")
        checks.check(_moonep_warnings(caplog) == [], f"warnings: {_moonep_warnings(caplog)}")
        checks.check(not moon.experts.moonep_state.fwd_outstanding, "forward still outstanding")
        checks.check(moon.experts.pools.grads_owner is None, "gradients left in the pools")
        checks.assert_all()

    def test_parity_below_capacity(self):
        # A Buffer sized for twice this step's tokens (s < S, e.g. a smaller eval micro-batch):
        # dispatch returns [plan.nvs_s, H] views shorter than the capacity NvS.
        base = build_moe_layer(alltoall_config(self.ep))
        moon = build_moe_layer(moonep_config(self.ep, 2 * MAX_TOKENS))
        copy_sequential_to_moonep(base, moon)
        x, grad = _inputs(seed_offset=2)
        out_b, dx_b = _run(base, x, grad)
        out_m, dx_m = _run(moon, x, grad)
        checks = RankChecks()
        manager = moon.token_dispatcher._comm_manager
        plan, nvs = manager.last_plan, int(manager.buffer._require_ctx()["NvS"])
        checks.check(plan.num_tokens == MAX_TOKENS, f"plan.num_tokens={plan.num_tokens}")
        checks.check(plan.nvs_s < nvs, f"plan.nvs_s={plan.nvs_s} is not below NvS={nvs}")
        checks.close("output", out_m, out_b, TOL)
        checks.close("input grad", dx_m, dx_b, TOL)
        checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
        _check_expert_grads(checks, moon, base)
        checks.assert_all()

    def test_skewed_routing_uses_prefetch(self):
        base, moon = self._pair()
        direction = _skew_routing(base, moon)
        x, grad = _inputs(seed_offset=1)
        x = x + 4.0 * direction.to(x.dtype)
        out_b, dx_b = _run(base, x, grad)
        out_m, dx_m = _run(moon, x, grad)
        checks = RankChecks()
        # Every token picks hot experts, owned by the upper half of the EP ranks: MoonEP copies
        # them to the idle ranks and reduces their grads back to the owners.
        plan = moon.token_dispatcher._comm_manager.last_plan
        checks.check(bool((plan.experts_to_copy >= 0).any()), "no expert was copied")
        # Each used slot of this rank holds the bf16 weights its owner pushed for this step.
        experts, pools = moon.experts, moon.experts.pools
        copies = plan.experts_to_copy[experts.ep_group.rank()].tolist()
        fc1, fc2 = experts.expert_weights()
        own = {
            "gate": torch.stack([w[:FFN] for w in fc1]),
            "up": torch.stack([w[FFN:] for w in fc1]),
            "down": torch.stack(list(fc2)),
        }
        for name, local in own.items():
            gathered = [torch.empty_like(local) for _ in range(self.ep)]
            torch.distributed.all_gather(gathered, local.detach(), group=experts.ep_group)
            every_expert = torch.cat(gathered)
            for slot, expert in enumerate(copies):
                if expert >= 0:
                    checks.equal(
                        f"{name} slot {slot} (expert {expert})",
                        pools.weights(name)[pools.epn + slot],
                        every_expert[expert],
                    )
        checks.close("output", out_m, out_b, TOL)
        checks.close("input grad", dx_m, dx_b, TOL)
        checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
        _check_expert_grads(checks, moon, base)
        checks.assert_all()

    def test_accumulation_over_micro_batches(self, caplog):
        base, moon = self._pair()
        caplog.set_level(logging.WARNING)
        for micro_batch in range(3):
            x, grad = _inputs(seed_offset=10 + micro_batch)
            _run(base, x, grad)
            _run(moon, x, grad)
        checks = RankChecks()
        _check_expert_grads(checks, moon, base)
        checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
        checks.check(_moonep_warnings(caplog) == [], f"warnings: {_moonep_warnings(caplog)}")
        checks.assert_all()

    @pytest.mark.parametrize("skewed", [False, True], ids=["balanced", "skewed"])
    def test_chain_matches_layers_run_alone(self, skewed):
        """Three layers share the pools: each one's backward must restage its own weights (the
        last forward's layer still finds its copies). The chain equals each layer run alone on
        the chain's input to it, bit for bit, and its gradients reach the right parameters."""
        chain = self._moonep_layers(3)
        alone = self._moonep_layers(3, seed=100)
        for src, dst in zip(chain, alone):
            copy_moonep_layer(src, dst)
        x, grad = _inputs(seed_offset=3)
        if skewed:
            direction = _skew_routing(*chain, *alone)
            x = x + 4.0 * direction.to(x.dtype)
        inputs, outputs = [], []
        hidden = x.clone().requires_grad_(True)
        with moonep_calls(chain[0].token_dispatcher._comm_manager.buffer) as calls:
            for layer in chain:
                hidden.retain_grad()
                inputs.append(hidden)
                hidden, _ = layer(hidden)
                outputs.append(hidden)
            hidden.backward(grad)
        checks = RankChecks()
        # Three forward pushes, then one restage each for layers 2 and 1.
        expected = {"fresh": 3, "replay": 0, "combine_backward": 3, "prefetch": 5, "reduce": 3}
        checks.check(calls == expected, f"MoonEP calls {calls}, expected {expected}")
        out_grads = [inp.grad for inp in inputs[1:]] + [grad]
        for i, layer in enumerate(alone):
            inp = inputs[i].detach().clone().requires_grad_(True)
            out, _ = layer(inp)
            out.backward(out_grads[i])
            checks.equal(f"layer {i + 1} output", out, outputs[i].detach())
            checks.equal(f"layer {i + 1} input grad", inp.grad, inputs[i].grad)
            got, want = _layer_grads(chain[i]), _layer_grads(layer)
            for name in want:
                checks.equal(f"layer {i + 1} {name} grad", got[name], want[name])
        checks.assert_all()

    def test_interleaved_layers_match_sequential_order(self):
        """F_A F_B B_B B_A equals F_A B_A F_B B_B: the backward of A restages A after B used the
        pools, and the gradients of B never leak into A's."""
        a1, b1, a2, b2 = self._moonep_layers(4)
        copy_moonep_layer(a1, a2)
        copy_moonep_layer(b1, b2)
        xa, ga = _inputs(seed_offset=4)
        xb, gb = _inputs(seed_offset=5)
        ia, ib = xa.clone().requires_grad_(True), xb.clone().requires_grad_(True)
        out_a, _ = a1(ia)
        out_b, _ = b1(ib)
        out_b.backward(gb)
        out_a.backward(ga)
        out_a2, dx_a2 = _run(a2, xa, ga)
        out_b2, dx_b2 = _run(b2, xb, gb)
        checks = RankChecks()
        checks.equal("A output", out_a.detach(), out_a2)
        checks.equal("B output", out_b.detach(), out_b2)
        checks.equal("A input grad", ia.grad, dx_a2)
        checks.equal("B input grad", ib.grad, dx_b2)
        for got_layer, want_layer, tag in ((a1, a2, "A"), (b1, b2, "B")):
            got, want = _layer_grads(got_layer), _layer_grads(want_layer)
            for name in want:
                checks.equal(f"{tag} {name} grad", got[name], want[name])
        checks.assert_all()

    def test_no_grad_forward_between_forward_and_backward(self):
        """An eval forward of another layer, or of the same layer, between a forward and its
        backward overwrites the pools; the backward restages and gives the same results."""
        layer, other, ref = self._moonep_layers(3)
        copy_moonep_layer(layer, ref)
        x, grad = _inputs(seed_offset=6)
        probe, _ = _inputs(seed_offset=7)
        inp = x.clone().requires_grad_(True)
        out, _ = layer(inp)
        with torch.no_grad():
            other(probe)
            layer.eval()
            layer(probe)
            layer.train()
        out.backward(grad)
        out_ref, dx_ref = _run(ref, x, grad)
        checks = RankChecks()
        checks.equal("output", out.detach(), out_ref)
        checks.equal("input grad", inp.grad, dx_ref)
        got, want = _layer_grads(layer), _layer_grads(ref)
        for name in want:
            checks.equal(f"{name} grad", got[name], want[name])
        checks.assert_all()

    def test_weight_update_reaches_the_next_forward(self):
        """In-place weight updates (the optimizer step) are staged by the next forward."""
        base, moon = self._pair()
        x, grad = _inputs(seed_offset=8)
        _run(base, x, grad)
        _run(moon, x, grad)
        with torch.no_grad():
            for expert in base.experts.local_experts:
                expert.linear_fc1.weight.mul_(1.5)
                expert.linear_fc2.weight.add_(0.01)
            fc1, fc2 = moon.experts.expert_weights()
            for w1, w2 in zip(fc1, fc2):
                w1.mul_(1.5)
                w2.add_(0.01)
        out_b, _ = _run(base, x, grad)
        out_m, _ = _run(moon, x, grad)
        checks = RankChecks()
        checks.close("output after the update", out_m, out_b, TOL)
        checks.assert_all()

    @pytest.mark.parametrize("overwrite", [False, True], ids=["accumulate", "overwrite"])
    def test_main_grad_like_ddp_and_fsdp(self, overwrite):
        """Under a wrapper (params carry grad_added_to_main_grad) the fp32 sums go into
        main_grad, accumulated like DDP / ZeRO-1 or overwritten like ZeRO-2, and autograd only
        gets placeholders. Two micro-batches against the same layer without a wrapper."""
        layers = self._moonep_layers(2)
        wrapped, plain = layers
        copy_moonep_layer(wrapped, plain)
        for param in wrapped.experts.parameters():
            param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device="cuda")
            param.grad_added_to_main_grad = False
            param.overwrite_main_grad = overwrite
        checks = RankChecks()
        per_batch = []
        for micro_batch in range(2):
            x, grad = _inputs(seed_offset=20 + micro_batch)
            out_w, dx_w = _run(wrapped, x, grad)
            for param in plain.experts.parameters():
                param.grad = None
            out_p, dx_p = _run(plain, x, grad)
            checks.equal(f"micro-batch {micro_batch} output", out_w, out_p)
            checks.equal(f"micro-batch {micro_batch} input grad", dx_w, dx_p)
            per_batch.append({n: p.grad.float() for n, p in plain.experts.named_parameters()})
        for name, param in wrapped.experts.named_parameters():
            checks.check(param.grad_added_to_main_grad, f"{name}: grad_added_to_main_grad unset")
            want = per_batch[-1][name] if overwrite else per_batch[0][name] + per_batch[1][name]
            # The plain path rounds each micro-batch's fp32 sum to bf16 once.
            checks.close(f"{name} main_grad", param.main_grad, want, 1e-2)
        checks.assert_all()

    def test_frozen_router_and_input_still_train_experts(self):
        base, moon = self._pair()
        for layer in (base, moon):
            layer.router.weight.requires_grad_(False)
        x, grad = _inputs(seed_offset=9)
        out_b = base(x)[0]
        out_b.backward(grad)
        out_m = moon(x)[0]
        out_m.backward(grad)
        checks = RankChecks()
        checks.close("output", out_m.detach(), out_b.detach(), TOL)
        _check_expert_grads(checks, moon, base)
        checks.check(not moon.experts.moonep_state.fwd_outstanding, "forward still outstanding")
        checks.assert_all()

    def test_frozen_experts_get_no_gradient(self):
        base, moon = self._pair()
        for layer in (base, moon):
            layer.experts.requires_grad_(False)
        x, grad = _inputs(seed_offset=11)
        with moonep_calls(moon.token_dispatcher._comm_manager.buffer) as calls:
            _, dx_m = _run(moon, x, grad)
        _, dx_b = _run(base, x, grad)
        checks = RankChecks()
        checks.close("input grad", dx_m, dx_b, TOL)
        checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
        for name, param in moon.experts.named_parameters():
            checks.check(param.grad is None, f"{name} got a gradient")
        checks.check(calls["reduce"] == 0, f"reduce_grad ran for frozen experts: {calls}")
        moon.experts.linear_fc2.weight0.requires_grad_(True)
        with pytest.raises(RuntimeError, match="trainable and frozen expert weights"):
            moon.experts.weights_require_grad()
        moon.experts.linear_fc2.weight0.requires_grad_(False)
        checks.assert_all()

    def test_second_forward_before_backward_raises(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        layer.set_layer_number(3)
        x, grad = _inputs()
        inp = x.clone().requires_grad_(True)
        out, _ = layer(inp)
        with pytest.raises(RuntimeError, match="MoonEP: layer 3 started a second gradient"):
            layer(x.clone().requires_grad_(True))
        out.backward(grad)
        checks = RankChecks()
        state = layer.experts.moonep_state
        checks.check(not state.fwd_outstanding, "forward still outstanding")
        checks.check(layer.experts.layer_number == 3 and state.layer_number == 3, "layer number")
        checks.assert_all()

    def test_experts_trainable_on_some_ranks_raise_everywhere(self):
        """Whether a layer reduces dW drives a collective, so experts trainable on some EP ranks
        and frozen on others are rejected on every rank at the layer's first dispatch with grad
        enabled, before any MoonEP collective."""
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        if torch.distributed.get_rank() == 0:
            layer.experts.requires_grad_(False)
        x, _ = _inputs()
        with pytest.raises(RuntimeError, match="all trainable or all frozen on every EP rank"):
            layer(x.clone().requires_grad_(True))
        assert not layer.experts.moonep_state.fwd_outstanding

    def test_finalize_forgets_a_forward_without_backward(self):
        """A gradient-tracking forward whose backward never ran (its graph dropped) would trip
        the phase-1 guard forever; finalize_moonep_step, which runs after every backward of a
        step, forgets it."""
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        x, grad = _inputs()
        layer(x.clone().requires_grad_(True))
        checks = RankChecks()
        state = layer.experts.moonep_state
        checks.check(state.fwd_outstanding, "the forward is not outstanding")
        finalize_moonep_step([layer])
        checks.check(not state.fwd_outstanding, "finalize kept the forward outstanding")
        out, _ = layer(x.clone().requires_grad_(True))
        out.backward(grad)
        checks.check(not state.fwd_outstanding, "forward still outstanding")
        checks.assert_all()

    def test_failed_push_forgets_the_occupant(self, monkeypatch):
        """stage() overwrites the staging rows before it pushes, so a push that raises must not
        leave the pools claiming to hold the previous (layer, plan): its re-stage would be
        skipped and the GEMMs would read whatever the failed stage copied."""
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        x, grad = _inputs()
        out, _ = layer(x.clone().requires_grad_(True))
        out.backward(grad)
        pools = layer.experts.pools
        checks = RankChecks()
        checks.check(pools.occupant is not None, "no occupant after a forward and backward")

        def fail(plan):
            raise RuntimeError("injected push failure")

        monkeypatch.setattr(pools.pools, "push", fail)
        with pytest.raises(RuntimeError, match="injected push failure"):
            layer(x.clone().requires_grad_(True))
        checks.check(pools.occupant is None, "the occupant survived a failed push")
        monkeypatch.undo()
        finalize_moonep_step([layer])
        out, _ = layer(x.clone().requires_grad_(True))
        out.backward(grad)
        checks.check(pools.occupant is not None, "no occupant after the recovery step")
        checks.assert_all()

    def test_fsdp_params_without_unit_gather_raise(self):
        """Megatron-FSDP marks every parameter __fsdp_param__ when it wraps the model but sets
        overwrite_main_grad only through an FSDP unit's hooks, which never see MoonEP's expert
        weights outside a unit. The hand-off then cannot tell overwrite (ZeRO-2) from
        accumulate and raises on every rank, after the collective reduce."""
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        for param in layer.experts.parameters():
            buffer = torch.zeros(param.shape, dtype=torch.float32, device="cuda")
            param.get_main_grad = lambda buffer=buffer: buffer
            param.grad_added_to_main_grad = False
            param.__fsdp_param__ = True
        x, grad = _inputs()
        out, _ = layer(x.clone().requires_grad_(True))
        with pytest.raises(RuntimeError, match="never gathered the expert weights of MoE layer"):
            out.backward(grad)

    def test_capacity_and_inference_guards(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS // 2))
        x, _ = _inputs()
        with pytest.raises(RuntimeError, match="moe_moonep_max_tokens_per_rank=128"):
            layer(x)
        InferenceMode.set_active()
        with pytest.raises(RuntimeError, match="does not support Megatron inference engines"):
            layer(x[: SEQ // 4])
        InferenceMode.unset_active()
        assert not layer.experts.moonep_state.fwd_outstanding

    def test_no_grad_forward_leaves_nothing_outstanding(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        x, _ = _inputs()
        layer.eval()
        with torch.no_grad():
            layer(x)
        checks = RankChecks()
        checks.check(not layer.experts.moonep_state.fwd_outstanding, "forward outstanding")
        checks.check(layer.experts.pools.grads_owner is None, "gradients claimed the pools")
        checks.assert_all()

    def test_initial_weights_match_sequential(self):
        _set_random_seed(321)
        base = build_moe_layer(alltoall_config(self.ep))
        _set_random_seed(321)
        moon = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        fc1, fc2 = moon.experts.expert_weights()
        checks = RankChecks()
        for i, expert in enumerate(base.experts.local_experts):
            checks.equal(f"expert {i} fc1", fc1[i], expert.linear_fc1.weight)
            checks.equal(f"expert {i} fc2", fc2[i], expert.linear_fc2.weight)
        checks.assert_all()
