# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend: process-global Buffer, expert storage, grad-state machine and MoE layer.

Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>`` and without
``-x``: checks are collected and raised on every rank together (see ``RankChecks``).
"""

import logging

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moonep_backend import iter_moonep_experts, uses_moonep
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
    copy_sequential_to_moonep,
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

    # ---- Buffer (M3) --------------------------------------------------------------------------

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

    # ---- storage, mixed precision and the grad-state machine (M4) ---------------------------

    def test_storage_after_construction(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        experts = layer.experts
        storage = experts.storage
        checks = RankChecks()
        checks.check(experts.grad_state == "CLEAN" and not experts.finalized, "initial state")
        for name in ("gate", "up", "down"):
            own, slots = storage.fp32_targets(name)
            checks.check(torch.count_nonzero(own) == 0, f"{name}: owned grad rows not zeroed")
            checks.check(torch.count_nonzero(slots) == 0, f"{name}: reduce chunk not zeroed")
            master = getattr(experts, f"weight_{name}")
            checks.check(master.dtype == torch.float32, f"{name}: master dtype")
            checks.check(not master.requires_grad, f"{name}: master requires grad")
            checks.check(master.grad.data_ptr() == own.data_ptr(), f"{name}: .grad alias")
            checks.check(master.main_grad is master.grad, f"{name}: .main_grad alias")
            checks.equal(
                f"{name} own pool rows",
                storage.w[name][storage.lo : storage.hi],
                master.bfloat16(),
            )
            # Peers' masters are visible through the composite rows [0, E) over NVLink.
            gathered = [torch.empty_like(master) for _ in range(self.ep)]
            torch.distributed.all_gather(gathered, master.detach(), group=experts.ep_group)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            checks.equal(
                f"{name} peer pool rows",
                storage.w[name][: storage.E],
                torch.cat(gathered).bfloat16(),
            )
        checks.check(experts.check_invariants() == [], f"invariants: {experts.check_invariants()}")
        checks.check(list(iter_moonep_experts(layer)) == [experts], "iter_moonep_experts")
        checks.check(uses_moonep(layer.config), "uses_moonep")
        checks.assert_all()

    def test_mixed_precision_invariants(self):
        config = moonep_config(self.ep, MAX_TOKENS)
        layer = build_moe_layer(config)
        experts = layer.experts
        grad_ptrs = [p.grad.data_ptr() for p in experts.moonep_master_parameters()]
        pool_ptrs = [experts.storage.w[n].data_ptr() for n in ("gate", "up", "down")]
        wrapped = Float16Module(config, layer)
        wrapped.to(torch.float16)
        wrapped.cuda()
        checks = RankChecks()
        # Sibling modules are still converted.
        checks.check(layer.router.weight.dtype == torch.float16, "router not converted")
        for param, ptr in zip(experts.moonep_master_parameters(), grad_ptrs):
            checks.check(param.dtype == torch.float32, "master converted")
            checks.check(param.grad.dtype == torch.float32, ".grad converted")
            checks.check(param.grad.data_ptr() == ptr, ".grad alias broken")
        checks.check(
            [experts.storage.w[n].data_ptr() for n in ("gate", "up", "down")] == pool_ptrs,
            "pools replaced",
        )
        checks.check(experts.check_invariants() == [], f"invariants: {experts.check_invariants()}")
        with pytest.raises(RuntimeError, match="refusing to move"):
            experts.cpu()
        checks.check(wrapped.zero_moe_grads() == 1, "zero_moe_grads through Float16Module")
        checks.assert_all()

    def test_state_machine(self, caplog):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        experts = layer.experts
        caplog.set_level(logging.WARNING)
        experts._on_grad_forward_()
        assert experts.grad_state == "ACCUMULATING"
        experts.mark_grads_consumed_()
        assert experts.grad_state == "STALE"
        experts._on_grad_forward_()
        assert experts.grad_state == "ACCUMULATING"
        stale = [r for r in _moonep_warnings(caplog) if "were not zeroed" in r.getMessage()]
        assert len(stale) == 1

        experts.zero_moonep_grads_()
        experts.mark_zero_point_()
        assert experts.grad_state == "CLEAN"

        caplog.clear()
        experts._on_grad_forward_()
        experts.finalized = True
        experts.zero_moonep_grads_()
        assert any("discarded finalized" in r.getMessage() for r in _moonep_warnings(caplog))

        experts.config.moe_moonep_strict_zero_grad = True
        try:
            experts._on_grad_forward_()
            experts.mark_grads_consumed_()
            with pytest.raises(RuntimeError, match="were not zeroed"):
                experts._on_grad_forward_()
        finally:
            experts.config.moe_moonep_strict_zero_grad = False

        caplog.clear()
        experts.weight_gate.grad = None
        experts._on_grad_forward_()
        assert any("Re-binding" in r.getMessage() for r in _moonep_warnings(caplog))
        assert experts.check_invariants() == []

    # ---- MoE layer (M5) ---------------------------------------------------------------------

    def test_parity_vs_alltoall(self, caplog):
        base, moon = self._pair()
        x, grad = _inputs()
        caplog.set_level(logging.WARNING)
        out_b, dx_b = _run(base, x, grad)
        out_m, dx_m = _run(moon, x, grad)
        checks = RankChecks()
        checks.close("output", out_m, out_b, TOL)
        checks.close("input grad", dx_m, dx_b, TOL)
        checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
        _check_expert_grads(checks, moon, base)
        checks.check(_moonep_warnings(caplog) == [], f"warnings: {_moonep_warnings(caplog)}")
        checks.check(not moon.experts.storage.fwd_outstanding, "forward still outstanding")
        checks.check(moon.experts.check_invariants() == [], "invariants")
        checks.assert_all()

    def test_skewed_routing_uses_prefetch(self):
        base, moon = self._pair()
        hot = torch.arange(NUM_EXPERTS, device="cuda") >= NUM_EXPERTS // 2
        direction = torch.ones(HIDDEN, device="cuda") / HIDDEN**0.5
        with torch.no_grad():
            for layer in (base, moon):
                w = layer.router.weight
                w.mul_(0.01)
                w.add_(torch.where(hot, 1.0, -1.0)[:, None] * direction.to(w.dtype))
        x, grad = _inputs(seed_offset=1)
        x = x + 4.0 * direction.to(x.dtype)
        out_b, dx_b = _run(base, x, grad)
        out_m, dx_m = _run(moon, x, grad)
        checks = RankChecks()
        # Every token picks hot experts, owned by the upper half of the EP ranks: MoonEP copies
        # them to the idle ranks and reduces their grads back to the owners.
        plan = moon.token_dispatcher._comm_manager.last_plan
        checks.check(bool((plan.experts_to_copy >= 0).any()), "no expert was copied")
        checks.close("output", out_m, out_b, TOL)
        checks.close("input grad", dx_m, dx_b, TOL)
        checks.close("router grad", moon.router.weight.grad, base.router.weight.grad, TOL)
        _check_expert_grads(checks, moon, base)
        checks.assert_all()

    def test_accumulation_without_zeroing(self, caplog):
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

    def test_second_forward_before_backward_raises(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        layer.set_layer_number(3)
        x, grad = _inputs()
        inp = x.clone().requires_grad_(True)
        out, _ = layer(inp)
        with pytest.raises(RuntimeError, match="MoonEP: layer 3 started a second gradient"):
            layer(x.clone().requires_grad_(True))
        with pytest.raises(RuntimeError, match="has a forward without its backward"):
            layer.zero_moe_grads()
        out.backward(grad)
        checks = RankChecks()
        checks.check(not layer.experts.storage.fwd_outstanding, "forward still outstanding")
        checks.check(layer.zero_moe_grads() == 1, "zero_moe_grads")
        checks.assert_all()

    def test_layer_number_in_messages(self, caplog):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        layer.set_layer_number(3)
        caplog.set_level(logging.WARNING)
        layer.experts.mark_grads_consumed_()
        x, grad = _inputs()
        _run(layer, x, grad)
        checks = RankChecks()
        checks.check(layer.experts.layer_number == 3, "experts layer number")
        checks.check(layer.experts.storage.layer_number == 3, "storage layer number")
        checks.check(
            any("MoE layer 3 " in r.getMessage() for r in _moonep_warnings(caplog)),
            "stale notice without the layer number",
        )
        checks.assert_all()

    def test_capacity_and_inference_guards(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS // 2))
        x, _ = _inputs()
        with pytest.raises(RuntimeError, match="moe_moonep_max_tokens_per_rank=128"):
            layer(x)
        InferenceMode.set_active()
        with pytest.raises(RuntimeError, match="does not support Megatron inference engines"):
            layer(x[: SEQ // 4])
        InferenceMode.unset_active()
        assert not layer.experts.storage.fwd_outstanding

    def test_no_grad_forward_keeps_state(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        x, _ = _inputs()
        layer.eval()
        with torch.no_grad():
            layer(x)
        checks = RankChecks()
        checks.check(layer.experts.grad_state == "CLEAN", "state changed")
        checks.check(not layer.experts.storage.fwd_outstanding, "forward marked outstanding")
        checks.assert_all()

    def test_training_forward_without_grad_inputs_raises(self):
        layer = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        layer.router.weight.requires_grad_(False)
        x, _ = _inputs()
        with pytest.raises(RuntimeError, match="neither its input nor its router probabilities"):
            layer(x)
        assert not layer.experts.storage.fwd_outstanding

    def test_initial_weights_match_sequential(self):
        _set_random_seed(321)
        base = build_moe_layer(alltoall_config(self.ep))
        _set_random_seed(321)
        moon = build_moe_layer(moonep_config(self.ep, MAX_TOKENS))
        experts = moon.experts
        checks = RankChecks()
        for i, expert in enumerate(base.experts.local_experts):
            fc1 = expert.linear_fc1.weight.float()
            checks.equal(f"expert {i} gate", experts.weight_gate[i], fc1[:FFN])
            checks.equal(f"expert {i} up", experts.weight_up[i], fc1[FFN:])
            checks.equal(
                f"expert {i} down", experts.weight_down[i], expert.linear_fc2.weight.float()
            )
        checks.assert_all()
