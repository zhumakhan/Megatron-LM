# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend in a training loop: grad finalization, optimizer, zeroing notices, parity.

Every test builds a GPTModel (all layers MoE) under Float16Module + DDP twice, once on the
alltoall dispatcher with SequentialMLP experts and once on MoonEP, with identical weights, and
drives them through ``forward_backward_no_pipelining`` with ``finalize_model_grads``.
Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>``, without ``-x``.
"""

import logging

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.moonep_optimizer import MoonEPOptimizer
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.moonep_backend import iter_moonep_experts
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import (
    NUM_EXPERTS,
    RankChecks,
    copy_sequential_to_moonep,
    moe_config_kwargs,
    requires_moonep,
    teardown_moonep,
)

pytestmark = [pytest.mark.internal, requires_moonep]

SEQ, MBS, VOCAB, NUM_LAYERS = 128, 2, 1024, 2
STALE_TEXT = "were not zeroed since the last optimizer step"


def gpt_config(ep, backend, num_layers=NUM_LAYERS, **overrides):
    kwargs = moe_config_kwargs(
        ep,
        num_layers=num_layers,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        moe_aux_loss_coeff=1e-2,
        moe_router_load_balancing_type="aux_loss",
        **overrides,
    )
    if backend == "moonep":
        kwargs.update(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="moonep",
            moe_moonep_max_tokens_per_rank=SEQ * MBS,
        )
    else:
        kwargs.update(moe_token_dispatcher_type="alltoall")
    return TransformerConfig(**kwargs)


def build_model(config, seed, *, distopt, average_in_collective=False):
    """GPTModel -> Float16Module -> DDP, the order megatron/training uses."""
    _set_random_seed(seed)
    gpt = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(num_experts=NUM_EXPERTS),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
    )
    gpt.cuda()
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=distopt,
        average_in_collective=average_in_collective,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
    )
    model = DDP(config, ddp_config, Float16Module(config, gpt))
    config.finalize_model_grads_func = finalize_model_grads
    return model, gpt


@torch.no_grad()
def copy_gpt_weights(base_gpt, moon_gpt):
    moon_params = dict(moon_gpt.named_parameters())
    for name, param in base_gpt.named_parameters():
        if ".mlp.experts." not in name:
            moon_params[name].copy_(param)
    for base_layer, moon_layer in zip(base_gpt.decoder.layers, moon_gpt.decoder.layers):
        copy_sequential_to_moonep(base_layer.mlp, moon_layer.mlp)


def build_pair(ep, *, distopt=True, average_in_collective=False, **config_overrides):
    """(base_model, base_gpt, moon_model, moon_gpt) with identical weights, before optimizers.

    ``num_layers`` (default 2) goes to the config. With more than one layer the second layer's
    input already differs by bf16 rounding (combine accumulates in fp32), which flips near-tie
    top-k choices; tests that compare individual gradients use one layer.
    """
    base, base_gpt = build_model(
        gpt_config(ep, "alltoall", **config_overrides),
        seed=1234,
        distopt=distopt,
        average_in_collective=average_in_collective,
    )
    moon, moon_gpt = build_model(
        gpt_config(ep, "moonep", **config_overrides),
        seed=1234,
        distopt=distopt,
        average_in_collective=average_in_collective,
    )
    copy_gpt_weights(base_gpt, moon_gpt)
    return base, base_gpt, moon, moon_gpt


def build_optimizer(model, *, distopt=True, **overrides):
    kwargs = dict(
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=distopt,
    )
    kwargs.update(overrides)
    optimizer = get_megatron_optimizer(OptimizerConfig(**kwargs), [model])
    model.config.grad_scale_func = optimizer.scale_loss
    return optimizer


def make_batches(num_microbatches, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + dist.get_rank())
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(MBS, -1)
    causal = torch.triu(torch.ones(SEQ, SEQ, device="cuda", dtype=torch.bool), diagonal=1)
    batches = []
    for _ in range(num_microbatches):
        tokens = torch.randint(0, VOCAB, (MBS, SEQ), device="cuda", generator=gen)
        labels = torch.roll(tokens, shifts=-1, dims=1)
        loss_mask = torch.ones(MBS, SEQ, device="cuda")
        batches.append((tokens, labels, position_ids, causal[None, None], loss_mask))
    return batches


def forward_step(data_iterator, model):
    tokens, labels, position_ids, attention_mask, loss_mask = next(data_iterator)
    output = model(tokens, position_ids, attention_mask, labels=labels)

    def loss_func(output_tensor):
        losses = output_tensor.float().view(-1)
        mask = loss_mask.view(-1)
        loss = torch.sum(losses * mask)
        num_tokens = mask.sum().to(torch.int)
        reported = torch.cat([loss.detach().view(1), num_tokens.detach().float().view(1)])
        return loss, num_tokens, {"lm loss": reported}

    return output, loss_func


def forward_backward(model, batches, *, forward_only=False):
    """Returns the global token-mean loss of the batches."""
    outputs = forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=iter(batches),
        model=model,
        num_microbatches=len(batches),
        seq_length=SEQ,
        micro_batch_size=MBS,
        forward_only=forward_only,
    )
    totals = torch.stack([out["lm loss"] for out in outputs]).sum(dim=0)
    dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()


def train_step(model, optimizer, batches, *, zero_moe_grads=True):
    """One iteration in megatron/training/training.py:train_step order."""
    model.zero_grad_buffer()
    optimizer.zero_grad()
    if zero_moe_grads:
        model.zero_moe_grads()
    loss = forward_backward(model, batches)
    success, grad_norm, _ = optimizer.step()
    return loss, float(grad_norm), success


def moonep_warnings(caplog, text=""):
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "MoonEP" in r.getMessage() and text in r.getMessage()
    ]


def moonep_experts(gpt):
    return list(iter_moonep_experts(gpt))


class TestMoonEPTraining:
    def setup_method(self, method):
        self.ep = Utils.world_size
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=self.ep)

    def teardown_method(self, method):
        # The global MoE metrics tracker sizes its per-layer entries from the first model it sees.
        destroy_moe_metrics_tracker()
        teardown_moonep()

    def _reinit(self, ep):
        Utils.destroy_model_parallel()
        self.ep = ep
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)

    # ---- M6: gradient finalization ----------------------------------------------------------

    @pytest.mark.parametrize("ep_divisor", [1, 2])
    @pytest.mark.parametrize("average_in_collective", [False, True])
    @pytest.mark.parametrize("per_token", [False, True])
    def test_finalize_matches_ddp(self, ep_divisor, average_in_collective, per_token):
        if per_token and average_in_collective:
            pytest.skip("DDP rejects per-token loss with average_in_collective")
        if Utils.world_size // ep_divisor < 2:
            pytest.skip("needs EP >= 2")
        self._reinit(Utils.world_size // ep_divisor)
        base, base_gpt, moon, moon_gpt = build_pair(
            self.ep,
            distopt=False,
            average_in_collective=average_in_collective,
            calculate_per_token_loss=per_token,
            num_layers=1,
        )
        batches = make_batches(2, seed=1)
        for model in (base, moon):
            model.zero_grad_buffer()
            model.zero_moe_grads()
            forward_backward(model, batches)
        checks = RankChecks()
        ffn = moon_gpt.config.moe_ffn_hidden_size
        for layer_idx, (base_layer, moon_layer) in enumerate(
            zip(base_gpt.decoder.layers, moon_gpt.decoder.layers)
        ):
            experts = moon_layer.mlp.experts
            checks.check(experts.finalized, f"layer {layer_idx} not finalized")
            for i, expert in enumerate(base_layer.mlp.experts.local_experts):
                fc1 = expert.linear_fc1.weight.main_grad
                fc2 = expert.linear_fc2.weight.main_grad
                tag = f"layer {layer_idx} expert {i}"
                checks.close(f"{tag} gate", experts.weight_gate.grad[i], fc1[:ffn], 1e-2)
                checks.close(f"{tag} up", experts.weight_up.grad[i], fc1[ffn:], 1e-2)
                checks.close(f"{tag} down", experts.weight_down.grad[i], fc2, 1e-2)
            checks.close(
                f"layer {layer_idx} router",
                moon_layer.mlp.router.weight.main_grad,
                base_layer.mlp.router.weight.main_grad,
                1e-2,
            )
        checks.assert_all()

    def test_finalize_inf_is_fatal(self):
        _, _, moon, moon_gpt = build_pair(self.ep, distopt=False)
        moon.zero_grad_buffer()
        moon.zero_moe_grads()
        experts = moonep_experts(moon_gpt)[0]
        moon.config.finalize_model_grads_func = None
        forward_backward(moon, make_batches(1, seed=2))
        moon.config.finalize_model_grads_func = finalize_model_grads
        experts.weight_gate.grad[0, 0, 0] = float("inf")
        with pytest.raises(RuntimeError, match="found Inf in MoonEP expert grad 'weight_gate'"):
            finalize_model_grads([moon])

    # ---- M7: optimizer ----------------------------------------------------------------------

    def test_optimizer_layout_and_zero_grad_keeps_alias(self):
        _, _, moon, moon_gpt = build_pair(self.ep)
        optimizer = build_optimizer(moon)
        checks = RankChecks()
        last = optimizer.chained_optimizers[-1]
        checks.check(isinstance(last, MoonEPOptimizer), "MoonEPOptimizer is not last")
        masters = {id(p) for e in moonep_experts(moon_gpt) for p in e.moonep_master_parameters()}
        checks.check({id(p) for p in last.get_parameters()} == masters, "masters not owned")
        for other in optimizer.chained_optimizers[:-1]:
            checks.check(
                not masters & {id(p) for p in other.get_parameters()}, "masters leaked"
            )
        masters_in_order = [
            p for e in moonep_experts(moon_gpt) for p in e.moonep_master_parameters()
        ]
        ptrs = [p.grad.data_ptr() for p in masters_in_order]
        optimizer.zero_grad()  # default set_to_none=True
        after = [None if p.grad is None else p.grad.data_ptr() for p in masters_in_order]
        checks.check(after == ptrs, "zero_grad deleted or replaced MoonEP grads")
        checks.assert_all()

    def test_step_matches_baseline_and_syncs_pools(self, caplog):
        base, _, moon, moon_gpt = build_pair(self.ep)
        base_opt = build_optimizer(base)
        moon_opt = build_optimizer(moon)
        caplog.set_level(logging.WARNING)
        batches = make_batches(2, seed=3)
        base_loss, base_norm, _ = train_step(base, base_opt, batches)
        moon_loss, moon_norm, success = train_step(moon, moon_opt, batches)
        checks = RankChecks()
        checks.check(success, "MoonEP step failed")
        checks.check(
            abs(moon_loss - base_loss) / base_loss < 1e-2, f"loss {moon_loss} vs {base_loss}"
        )
        checks.check(
            abs(moon_norm - base_norm) / base_norm < 2e-2,
            f"grad norm {moon_norm} vs {base_norm}",
        )
        for experts in moonep_experts(moon_gpt):
            storage = experts.storage
            for name in ("gate", "up", "down"):
                master = getattr(experts, f"weight_{name}")
                own_rows = storage.w[name][storage.lo : storage.hi]
                checks.equal(f"{name} pool after step", own_rows, master.bfloat16())
            checks.check(experts.grad_state == "STALE", f"state {experts.grad_state}")
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()

    def test_module_zero_grad_is_repaired(self, caplog):
        _, _, moon, moon_gpt = build_pair(self.ep)
        optimizer = build_optimizer(moon)
        caplog.set_level(logging.WARNING)
        moon.zero_grad()  # torch.nn.Module.zero_grad(set_to_none=True) deletes every .grad
        _, _, success = train_step(moon, optimizer, make_batches(1, seed=4))
        checks = RankChecks()
        checks.check(success, "step failed")
        rebinds = moonep_warnings(caplog, "Re-binding")
        checks.check(len(rebinds) == 3 * NUM_LAYERS, f"{len(rebinds)} re-binding notices")
        for experts in moonep_experts(moon_gpt):
            checks.check(experts.check_invariants() == [], f"{experts.check_invariants()}")
        checks.assert_all()

    def test_step_without_finalize_raises(self):
        _, _, moon, _ = build_pair(self.ep)
        optimizer = build_optimizer(moon)
        moon.config.finalize_model_grads_func = None
        moon.zero_grad_buffer()
        optimizer.zero_grad()
        moon.zero_moe_grads()
        forward_backward(moon, make_batches(1, seed=5))
        with pytest.raises(RuntimeError, match="reached the optimizer without reduce_moonep"):
            optimizer.step()

    def test_mup_overrides_reach_masters(self):
        from megatron.core.optimizer import get_mup_config_overrides

        base, _, moon, _ = build_pair(self.ep)
        groups = {}
        for name, model in (("base", base), ("moon", moon)):
            config = OptimizerConfig(
                optimizer="adam",
                lr=1e-3,
                min_lr=1e-4,
                adam_eps=1e-8,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
            )
            overrides = get_mup_config_overrides(config, mup_width_mult=4.0)
            optimizer = get_megatron_optimizer(config, [model], config_overrides=overrides)
            groups[name] = [g for g in optimizer.param_groups if g.get("is_expert_parallel")]
        checks = RankChecks()
        base_group, moon_group = groups["base"][0], groups["moon"][-1]
        checks.check(moon_group.get("is_moonep", False), "last expert group is not MoonEP's")
        for key in ("max_lr", "min_lr", "eps"):
            checks.check(
                moon_group.get(key) == base_group.get(key),
                f"{key}: {moon_group.get(key)} vs {base_group.get(key)}",
            )
        checks.assert_all()

    # ---- M8: zero_moe_grads and notices -------------------------------------------------------

    def test_notify_every_time(self, caplog):
        _, _, moon, _ = build_pair(self.ep)
        optimizer = build_optimizer(moon)
        caplog.set_level(logging.WARNING)
        counts = []
        for iteration in range(3):
            caplog.clear()
            batches = make_batches(2, seed=6 + iteration)
            train_step(moon, optimizer, batches, zero_moe_grads=iteration == 0)
            counts.append(len(moonep_warnings(caplog, STALE_TEXT)))
        checks = RankChecks()
        expected = [0, NUM_LAYERS, NUM_LAYERS]
        checks.check(counts == expected, f"stale notices per iteration: {counts}")
        checks.assert_all()

    def test_no_false_positive_accumulation(self, caplog):
        _, _, moon, _ = build_pair(self.ep)
        optimizer = build_optimizer(moon)
        caplog.set_level(logging.WARNING)
        for iteration in range(2):
            train_step(moon, optimizer, make_batches(4, seed=9 + iteration))
        checks = RankChecks()
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()

    def test_strict_raises(self):
        _, _, moon, _ = build_pair(self.ep, moe_moonep_strict_zero_grad=True)
        optimizer = build_optimizer(moon)
        train_step(moon, optimizer, make_batches(1, seed=11))
        with pytest.raises(RuntimeError, match=STALE_TEXT):
            train_step(moon, optimizer, make_batches(1, seed=12), zero_moe_grads=False)

    def test_eval_no_transition(self):
        _, _, moon, moon_gpt = build_pair(self.ep)
        moon.eval()
        with torch.no_grad():
            forward_backward(moon, make_batches(1, seed=13), forward_only=True)
        checks = RankChecks()
        for experts in moonep_experts(moon_gpt):
            checks.check(experts.grad_state == "CLEAN", f"state {experts.grad_state}")
            checks.check(not experts.storage.fwd_outstanding, "forward outstanding")
        checks.assert_all()

    def test_eval_mode_with_grad_counts(self, caplog):
        _, _, moon, _ = build_pair(self.ep)
        optimizer = build_optimizer(moon)
        caplog.set_level(logging.WARNING)
        moon.zero_moe_grads()
        moon.eval()
        moon.config.finalize_model_grads_func = None
        forward_backward(moon, make_batches(1, seed=14))
        moon.config.finalize_model_grads_func = finalize_model_grads
        optimizer.zero_grad()
        moon.train()
        moon.zero_grad_buffer()
        forward_backward(moon, make_batches(1, seed=15))
        checks = RankChecks()
        stale = moonep_warnings(caplog, STALE_TEXT)
        checks.check(len(stale) == NUM_LAYERS, f"{len(stale)} stale notices")
        checks.assert_all()

    def test_notify_when_warnings_hidden(self, capfd):
        _, _, _, moon_gpt = build_pair(self.ep)
        experts = moonep_experts(moon_gpt)[0]
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.ERROR)
        try:
            experts._on_grad_forward_()
            experts.mark_grads_consumed_()
            experts._on_grad_forward_()
        finally:
            root.setLevel(old_level)
        assert STALE_TEXT in capfd.readouterr().err

    # ---- M10: model-level parity --------------------------------------------------------------

    def test_loss_parity(self, caplog):
        base, _, moon, _ = build_pair(self.ep)
        base_opt = build_optimizer(base)
        moon_opt = build_optimizer(moon)
        caplog.set_level(logging.WARNING)
        checks = RankChecks()
        for step in range(5):
            batches = make_batches(4, seed=100 + step)
            base_loss, base_norm, _ = train_step(base, base_opt, batches)
            moon_loss, moon_norm, success = train_step(moon, moon_opt, batches)
            checks.check(success, f"step {step} failed")
            checks.check(
                abs(moon_loss - base_loss) / base_loss < 2e-2,
                f"step {step} loss {moon_loss:.6f} vs {base_loss:.6f}",
            )
            checks.check(
                abs(moon_norm - base_norm) / base_norm < 5e-2,
                f"step {step} grad norm {moon_norm:.6f} vs {base_norm:.6f}",
            )
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()
