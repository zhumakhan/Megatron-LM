# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend in a training loop: DDP gradients, optimizers and parity with alltoall.

MoonEP's expert weights are ordinary parameters (``linear_fc{1,2}.weight{i}``, marked
``allreduce=False``). DDP keeps their gradients (fp32 here) in its expert-parallel grad buffer
and scales, NaN-checks and reduces them over expert-DP, and the standard optimizer chain (dense
optimizer, then expert optimizer) steps them, exactly like SequentialMLP's expert weights.

Every test builds a GPTModel (all layers MoE) under Float16Module + DDP twice, once on the
alltoall dispatcher with SequentialMLP experts and once on MoonEP, with identical weights, and
drives them through ``forward_backward_no_pipelining`` with ``finalize_model_grads``. Cells
named ``expert-dp-2`` run EP = world size / 2, so every expert has two replicas: DDP reduces
their gradients over expert-DP, and the distributed optimizer (ZeRO-1) shards their optimizer
state and all-gathers their weights after each step.

Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>`` and without
``-x``: checks are collected and raised on every rank together (see ``RankChecks``).
"""

import logging
import re

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import (
    OptimizerConfig,
    get_megatron_optimizer,
    get_mup_config_overrides,
    get_standard_config_overrides,
)
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import (
    FFN,
    HIDDEN,
    NUM_EXPERTS,
    RankChecks,
    moe_config_kwargs,
    requires_moonep,
    teardown_moonep,
)

pytestmark = [pytest.mark.internal, requires_moonep]

SEQ, MBS, VOCAB, NUM_LAYERS = 128, 2, 1024, 2
LR = 1e-3
TOL = 1e-2  # relative L2 error of a gradient against the alltoall baseline
# Relative tolerances (loss, grad norm) of a training step against the alltoall baseline. The
# first step starts from identical weights; later steps compound the bf16 rounding differences
# of the two dispatchers (MoonEP's combine sums the expert outputs in fp32).
FIRST_STEP_TOL, LATER_STEP_TOL = (1e-2, 2e-2), (2e-2, 5e-2)
# Parameter-group settings that optimizer config overrides set.
GROUP_SETTINGS = ("is_expert_parallel", "max_lr", "min_lr", "lr_mult", "wd_mult", "eps")
# SequentialMLP's expert ``i`` weight ``...experts.local_experts.{i}.linear_fc{1,2}.weight`` is
# MoonEP's ``...experts.linear_fc{1,2}.weight{i}`` (as in ``copy_sequential_to_moonep``).
SEQUENTIAL_EXPERT_WEIGHT = re.compile(
    r"(.*\.experts\.)local_experts\.(\d+)\.(linear_fc[12])\.weight"
)


def moonep_param_name(name):
    """The MoonEP GPT's name of an alltoall GPT parameter."""
    match = SEQUENTIAL_EXPERT_WEIGHT.fullmatch(name)
    if match is None:
        return name
    prefix, index, linear = match.groups()
    return f"{prefix}{linear}.weight{index}"


def gpt_config(ep, backend, num_layers=NUM_LAYERS, **overrides):
    """The GPT config of the alltoall baseline (SequentialMLP experts) or of MoonEP."""
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


def build_model(config, seed, *, distopt, average_in_collective=False, **ddp_overrides):
    """GPTModel -> Float16Module -> DDP, the order megatron/training uses.

    ``ddp_overrides`` go to ``DistributedDataParallelConfig``. With ``overlap_grad_reduce``,
    every micro-batch but the last runs under ``model.no_sync``, as megatron/training sets up.
    """
    _set_random_seed(seed)
    gpt = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(num_experts=NUM_EXPERTS),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
    )
    gpt.cuda()
    ddp_kwargs = dict(
        use_distributed_optimizer=distopt,
        average_in_collective=average_in_collective,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
    )
    ddp_kwargs.update(ddp_overrides)
    ddp_config = DistributedDataParallelConfig(**ddp_kwargs)
    model = DDP(config, ddp_config, Float16Module(config, gpt))
    config.finalize_model_grads_func = finalize_model_grads
    if ddp_config.overlap_grad_reduce:
        config.no_sync_func = model.no_sync
    return model, gpt


@torch.no_grad()
def copy_gpt_weights(base_gpt, moon_gpt):
    """Copy an alltoall GPT's weights into a MoonEP GPT, parameter by parameter: the same names,
    except that expert ``i``'s ``local_experts.{i}.linear_fc{1,2}.weight`` is MoonEP's
    ``linear_fc{1,2}.weight{i}``."""
    moon_params = dict(moon_gpt.named_parameters())
    base_params = {moonep_param_name(name): p for name, p in base_gpt.named_parameters()}
    assert base_params.keys() == moon_params.keys(), "the GPT parameters do not map one to one"
    for name, param in base_params.items():
        moon_params[name].copy_(param)
    torch.cuda.synchronize()
    dist.barrier()


def build_pair(ep, *, distopt=True, ddp_overrides=None, **config_overrides):
    """(base_model, base_gpt, moon_model, moon_gpt) with identical weights, before optimizers
    (the distributed optimizer copies its fp32 main weights from the model when it is built).

    ``num_layers`` (default 2) goes to the config. With more than one layer the second layer's
    input already differs by bf16 rounding (combine accumulates in fp32), which flips near-tie
    top-k choices; tests that compare individual gradients use one layer.
    """
    ddp_overrides = ddp_overrides or {}
    base, base_gpt = build_model(
        gpt_config(ep, "alltoall", **config_overrides), seed=1234, distopt=distopt, **ddp_overrides
    )
    moon, moon_gpt = build_model(
        gpt_config(ep, "moonep", **config_overrides), seed=1234, distopt=distopt, **ddp_overrides
    )
    copy_gpt_weights(base_gpt, moon_gpt)
    return base, base_gpt, moon, moon_gpt


def build_optimizer(model, *, distopt=True, **overrides):
    kwargs = dict(
        optimizer="adam",
        lr=LR,
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


def train_step(model, optimizer, batches):
    """One iteration in megatron/training/training.py:train_step order."""
    model.zero_grad_buffer()
    optimizer.zero_grad()
    loss = forward_backward(model, batches)
    success, grad_norm, _ = optimizer.step()
    return loss, float(grad_norm), success


def check_step(checks, step, got, want):
    """Compare MoonEP's ``train_step`` result with the alltoall baseline's."""
    (loss, norm, success), (base_loss, base_norm, _) = got, want
    loss_tol, norm_tol = FIRST_STEP_TOL if step == 0 else LATER_STEP_TOL
    checks.check(success, f"step {step}: the optimizer skipped the MoonEP model's update")
    checks.check(
        abs(loss - base_loss) / base_loss < loss_tol,
        f"step {step}: loss {loss:.6f} vs alltoall {base_loss:.6f}",
    )
    checks.check(
        abs(norm - base_norm) / base_norm < norm_tol,
        f"step {step}: grad norm {norm:.6f} vs alltoall {base_norm:.6f}",
    )


def moonep_warnings(caplog, text=""):
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "MoonEP" in r.getMessage() and text in r.getMessage()
    ]


def expert_weights_by_name(gpt):
    """The model's expert weights by parameter name (SequentialMLP's or MoonEP's)."""
    return {name: param for name, param in gpt.named_parameters() if ".mlp.experts." in name}


def sub_optimizer_layout(optimizer, model):
    """Per sub-optimizer of the chain: its class, which of the model's DDP buffer lists it steps
    (``buffers``: dense, ``expert_parallel_buffers``: experts) and their parameter elements."""
    kinds = {
        tuple(map(id, model.buffers)): "dense",
        tuple(map(id, model.expert_parallel_buffers)): "expert",
    }
    layout = []
    for sub in optimizer.chained_optimizers:
        kind = kinds.get(tuple(map(id, sub.buffers)), "other")
        numel = sum(param.numel() for buffer in sub.buffers for param in buffer.params)
        layout.append((type(sub).__name__, kind, numel))
    return layout


def owned_numel(param):
    """Elements of ``param`` whose fp32 main copy this rank's distributed optimizer steps."""
    main_param = getattr(param, "main_param", None)
    return 0 if main_param is None else main_param.numel()


def param_group_settings(optimizer, gpt):
    """For every parameter with optimizer state on this rank, by its MoonEP name: the indices of
    its sub-optimizer and parameter group, and the group's ``GROUP_SETTINGS``."""
    names = {param: moonep_param_name(name) for name, param in gpt.named_parameters()}
    settings = {}
    for index, sub in enumerate(optimizer.chained_optimizers):
        # DistributedOptimizer: each group's model parameters that have a shard on this rank.
        groups = zip(
            sub.optimizer.param_groups, sub.model_float16_groups, sub.model_fp32_groups, strict=True
        )
        for group_index, (group, float16_params, fp32_params) in enumerate(groups):
            setting = {key: group.get(key) for key in GROUP_SETTINGS}
            setting.update(optimizer=index, group=group_index)
            for param in [*float16_params, *fp32_params]:
                settings[names[param]] = setting
    return settings


def expert_replicas_match(gpt):
    """Whether every MoonEP expert weight is bitwise equal on all its expert-DP replicas.
    Collective over expert-DP."""
    weights = expert_weights_by_name(gpt).values()
    flat = torch.cat([weight.detach().reshape(-1) for weight in weights])
    group = gpt.decoder.layers[0].mlp.experts.expt_dp_group
    gathered = [torch.empty_like(flat) for _ in range(group.size())]
    dist.all_gather(gathered, flat, group=group)
    return all(torch.equal(replica, gathered[0]) for replica in gathered[1:])


# (expert_dp, calculate_per_token_loss, average_in_collective), ordered by EP size. DDP rejects
# per-token loss with average_in_collective.
FINALIZE_CELLS = [
    pytest.param(1, False, False, id="expert-dp-1"),
    pytest.param(1, True, False, id="expert-dp-1-per-token"),
    pytest.param(2, False, False, id="expert-dp-2"),
    pytest.param(2, True, False, id="expert-dp-2-per-token"),
    pytest.param(2, False, True, id="expert-dp-2-average-in-collective"),
]


class TestMoonEPTraining:
    def setup_method(self, method):
        self.ep = Utils.world_size
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=self.ep)

    def teardown_method(self, method):
        # The global MoE metrics tracker sizes its per-layer entries from the first model it sees.
        destroy_moe_metrics_tracker()
        teardown_moonep()

    def _use_expert_dp(self, expert_dp):
        """Run with EP = world size / expert_dp: every expert has ``expert_dp`` replicas."""
        ep = Utils.world_size // expert_dp
        if ep < 2:
            pytest.skip(f"expert-DP {expert_dp} leaves EP={ep} on {Utils.world_size} GPUs")
        if ep != self.ep:
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)
            self.ep = ep

    # ---- gradients: DDP reduces, scales and checks them ------------------------------------

    @pytest.mark.parametrize("expert_dp, per_token, average_in_collective", FINALIZE_CELLS)
    def test_finalize_matches_ddp(self, expert_dp, per_token, average_in_collective):
        """After the backward and finalize_model_grads, MoonEP's expert gradients are DDP's fp32
        main grads, reduced over expert-DP and scaled (per token or per micro-batch, summed or
        averaged in the collective) like SequentialMLP's, and the placeholder grads MoonEP hands
        autograd are gone. One layer: a second layer's input differs by bf16 rounding and flips
        near-tie routing, which makes individual expert gradients incomparable."""
        self._use_expert_dp(expert_dp)
        base, base_gpt, moon, moon_gpt = build_pair(
            self.ep,
            distopt=False,
            ddp_overrides=dict(average_in_collective=average_in_collective),
            calculate_per_token_loss=per_token,
            num_layers=1,
        )
        batches = make_batches(2, seed=1)
        for model in (base, moon):
            model.zero_grad_buffer()
            forward_backward(model, batches)
        checks = RankChecks()
        layers = zip(base_gpt.decoder.layers, moon_gpt.decoder.layers)
        for layer_idx, (base_layer, moon_layer) in enumerate(layers):
            fc1, fc2 = moon_layer.mlp.experts.expert_weights()
            for i, expert in enumerate(base_layer.mlp.experts.local_experts):
                tag = f"layer {layer_idx} expert {i}"
                for name, param in (("fc1", fc1[i]), ("fc2", fc2[i])):
                    main_grad = param.main_grad
                    checks.check(
                        main_grad.dtype == torch.float32, f"{tag} {name}: {main_grad.dtype} grad"
                    )
                    checks.check(main_grad.count_nonzero() > 0, f"{tag} {name}: zero main_grad")
                    checks.check(param.grad is None, f"{tag} {name}: .grad outlived DDP's hook")
                want_fc1 = expert.linear_fc1.weight.main_grad
                want_fc2 = expert.linear_fc2.weight.main_grad
                checks.close(f"{tag} gate", fc1[i].main_grad[:FFN], want_fc1[:FFN], TOL)
                checks.close(f"{tag} up", fc1[i].main_grad[FFN:], want_fc1[FFN:], TOL)
                checks.close(f"{tag} down", fc2[i].main_grad, want_fc2, TOL)
            checks.close(
                f"layer {layer_idx} router",
                moon_layer.mlp.router.weight.main_grad,
                base_layer.mlp.router.weight.main_grad,
                TOL,
            )
        checks.assert_all()

    def test_nan_check_covers_expert_grads(self):
        """DDP's check_for_nan_in_grad covers MoonEP's expert gradients like any expert
        parameter's: an Inf in one expert weight's main grad fails finalize_model_grads on every
        rank, before the expert-DP collective."""
        if get_rerun_state_machine().get_mode() != RerunMode.DISABLED:
            pytest.skip("DDP raises on non-finite gradients only with reruns disabled (default)")
        moon, moon_gpt = build_model(
            gpt_config(self.ep, "moonep", num_layers=1), seed=1234, distopt=False
        )
        moon.zero_grad_buffer()
        moon.config.finalize_model_grads_func = None
        forward_backward(moon, make_batches(1, seed=2))
        moon.config.finalize_model_grads_func = finalize_model_grads
        checks = RankChecks()
        buffers = moon.buffers + moon.expert_parallel_buffers
        finite = all(bool(torch.isfinite(buffer.grad_data).all()) for buffer in buffers)
        checks.check(finite, "non-finite gradients before the Inf was planted")
        fc1, _ = moon_gpt.decoder.layers[0].mlp.experts.expert_weights()
        fc1[0].main_grad[0, 0] = float("inf")
        error = ""
        try:
            finalize_model_grads([moon])
        except RuntimeError as exc:
            error = str(exc)
        checks.check(
            "found Inf in local grad norm" in error, f"finalize_model_grads raised {error!r}"
        )
        checks.assert_all()

    # ---- optimizers: the standard chain steps the expert weights ----------------------------

    def test_param_group_overrides_match_sequential(self):
        """MoonEP's expert weights land in the same sub-optimizer and parameter group, with the
        same lr, eps and weight-decay settings, as SequentialMLP's, here under the standard
        overrides plus muP's lr and eps scaling of hidden weights. At expert-DP 1 every expert
        weight has optimizer state on its rank."""
        base, base_gpt, moon, moon_gpt = build_pair(self.ep)
        settings = {}
        for tag, model, gpt in (("alltoall", base, base_gpt), ("moonep", moon, moon_gpt)):
            config = OptimizerConfig(
                optimizer="adam",
                lr=LR,
                min_lr=LR / 10,
                adam_eps=1e-8,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
            )
            overrides = {
                **get_standard_config_overrides(config),
                **get_mup_config_overrides(config, mup_width_mult=4.0),
            }
            optimizer = get_megatron_optimizer(config, [model], config_overrides=overrides)
            settings[tag] = param_group_settings(optimizer, gpt)
        got, want = settings["moonep"], settings["alltoall"]
        checks = RankChecks()
        checks.check(got.keys() == want.keys(), f"other parameters: {got.keys() ^ want.keys()}")
        for name in sorted(got.keys() & want.keys()):
            checks.check(got[name] == want[name], f"{name}: {got[name]}, alltoall {want[name]}")
        for name in expert_weights_by_name(moon_gpt):
            setting = got.get(name)
            # muP scaled the expert weights' lr, so the comparison above is not vacuous.
            ok = setting is not None and setting["is_expert_parallel"] and setting["max_lr"] < LR
            checks.check(ok, f"{name}: {setting}")
        checks.assert_all()

    def test_loss_parity(self, caplog):
        """Five steps of plain DDP (gradient all-reduce, no ZeRO) with Megatron's mixed-precision
        optimizer: the optimizer chain is alltoall's, and loss and grad norm track the alltoall
        baseline step by step."""
        base, _, moon, _ = build_pair(self.ep, distopt=False)
        base_opt = build_optimizer(base, distopt=False)
        moon_opt = build_optimizer(moon, distopt=False)
        chain = [type(sub).__name__ for sub in moon_opt.chained_optimizers]
        want_chain = [type(sub).__name__ for sub in base_opt.chained_optimizers]
        caplog.set_level(logging.WARNING)
        checks = RankChecks()
        checks.check(chain == want_chain, f"optimizer chain {chain}, alltoall {want_chain}")
        for step in range(5):
            batches = make_batches(2, seed=100 + step)
            want = train_step(base, base_opt, batches)
            check_step(checks, step, train_step(moon, moon_opt, batches), want)
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()

    @pytest.mark.parametrize("expert_dp", [1, 2], ids=["expert-dp-1", "expert-dp-2"])
    def test_distributed_optimizer_matches_alltoall(self, expert_dp, caplog):
        """The distributed optimizer (ZeRO-1) steps MoonEP's expert weights like SequentialMLP's:
        the chain has alltoall's structure (a dense and an expert DistributedOptimizer), every
        expert weight is a trainable bf16 expert parameter in DDP's expert-parallel buffers only,
        every element of it is stepped on exactly one expert-DP rank, and three steps track the
        alltoall baseline. Expert-DP 2 shards the expert optimizer state over two ranks."""
        self._use_expert_dp(expert_dp)
        base, _, moon, moon_gpt = build_pair(self.ep)
        base_opt, moon_opt = build_optimizer(base), build_optimizer(moon)
        checks = RankChecks()
        layout = sub_optimizer_layout(moon_opt, moon)
        want_layout = sub_optimizer_layout(base_opt, base)
        checks.check(layout == want_layout, f"optimizer chain {layout}, alltoall {want_layout}")
        kinds = [kind for _, kind, _ in layout]
        checks.check(kinds == ["dense", "expert"], f"optimizer chain {layout}")
        in_expert_buffers = {p for buffer in moon.expert_parallel_buffers for p in buffer.params}
        in_dense_buffers = {p for buffer in moon.buffers for p in buffer.params}
        weights = expert_weights_by_name(moon_gpt)
        for name, param in weights.items():
            buffered = param in in_expert_buffers and param not in in_dense_buffers
            checks.check(buffered, f"{name}: not in DDP's expert-parallel buffers only")
            expert = getattr(param, "allreduce", True) is False
            ok = param.requires_grad and expert and param.dtype == torch.bfloat16
            checks.check(ok, f"{name}: requires_grad={param.requires_grad}, {param.dtype}")
        owned = torch.tensor([owned_numel(p) for p in weights.values()], device="cuda")
        dist.all_reduce(owned, group=moon_gpt.decoder.layers[0].mlp.experts.expt_dp_group)
        numel = torch.tensor([p.numel() for p in weights.values()], device="cuda")
        checks.check(
            torch.equal(owned, numel), "expert weight elements not stepped exactly once (expert-DP)"
        )
        caplog.set_level(logging.WARNING)
        for step in range(3):
            batches = make_batches(2, seed=3 + step)
            want = train_step(base, base_opt, batches)
            check_step(checks, step, train_step(moon, moon_opt, batches), want)
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()

    def test_overlapped_grad_reduce_and_param_gather(self, caplog):
        """DDP with overlap_grad_reduce and overlap_param_gather and the distributed optimizer at
        expert-DP 2, with one DDP bucket per MoE layer's experts, so NCCL reduce-scatters and
        all-gathers of expert weights run next to MoonEP's kernels. The overlapped reduction
        needs every expert weight's DDP hook, which MoonEP's placeholder grads trigger. No
        module pre-hook gathers the expert weights before MoonEP stages them in its dispatch,
        so ensure_params_ready must: after every forward the two replicas of each expert weight
        are equal, and three steps track the alltoall baseline."""
        self._use_expert_dp(2)
        ddp_overrides = dict(
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            # One bucket per MoE layer's local experts (SwiGLU fc1 [2*FFN, H] and fc2 [H, FFN]).
            bucket_size=NUM_EXPERTS // self.ep * 3 * HIDDEN * FFN,
        )
        base, _, moon, moon_gpt = build_pair(self.ep, ddp_overrides=ddp_overrides)
        base_opt = build_optimizer(base, overlap_param_gather=True)
        moon_opt = build_optimizer(moon, overlap_param_gather=True)
        caplog.set_level(logging.WARNING)
        checks = RankChecks()
        num_groups = len(moon.expert_parallel_bucket_groups)
        checks.check(num_groups == NUM_LAYERS, f"{num_groups} expert bucket groups")
        for step in range(3):
            batches = make_batches(2, seed=20 + step)
            want = train_step(base, base_opt, batches)
            # train_step, looking at the expert weights between the forward-backward and the step
            moon.zero_grad_buffer()
            moon_opt.zero_grad()
            loss = forward_backward(moon, batches)
            replicas_match = expert_replicas_match(moon_gpt)
            checks.check(replicas_match, f"step {step}: expert weight replicas differ")
            success, grad_norm, _ = moon_opt.step()
            check_step(checks, step, (loss, float(grad_norm), success), want)
        checks.check(moonep_warnings(caplog) == [], f"warnings: {moonep_warnings(caplog)}")
        checks.assert_all()
