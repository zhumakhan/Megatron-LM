# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP expert weights under Megatron-FSDP v1 ZeRO-1 ('optim') and ZeRO-2 ('optim_grads').

MoonEP's expert weights are ordinary parameters named like TEGroupedMLP's (``linear_fc1.weight{i}``
and ``linear_fc2.weight{i}``), so Megatron-FSDP keeps their fp32 main weights, shards those, the
Adam state and (ZeRO-2) the gradients over expert-DP, and reduces their gradients like those of
any expert linear layer. The GPU tests build GPTModels whose every layer is MoE, set their weights
and only then wrap them as megatron/training's get_model does (GPTModel -> Float16Module ->
Megatron-FSDP), because Megatron-FSDP builds its buffers from the weights it finds.

EP is 4, so expert-DP is 2 on 8 GPUs: with EP equal to the world size ZeRO-1/2 would shard nothing
of the experts. Run with ``torch.distributed.run --nproc-per-node 8`` (one NVLink domain), without
``-x`` (checks are collected and raised on every rank together, see ``RankChecks``), and with
CUDA_DEVICE_MAX_CONNECTIONS unset (see ``needs_concurrent_streams``).
"""

import logging
import math
import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.tensor import DTensor

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel,
    FullyShardedDataParallelV2,
)
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import Adam, OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.moonep_backend.grads import (
    assert_moonep_wrapper_supported,
    finalize_moonep_step,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import unwrap_model
from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import (
    FFN,
    HIDDEN,
    NUM_EXPERTS,
    RankChecks,
    copy_sequential_to_moonep,
    moe_config_kwargs,
    moonep_calls,
    requires_moonep,
    teardown_moonep,
)

pytestmark = [pytest.mark.internal]

EP = 4
SEQ, MBS, VOCAB, NUM_LAYERS, STEPS = 128, 2, 1024, 2, 3
MAX_TOKENS = SEQ * MBS

# MoonEP against alltoall + SequentialMLP (test_moonep_training.py's tolerances): the first step
# starts from identical weights, later steps from weights that drifted apart through the bf16
# rounding of the combine, which also flips near-tie top-k choices.
FIRST_STEP_TOL = {"loss": 1e-2, "grad_norm": 2e-2}
LATER_STEP_TOL = {"loss": 2e-2, "grad_norm": 5e-2}
# Norm of each local expert weight's reduced gradient after the first step.
EXPERT_GRAD_TOL = 1e-2
# MoonEP under Megatron-FSDP against MoonEP under DDP + distributed optimizer, from the same
# weights on the same data. The forward and every rank's local fp32 gradients are the same
# computation; the wrappers differ only in their bucket and shard layouts, i.e. in the order of
# the fp32 sums of the reduce-scatter and of the gradient-norm partial sums (~1e-7 relative).
# Adam's first update is about lr * sign(grad), blind to that; later updates differ by about
# lr * 1e-7, which rounds to the same bf16 weights but for a handful of elements, and a near-tie
# top-k flip caused by one of them moves the loss by < 1e-6 and the gradient norm by ~1e-4. A
# wrapper that mishandles the expert weights (a lost, doubled or wrongly scaled expert gradient, a
# misplaced shard) is off by orders of magnitude more.
WRAPPER_TOL = {"loss": 1e-5, "grad_norm": 1e-3}

# ZeRO-2 reduce-scatters a layer's gradients on Megatron-FSDP's side stream while the backward of
# the layers below runs MoonEP's barrier kernels, which spin until every EP rank arrives. With
# CUDA_DEVICE_MAX_CONNECTIONS=1 all streams of a device share one hardware queue, which can hold a
# rank's reduce-scatter behind a barrier that waits for a peer held behind its own reduce-scatter
# (megatron/training refuses Megatron-FSDP with that setting). The variable takes effect when the
# CUDA context is created, so a test cannot unset it.
needs_concurrent_streams = pytest.mark.skipif(
    os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1",
    reason="ZeRO-2 overlaps NCCL with MoonEP barriers: unset CUDA_DEVICE_MAX_CONNECTIONS",
)

# Transformer Engine's FusedAdam creates state for every parameter, including the empty shards
# that Megatron-FSDP's uneven sharding leaves on some ranks. Apex's and torch's Adam skip
# parameters without gradients, so the ranks of a shard group would put different optimizer keys
# into an fsdp_dtensor checkpoint, whose preprocessing runs one collective per key.
needs_te_fused_adam = pytest.mark.skipif(
    not Adam.__module__.startswith("transformer_engine"),
    reason="fsdp_dtensor optimizer checkpoints need Transformer Engine's FusedAdam",
)

# What megatron/training passes to preprocess_fsdp_dtensor_state_dict for this model.
_FSDP_DTENSOR_ARGS = SimpleNamespace(swiglu=True, num_experts=NUM_EXPERTS)

# MoonEP's ``linear_fc{1,2}.weight{i}`` and SequentialMLP's ``local_experts.{i}.linear_fc{1,2}``.
_EXPERT_PARAM = re.compile(
    r"layers\.(\d+)\.mlp\.experts\."
    r"(?:linear_fc([12])\.weight(\d+)|local_experts\.(\d+)\.linear_fc([12])\.weight)$"
)
# An expert weight's fsdp_dtensor checkpoint entry: global expert id, SwiGLU halves of fc1.
_EXPERT_CHECKPOINT_KEY = re.compile(
    r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)(_w|_v)?$"
)


# ---- models ------------------------------------------------------------------------------------


def _gpt_config(backend, num_layers=NUM_LAYERS, **overrides):
    """test_moonep_training.py's MoE GPT: every layer MoE, no dropout, aux-loss balancing."""
    kwargs = moe_config_kwargs(
        EP,
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
            moe_moonep_max_tokens_per_rank=MAX_TOKENS,
        )
    else:
        kwargs.update(moe_token_dispatcher_type="alltoall")
    return TransformerConfig(**kwargs)


def _build_gpt(config, seed):
    """An unwrapped GPTModel on the GPU; wrap it only once its weights are final."""
    _set_random_seed(seed)
    gpt = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(num_experts=NUM_EXPERTS),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
    )
    return gpt.cuda()


@torch.no_grad()
def _copy_alltoall_to_moonep(base_gpt, moon_gpt):
    """Every weight of an alltoall GPT into a MoonEP GPT; SequentialMLP's local expert ``i``
    becomes MoonEP's ``linear_fc{1,2}.weight{i}``."""
    moon_params = dict(moon_gpt.named_parameters())
    for name, param in base_gpt.named_parameters():
        if ".mlp.experts." not in name:
            moon_params[name].copy_(param)
    for base_layer, moon_layer in zip(base_gpt.decoder.layers, moon_gpt.decoder.layers):
        copy_sequential_to_moonep(base_layer.mlp, moon_layer.mlp)


@torch.no_grad()
def _copy_gpt(src_gpt, dst_gpt):
    """Every weight of a GPT into another one of the same structure."""
    dst_params = dict(dst_gpt.named_parameters())
    for name, param in src_gpt.named_parameters():
        dst_params[name].copy_(param)
    torch.cuda.synchronize()
    dist.barrier()


@torch.no_grad()
def _skew_routing(gpt):
    """Send every token of every MoE layer to the experts of the upper half of the EP ranks.

    The router reads the pre-MLP LayerNorm output, which has zero mean over the hidden dimension,
    so the shared direction ``ones / sqrt(H)`` enters through that LayerNorm's bias: every token
    scores about +4 on the hot experts and -4 on the others, and a tenth of the initial router
    weights decides among the hot ones. MoonEP then copies hot experts to the idle ranks
    (``prefetch_weight``) and reduces their gradients back (``reduce_grad``); the other experts
    get exactly zero gradients.
    """
    hot = torch.arange(NUM_EXPERTS, device="cuda") >= NUM_EXPERTS // 2
    direction = torch.ones(HIDDEN, device="cuda") / HIDDEN**0.5
    for layer in gpt.decoder.layers:
        layer.pre_mlp_layernorm.bias.copy_(4.0 * direction)
        weight = layer.mlp.router.weight
        weight.mul_(0.1)
        weight.add_(torch.where(hot, 1.0, -1.0)[:, None] * direction.to(weight.dtype))


def _fsdp_config(strategy):
    """Megatron-FSDP v1 ZeRO-1 ('optim') or ZeRO-2 ('optim_grads') with fp32 main gradients."""
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy=strategy,
        megatron_fsdp_main_grads_dtype=torch.float32,
        # Megatron-FSDP turns overlapped reduce-scatters on for ZeRO-2 anyway.
        overlap_grad_reduce=strategy == "optim_grads",
        bucket_size=None,
    )


def _wrap_ddp(gpt):
    """GPTModel -> Float16Module -> DDP with the distributed optimizer (ZeRO-1)."""
    config = gpt.config
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, grad_reduce_in_fp32=True, check_for_nan_in_grad=True
    )
    model = DDP(config, ddp_config, Float16Module(config, gpt))
    config.finalize_model_grads_func = finalize_model_grads
    return model


def _optimizer(model):
    """Distributed Adam for a DDP or Megatron-FSDP model, as megatron/training sets it up."""
    config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    optimizer = get_megatron_optimizer(config, [model])
    model.config.grad_scale_func = optimizer.scale_loss
    return optimizer


# ---- training ----------------------------------------------------------------------------------


def _batches(num_microbatches, seed):
    """Random-token micro-batches, different on every rank."""
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


def _forward_step(data_iterator, model):
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


def _forward_backward(model, batches, *, forward_only=False):
    """The global token-mean loss of ``batches`` through forward_backward_no_pipelining."""
    outputs = forward_backward_no_pipelining(
        forward_step_func=_forward_step,
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


def _train_step(model, optimizer, batches, *, expert_grads=False):
    """One iteration in megatron/training/training.py:train_step order.

    Returns ``(loss, grad norm, success, expert grads)``; with ``expert_grads`` the last item is
    the reduced expert gradients' ``_expert_sq_norms``, read between finalize_model_grads and the
    optimizer step.
    """
    model.zero_grad_buffer()
    optimizer.zero_grad()
    loss = _forward_backward(model, batches)
    grads = _expert_sq_norms(model, lambda param: param.grad) if expert_grads else None
    success, grad_norm, _ = optimizer.step()
    return loss, float(grad_norm), success, grads


def _eval_loss(model, batches):
    """The forward-only loss in eval mode, without gradient tracking."""
    model.eval()
    with torch.no_grad():
        loss = _forward_backward(model, batches, forward_only=True)
    model.train()
    return loss


# ---- inspection --------------------------------------------------------------------------------


def _local(tensor):
    """This rank's shard of a DTensor (Megatron-FSDP's are unevenly sharded), else the tensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _expert_param_key(name):
    """``(layer, local expert, fc)`` of an expert weight's parameter name, None for others."""
    match = _EXPERT_PARAM.search(name)
    if match is None:
        return None
    layer, fc, expert, sequential_expert, sequential_fc = match.groups()
    if fc is not None:
        return int(layer), int(expert), int(fc)
    return int(layer), int(sequential_expert), int(sequential_fc)


def _expert_sq_norms(model, tensor_of):
    """``{(layer, local expert, fc): squared L2 norm of tensor_of(param)}`` over the local expert
    weights of a Megatron-FSDP model, summed over their expert-DP shards.

    ``param`` is the optimizer's parameter, a DTensor over this rank's shard of the fp32 main
    weight; ``tensor_of`` returns a tensor or DTensor shard, or None. Collective over expert-DP.
    """
    named = []
    for name, param in unwrap_model(model).param_and_grad_buffer.optimizer_named_parameters:
        key = _expert_param_key(name)
        if key is not None:
            named.append((key, param))
    named.sort(key=lambda item: item[0])
    sums = []
    for _, param in named:
        tensor = tensor_of(param)
        if tensor is None:
            sums.append(torch.zeros((), device="cuda"))
        else:
            sums.append(_local(tensor).detach().float().square().sum())
    sums = torch.stack(sums)
    dist.all_reduce(sums, group=parallel_state.get_expert_data_parallel_group())
    return {key: value for (key, _), value in zip(named, sums.tolist())}


def _adam_state(optimizer, name):
    """``tensor_of`` for ``_expert_sq_norms``: the optimizer's ``name`` state of a parameter."""
    return lambda param: optimizer.optimizer.state.get(param, {}).get(name)


def _expert_weights(gpt):
    """Copies of the bf16 expert weights the forward reads, by (layer, local expert, fc)."""
    weights = {}
    for layer_idx, layer in enumerate(gpt.decoder.layers):
        for fc, params in zip((1, 2), layer.mlp.experts.expert_weights()):
            for i, param in enumerate(params):
                # Between steps the module holds the optimizer's parameter, which wraps this one.
                weight = getattr(param, "orig_param", param)
                weights[(layer_idx, i, fc)] = weight.detach().clone()
    return weights


def _copied_experts(gpt):
    """Per MoonEP layer, whether its last dispatch copied any expert to another rank."""
    return [
        bool((layer.mlp.token_dispatcher._comm_manager.last_plan.experts_to_copy >= 0).any())
        for layer in gpt.decoder.layers
    ]


def _moonep_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "MoonEP" in r.getMessage()
    ]


def _rel(got, want):
    return abs(got - want) / max(abs(want), 1e-12)


def _check_step(checks, step, got, want, tol):
    """Record whether one step's loss and gradient norm (``_train_step`` results) agree."""
    loss, grad_norm, success, _ = got
    want_loss, want_grad_norm, want_success, _ = want
    checks.check(success and want_success, f"step {step}: an optimizer step was skipped")
    checks.check(
        math.isfinite(loss) and math.isfinite(grad_norm),
        f"step {step}: loss {loss}, grad norm {grad_norm}",
    )
    checks.check(
        _rel(loss, want_loss) < tol["loss"], f"step {step}: loss {loss!r} vs {want_loss!r}"
    )
    checks.check(
        _rel(grad_norm, want_grad_norm) < tol["grad_norm"],
        f"step {step}: grad norm {grad_norm!r} vs {want_grad_norm!r}",
    )


def _check_sq_norms(checks, what, got, want, tol):
    """Record whether two ``_expert_sq_norms`` results agree: the norms within ``tol``
    (relative), and exactly zero where ``want`` is zero."""
    checks.check(
        got.keys() == want.keys() and len(want) > 0, f"{what}: {sorted(got)} vs {sorted(want)}"
    )
    for key, want_sq in want.items():
        got_norm, want_norm = math.sqrt(got.get(key, math.nan)), math.sqrt(want_sq)
        layer, expert, fc = key
        tag = f"{what} of layer {layer} local expert {expert} fc{fc}"
        if want_norm == 0.0:
            checks.check(got_norm == 0.0, f"{tag}: {got_norm!r}, expected 0")
        else:
            checks.check(_rel(got_norm, want_norm) < tol, f"{tag}: {got_norm!r} vs {want_norm!r}")


# ---- fsdp_dtensor checkpoints --------------------------------------------------------------------


def _fsdp_dtensor_state_dict(model, optimizer, *, is_loading=False):
    """``(module, state_dict)`` as megatron/training's generate_state_dict builds them for
    ``--ckpt-format fsdp_dtensor``: ``module`` is ``unwrap_model(model)`` (the MegatronFSDP
    module, whose state dict keys carry one ``module.`` level)."""
    module = unwrap_model(model)
    state_dict = {"model": module.state_dict_for_save_checkpoint()}
    metadata = {
        "distrib_optim_sharding_type": "fsdp_dtensor",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True),
    }
    state_dict["optimizer"] = optimizer.sharded_state_dict(
        state_dict, is_loading=is_loading, metadata=metadata
    )
    return module, state_dict


def _save_fsdp_dtensor(model, optimizer, ckpt_dir):
    """megatron/training's synchronous fsdp_dtensor save (no nvidia_resiliency_ext)."""
    module, state_dict = _fsdp_dtensor_state_dict(model, optimizer)
    state_dict = preprocess_fsdp_dtensor_state_dict(_FSDP_DTENSOR_ARGS, state_dict, module)
    dcp.save(state_dict, storage_writer=dcp.FileSystemWriter(ckpt_dir))


def _load_fsdp_dtensor(model, optimizer, ckpt_dir):
    """megatron/training's fsdp_dtensor load: DCP reads into the preprocessed DTensors in place
    (every requested key must exist), then the wrapper's load_state_dict installs the fp32 main
    weights into the bf16 weights and the optimizer takes its state."""
    module, state_dict = _fsdp_dtensor_state_dict(model, optimizer, is_loading=True)
    raw_model_state_dict = state_dict["model"].copy()
    raw_optimizer_state_dict = state_dict["optimizer"].copy()
    state_dict = preprocess_fsdp_dtensor_state_dict(_FSDP_DTENSOR_ARGS, state_dict, module)
    dcp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dcp.FileSystemReader(ckpt_dir),
        planner=DefaultLoadPlanner(allow_partial_load=False),
    )
    model.load_state_dict(raw_model_state_dict, strict=True)
    optimizer.load_state_dict(raw_optimizer_state_dict)


def _expert_checkpoint_keys(entries):
    """``{(layer, fc, expert id, half): key}`` for the expert weights among a checkpoint's
    ``entries``, and the other keys that mention experts (there should be none)."""
    found, unexpected = {}, []
    for key in entries:
        match = _EXPERT_CHECKPOINT_KEY.search(key)
        if match is not None:
            layer, fc, expert, half = match.groups()
            found[(int(layer), int(fc), int(expert), half or "")] = key
        elif ".experts." in key:
            unexpected.append(key)
    return found, unexpected


def _joined_local(entries):
    """The local shards of an expert entry's halves (``_w`` then ``_v``, or the whole weight),
    flattened and joined: the shard of the weight they were split from."""
    return torch.cat([_local(entry).reshape(-1) for entry in entries])


# ---- tests -------------------------------------------------------------------------------------


@requires_moonep
@pytest.mark.skipif(
    Utils.world_size < 2 * EP or Utils.world_size % EP != 0,
    reason=f"needs a multiple of {EP} GPUs, at least {2 * EP}, so that expert-DP >= 2",
)
class TestMoonEPMegatronFSDP:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=EP)
        self.fsdp_models = []

    def teardown_method(self, method):
        # Megatron-FSDP's outstanding communication finishes before anything is freed.
        for model in self.fsdp_models:
            model.stop_communication()
        self.fsdp_models.clear()
        # The MoE metrics tracker sizes its per-layer entries from the first model it sees.
        destroy_moe_metrics_tracker()
        teardown_moonep()

    def _wrap_fsdp(self, gpt, strategy):
        """GPTModel -> Float16Module -> Megatron-FSDP v1, as megatron/training's get_model."""
        config = gpt.config
        model = FullyShardedDataParallel(
            config=config, ddp_config=_fsdp_config(strategy), module=Float16Module(config, gpt)
        )
        self.fsdp_models.append(model)
        config.finalize_model_grads_func = finalize_model_grads
        if model.ddp_config.overlap_grad_reduce:
            # As megatron/training's train() does when gradient reduction is overlapped.
            config.no_sync_func = model.no_sync
        return model

    @pytest.mark.parametrize("skewed", [False, True], ids=["balanced", "skewed"])
    @pytest.mark.parametrize(
        "strategy", ["optim", pytest.param("optim_grads", marks=needs_concurrent_streams)]
    )
    def test_matches_alltoall(self, strategy, skewed, caplog):
        """MoonEP and alltoall + SequentialMLP under the same Megatron-FSDP wrapper and
        optimizer: the same loss and gradient norm at every step (two micro-batches each), and
        after the first step the same reduced gradient norm for every local expert weight. With
        skewed routing MoonEP copies the hot experts to the idle ranks, so part of their
        gradients reaches Megatron-FSDP through reduce_grad, and the other experts' are zero."""
        base_gpt = _build_gpt(_gpt_config("alltoall"), seed=1234)
        moon_gpt = _build_gpt(_gpt_config("moonep"), seed=1234)
        if skewed:
            _skew_routing(base_gpt)
        _copy_alltoall_to_moonep(base_gpt, moon_gpt)
        base, moon = self._wrap_fsdp(base_gpt, strategy), self._wrap_fsdp(moon_gpt, strategy)
        base_opt, moon_opt = _optimizer(base), _optimizer(moon)
        caplog.set_level(logging.WARNING)
        checks = RankChecks()
        for step in range(STEPS):
            batches = _batches(2, seed=100 + step)
            first = step == 0
            want = _train_step(base, base_opt, batches, expert_grads=first)
            got = _train_step(moon, moon_opt, batches, expert_grads=first)
            _check_step(checks, step, got, want, FIRST_STEP_TOL if first else LATER_STEP_TOL)
            if not first:
                continue
            _check_sq_norms(checks, "expert grad norm", got[3], want[3], EXPERT_GRAD_TOL)
            if skewed:
                checks.check(all(_copied_experts(moon_gpt)), "skewed routing copied no expert")
                hot_rank = parallel_state.get_expert_model_parallel_rank() >= EP // 2
                trained = [sq > 0 for sq in want[3].values()]
                checks.check(
                    all(trained) if hot_rank else not any(trained),
                    f"skewed routing: local experts with gradients {trained}",
                )
        checks.check(_moonep_warnings(caplog) == [], f"warnings: {_moonep_warnings(caplog)}")
        checks.assert_all()

    @pytest.mark.parametrize(
        "strategy", ["optim", pytest.param("optim_grads", marks=needs_concurrent_streams)]
    )
    def test_matches_ddp_distributed_optimizer(self, strategy):
        """MoonEP under Megatron-FSDP and under DDP with the distributed optimizer, from the same
        weights on the same data, agree within WRAPPER_TOL at every step (see there for why it
        can be that tight)."""
        ddp_gpt = _build_gpt(_gpt_config("moonep"), seed=1234)
        fsdp_gpt = _build_gpt(_gpt_config("moonep"), seed=1234)
        _copy_gpt(ddp_gpt, fsdp_gpt)
        ddp, fsdp = _wrap_ddp(ddp_gpt), self._wrap_fsdp(fsdp_gpt, strategy)
        ddp_opt, fsdp_opt = _optimizer(ddp), _optimizer(fsdp)
        checks = RankChecks()
        for step in range(STEPS):
            batches = _batches(2, seed=200 + step)
            want = _train_step(ddp, ddp_opt, batches)
            got = _train_step(fsdp, fsdp_opt, batches)
            _check_step(checks, step, got, want, WRAPPER_TOL)
        checks.assert_all()

    @needs_concurrent_streams
    def test_zero2_stress_with_block_recompute(self, caplog):
        """ZeRO-2 with four MoE layers, four micro-batches and full block recompute of the first
        two layers, under skewed routing. Megatron-FSDP reduce-scatters each layer's gradients
        on its side stream as soon as they are complete, so the NCCL kernels overlap the MoonEP
        barrier kernels (dispatch, combine, restaging, recompute replays) of the layers below.
        Every step must finish, with finite losses and gradient norms close to alltoall's; the
        recomputes replay their plans and none routes tokens differently from its forward."""
        recompute = dict(
            recompute_granularity="full", recompute_method="block", recompute_num_layers=2
        )
        num_layers, micro_batches = 4, 4
        base_gpt = _build_gpt(_gpt_config("alltoall", num_layers, **recompute), seed=99)
        moon_gpt = _build_gpt(_gpt_config("moonep", num_layers, **recompute), seed=99)
        _skew_routing(base_gpt)
        _copy_alltoall_to_moonep(base_gpt, moon_gpt)
        base = self._wrap_fsdp(base_gpt, "optim_grads")
        moon = self._wrap_fsdp(moon_gpt, "optim_grads")
        base_opt, moon_opt = _optimizer(base), _optimizer(moon)
        buffer = moon_gpt.decoder.layers[0].mlp.token_dispatcher._comm_manager.buffer
        caplog.set_level(logging.WARNING)
        checks = RankChecks()
        for step in range(STEPS):
            batches = _batches(micro_batches, seed=300 + step)
            want = _train_step(base, base_opt, batches)
            with moonep_calls(buffer) as calls:
                got = _train_step(moon, moon_opt, batches)
            _check_step(checks, step, got, want, LATER_STEP_TOL)
            # Per micro-batch: the two checkpointed layers replay their forward's plan, and
            # every layer's expert gradients are reduced once.
            expected = {"replay": 2 * micro_batches, "reduce": num_layers * micro_batches}
            observed = {name: calls[name] for name in expected}
            checks.check(observed == expected, f"step {step}: MoonEP calls {calls}")
            if step == 0:
                checks.check(all(_copied_experts(moon_gpt)), "skewed routing copied no expert")
        checks.check(_moonep_warnings(caplog) == [], f"warnings: {_moonep_warnings(caplog)}")
        checks.assert_all()

    @needs_te_fused_adam
    def test_fsdp_dtensor_names_experts_by_global_id(self):
        """megatron/training's fsdp_dtensor checkpoint path (preprocess_fsdp_dtensor_state_dict)
        names MoonEP's expert weights like TEGroupedMLP's, by GLOBAL expert id ``g``:
        ``mlp.experts.linear_fc1.weight{g}_w`` and ``_v`` (the SwiGLU gate and up rows) and
        ``mlp.experts.linear_fc2.weight{g}``, in the model and in the optimizer state. The entry
        of ``g = ep_rank * num_local_experts + i`` holds local expert ``i``'s shard, the replicas
        of an expert across expert-DP write the same entries, and no two EP ranks collide."""
        gpt = _build_gpt(_gpt_config("moonep"), seed=1234)
        model = self._wrap_fsdp(gpt, "optim")
        optimizer = _optimizer(model)
        _train_step(model, optimizer, _batches(1, seed=7))  # creates the Adam moments
        module, state_dict = _fsdp_dtensor_state_dict(model, optimizer)
        state_dict = preprocess_fsdp_dtensor_state_dict(_FSDP_DTENSOR_ARGS, state_dict, module)
        model_entries, optim_entries = state_dict["model"], state_dict["optimizer"]["state"]
        model_keys, unexpected = _expert_checkpoint_keys(model_entries)
        optim_keys, unexpected_optim = _expert_checkpoint_keys(optim_entries)
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        num_local = NUM_EXPERTS // EP
        halves = {1: ("_w", "_v"), 2: ("",)}
        expected = {
            (layer, fc, ep_rank * num_local + i, half)
            for layer in range(NUM_LAYERS)
            for i in range(num_local)
            for fc in (1, 2)
            for half in halves[fc]
        }
        checks = RankChecks()
        checks.check(unexpected + unexpected_optim == [], f"{unexpected + unexpected_optim}")
        checks.check(set(model_keys) == expected, f"model expert keys {sorted(model_keys)}")
        checks.check(set(optim_keys) == expected, f"optimizer expert keys {sorted(optim_keys)}")
        checks.check(
            all(key.startswith("module.decoder.layers.") for key in model_keys.values()),
            f"model expert keys {sorted(model_keys.values())}",
        )
        shapes = {1: (FFN, HIDDEN), 2: (HIDDEN, FFN)}
        for (_, fc, _, _), key in model_keys.items():
            shape = tuple(model_entries[key].shape)
            checks.check(shape == shapes[fc], f"{key}: global shape {shape}")
        for name, param in module.param_and_grad_buffer.optimizer_named_parameters:
            local_key = _expert_param_key(name)
            if local_key is None:
                continue
            layer, i, fc = local_key
            ids = [(layer, fc, ep_rank * num_local + i, half) for half in halves[fc]]
            tag = f"layer {layer} expert {ep_rank * num_local + i} fc{fc}"
            entries = [model_entries.get(model_keys.get(key, "")) for key in ids]
            if None not in entries:
                checks.equal(f"{tag} weight", _joined_local(entries), _local(param).reshape(-1))
            state = optimizer.optimizer.state.get(param, {})
            moments = [optim_entries.get(optim_keys.get(key, ""), {}) for key in ids]
            for moment in ("exp_avg", "exp_avg_sq"):
                if moment in state and all(moment in entry for entry in moments):
                    checks.equal(
                        f"{tag} {moment}",
                        _joined_local([entry[moment] for entry in moments]),
                        _local(state[moment]).reshape(-1),
                    )
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (ep_rank, sorted(model_keys), sorted(optim_keys)))
        every_expert = {
            (layer, fc, expert, half)
            for layer in range(NUM_LAYERS)
            for expert in range(NUM_EXPERTS)
            for fc in (1, 2)
            for half in halves[fc]
        }
        written = {key for _, keys, _ in gathered for key in keys}
        checks.check(written == every_expert, f"missing entries {sorted(every_expert - written)}")
        for rank, (other_ep_rank, other_model_keys, other_optim_keys) in enumerate(gathered):
            if other_ep_rank == ep_rank:
                same = other_model_keys == sorted(model_keys)
                same = same and other_optim_keys == sorted(optim_keys)
                checks.check(same, f"rank {rank} names this EP rank's experts differently")
            else:
                clash = set(other_model_keys) & set(model_keys)
                clash |= set(other_optim_keys) & set(optim_keys)
                checks.check(not clash, f"rank {rank} writes the same entries {sorted(clash)}")
        checks.assert_all()

    @needs_te_fused_adam
    def test_fsdp_dtensor_round_trip_optim_to_optim_grads(self, tmp_path_dist_ckpt):
        """A ZeRO-1 ('optim') checkpoint saved and loaded with megatron/training's fsdp_dtensor
        steps (synchronous torch DCP, no nvidia_resiliency_ext) into a differently initialized
        ZeRO-2 ('optim_grads') model; every key the loader asks for must exist in the
        checkpoint. Megatron-FSDP v1 loads GLU models with strict=False, so a key mismatch could
        silently load nothing: the loaded bf16 expert weights must equal the saver's and differ
        from the loader's own, its loss must be the saver's, and every MoonEP expert weight's
        fp32 main weight and Adam moments must match the saver's."""
        src_gpt = _build_gpt(_gpt_config("moonep"), seed=1234)
        src = self._wrap_fsdp(src_gpt, "optim")
        src_opt = _optimizer(src)
        for step in range(2):
            _train_step(src, src_opt, _batches(2, seed=400 + step))
        dst_gpt = _build_gpt(_gpt_config("moonep"), seed=4321)
        dst = self._wrap_fsdp(dst_gpt, "optim_grads")
        dst_opt = _optimizer(dst)
        # No forward may run before the load: Megatron-FSDP gathers the loaded shards into the
        # bf16 weights of all ranks in the first forward's parameter all-gather.
        initial = _expert_weights(dst_gpt)
        with TempNamedDir(tmp_path_dist_ckpt / "moonep_fsdp_dtensor", sync=True) as ckpt_dir:
            _save_fsdp_dtensor(src, src_opt, ckpt_dir)
            _load_fsdp_dtensor(dst, dst_opt, ckpt_dir)
        probe = _batches(2, seed=500)
        src_loss, dst_loss = _eval_loss(src, probe), _eval_loss(dst, probe)
        checks = RankChecks()
        # Same bf16 weights, same kernels: the tolerance only absorbs a different GEMM choice.
        checks.check(_rel(dst_loss, src_loss) < 1e-5, f"loss {dst_loss!r} vs {src_loss!r}")
        src_weights, dst_weights = _expert_weights(src_gpt), _expert_weights(dst_gpt)
        for key, weight in src_weights.items():
            checks.equal(f"bf16 expert weight {key}", dst_weights[key], weight)
            checks.check(not torch.equal(dst_weights[key], initial[key]), f"{key} not loaded")
        # Same values in shards of possibly different layouts: fp32 summation order only.
        got = _expert_sq_norms(dst, lambda param: param)
        want = _expert_sq_norms(src, lambda param: param)
        _check_sq_norms(checks, "fp32 main weight", got, want, 1e-5)
        for name in ("exp_avg", "exp_avg_sq"):
            got = _expert_sq_norms(dst, _adam_state(dst_opt, name))
            want = _expert_sq_norms(src, _adam_state(src_opt, name))
            _check_sq_norms(checks, name, got, want, 1e-5)
        checks.assert_all()


class TestMoonEPWrapperCheck:
    """MoonEP accepts only data-parallel wrappers that leave its expert weights unsharded: DDP
    (with or without the distributed optimizer) and Megatron-FSDP v1 'optim' (ZeRO-1) or
    'optim_grads' (ZeRO-2). finalize_model_grads runs the check (finalize_moonep_step) in the
    first step of a set of model chunks. It only reads each chunk's type and ``ddp_config``, so
    stand-ins replace the wrappers: no GPU, model or process group is needed."""

    @staticmethod
    def _chunk(**ddp_config):
        chunk = torch.nn.Module()
        chunk.ddp_config = DistributedDataParallelConfig(**ddp_config)
        return chunk

    @pytest.mark.parametrize(
        "dense, experts",
        [("optim_grads_params", None), ("optim", "optim_grads_params"), ("no_shard", None)],
        ids=["zero3", "zero3-experts", "no-shard"],
    )
    def test_rejects_other_megatron_fsdp_strategies(self, dense, experts):
        chunk = self._chunk(
            use_megatron_fsdp=True,
            data_parallel_sharding_strategy=dense,
            expert_data_parallel_sharding_strategy=experts,
        )
        match = "MoonEP with Megatron-FSDP supports the 'optim'"
        with pytest.raises(ValueError, match=match):
            assert_moonep_wrapper_supported([chunk])
        # finalize_model_grads runs it through finalize_moonep_step, for every chunk.
        supported = self._chunk(use_megatron_fsdp=True, data_parallel_sharding_strategy="optim")
        with pytest.raises(ValueError, match=match):
            finalize_moonep_step([supported, chunk])

    @pytest.mark.parametrize(
        "ddp_config, match",
        [
            ({"num_distributed_optimizer_instances": 2}, "hybrid sharding"),
            ({"outer_dp_sharding_strategy": "optim"}, "hybrid sharding"),
            (
                {"megatron_fsdp_enable_fine_grained_param_gather": True},
                "whole-layer parameter gathering",
            ),
        ],
        ids=["optimizer-instances", "outer-dp-sharding", "fine-grained-param-gather"],
    )
    def test_rejects_megatron_fsdp_gather_modes(self, ddp_config, match):
        chunk = self._chunk(
            use_megatron_fsdp=True, data_parallel_sharding_strategy="optim_grads", **ddp_config
        )
        with pytest.raises(ValueError, match=match):
            assert_moonep_wrapper_supported([chunk])

    def test_requires_moe_layers_inside_fsdp_units(self):
        """Only an enclosing FSDP unit's hooks gather MoonEP's expert weights (their modules
        never run forward), so an MoE layer outside every unit is rejected."""

        class Unit(torch.nn.Module):
            pass

        class Experts(torch.nn.Module):
            _is_moonep_experts = True

        chunk = self._chunk(use_megatron_fsdp=True, data_parallel_sharding_strategy="optim_grads")
        chunk.fsdp_unit_modules = [Unit]
        chunk.layer = Unit()
        chunk.layer.experts = Experts()
        assert_moonep_wrapper_supported([chunk])
        chunk.head = Experts()
        with pytest.raises(ValueError, match="head is outside every unit"):
            assert_moonep_wrapper_supported([chunk])
        chunk.fsdp_unit_modules = []
        with pytest.raises(ValueError, match="layer.experts is outside every unit"):
            assert_moonep_wrapper_supported([chunk])

    @pytest.mark.parametrize(
        "wrapper",
        [FullyShardedDataParallelV2, TorchFullyShardedDataParallel],
        ids=["megatron-fsdp-v2", "torch-fsdp2"],
    )
    def test_rejects_parameter_sharding_wrappers(self, wrapper):
        # Only the type is checked; an instance made without __init__ stands in for a wrapper.
        with pytest.raises(ValueError, match=f"{wrapper.__name__} shards parameters"):
            assert_moonep_wrapper_supported([object.__new__(wrapper)])

    @pytest.mark.parametrize(
        "ddp_config",
        [
            {},
            {"use_distributed_optimizer": True},
            {"use_megatron_fsdp": True, "data_parallel_sharding_strategy": "optim"},
            {"use_megatron_fsdp": True, "data_parallel_sharding_strategy": "optim_grads"},
            {
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim",
                "expert_data_parallel_sharding_strategy": "optim_grads",
            },
        ],
        ids=["ddp", "ddp-distopt", "fsdp-optim", "fsdp-optim-grads", "fsdp-mixed"],
    )
    def test_accepts_zero1_and_zero2(self, ddp_config):
        assert_moonep_wrapper_supported([self._chunk(**ddp_config)])
