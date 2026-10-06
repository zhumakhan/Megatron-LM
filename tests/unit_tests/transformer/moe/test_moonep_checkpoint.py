# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend: distributed checkpoints interchangeable with SequentialMLP.

The MoonEP expert weights are ordinary bf16 parameters, ``linear_fc1.weight{i}`` ``[2*Hi, H]``
and ``linear_fc2.weight{i}`` ``[H, Hi]``, and their sharded state dict mirrors SequentialMLP's
torch_dist layout. Checkpoints therefore move bit for bit between MoonEP and SequentialMLP,
reshard over expert parallelism, and carry the standard distributed optimizer's state of the
expert weights in both of its torch_dist formats.

Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>``, without ``-x``:
checks are gathered with ``RankChecks`` so every rank fails together.
"""

from argparse import Namespace
from collections import Counter

import pytest
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import load, load_plain_tensors, load_tensors_metadata, save
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
    apply_factories,
)
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.moonep_backend import iter_moonep_experts
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.checkpointing import _build_sharded_state_dict_metadata
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_utils import (
    FFN,
    HIDDEN,
    NUM_EXPERTS,
    RankChecks,
    alltoall_config,
    build_moe_layer,
    free_moonep_models,
    moe_config_kwargs,
    moonep_config,
    requires_moonep,
    teardown_moonep,
)

pytestmark = [pytest.mark.internal, requires_moonep]

SEQ, MBS, VOCAB, NUM_LAYERS = 64, 2, 1024, 2
MAX_TOKENS = SEQ * MBS
TOL = 1e-2
PREFIX = "decoder.layers.0.mlp."
# SequentialMLP's sharded keys of the expert weights, under the MoE layer's `experts.` module.
FC1_KEY = f"{PREFIX}experts.experts.linear_fc1.weight"
FC2_KEY = f"{PREFIX}experts.experts.linear_fc2.weight"
EXTRA_STATE_KEYS = [f"{PREFIX}experts.experts.linear_fc{k}._extra_state" for k in (1, 2)]


def _expert_dp_ep():
    """world/2 (4 on 8 GPUs): expert-DP is 2, so the expert weights are replicated and the
    expert optimizer state is sharded. The world size below 4 GPUs."""
    ep = Utils.world_size // 2
    return ep if ep >= 2 and NUM_EXPERTS % ep == 0 else Utils.world_size


def _init(ep):
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)
    _set_random_seed(123)


def _layer(backend, ep, seed):
    """An MoE layer with MoonEP or SequentialMLP (alltoall) experts, initialized from ``seed``."""
    _set_random_seed(seed)
    if backend == "moonep":
        return build_moe_layer(moonep_config(ep, MAX_TOKENS))
    return build_moe_layer(alltoall_config(ep))


def _expert_weights(layer):
    """``[(fc1, fc2)]`` of the local experts of a MoonEP or SequentialMLP layer."""
    experts = layer.experts
    if getattr(experts, "_is_moonep_experts", False):
        return list(zip(*experts.expert_weights()))
    return [(e.linear_fc1.weight, e.linear_fc2.weight) for e in experts.local_experts]


def _global_experts(layer):
    """``(fc1 [E, 2*FFN, H], fc2 [E, H, FFN])`` of every expert of a MoonEP layer, gathered over
    EP in global expert order."""
    experts = layer.experts
    gathered = []
    for weights in experts.expert_weights():
        local = torch.stack([w.detach() for w in weights])
        parts = [torch.empty_like(local) for _ in range(experts.ep_group.size())]
        dist.all_gather(parts, local, group=experts.ep_group)
        gathered.append(torch.cat(parts))
    return gathered


def _hidden():
    """This rank's ``[SEQ, MBS, H]`` bf16 input of an MoE layer."""
    gen = torch.Generator(device="cuda").manual_seed(4000 + dist.get_rank())
    return torch.randn(SEQ, MBS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)


@torch.no_grad()
def _eval_forward(layer, hidden):
    """The layer's output in eval mode without grad; a MoonEP layer stages its weights anew."""
    layer.eval()
    output, _ = layer(hidden)
    layer.train()
    return output


def _load_into(layer, ckpt_dir):
    """Load a layer checkpoint into ``layer``; returns the missing and unexpected keys."""
    loaded = load(layer.sharded_state_dict(prefix=PREFIX), ckpt_dir)
    state_dict = {k.removeprefix(PREFIX): v for k, v in loaded.items()}
    return layer.load_state_dict(state_dict, strict=False)


def _shards(sharded_state_dict):
    """Multiset of every shard once the SwiGLU factories are built: ``(key, global offset,
    layout)``, a tensor's layout being its dtype, local and global shape, fragmentation,
    prepended axes and replica id."""
    apply_factories(sharded_state_dict)
    shards = Counter()
    for sh in nested_values(sharded_state_dict):
        if isinstance(sh, ShardedTensor):
            layout = (
                sh.dtype,
                sh.local_shape,
                sh.global_shape,
                sh.axis_fragmentations,
                sh.prepend_axis_num,
                sh.replica_id,
            )
            shards[(sh.key, sh.global_offset, layout)] += 1
        elif isinstance(sh, ShardedObject):
            shards[(sh.key, sh.global_offset, (sh.global_shape, sh.replica_id))] += 1
        else:
            shards[("not sharded", type(sh).__name__)] += 1
    return shards


def _checkpoint_layout(ckpt_dir):
    """Global shape and dtype of every tensor and object (by unique key) in a checkpoint."""
    metadata = load_sharded_metadata(ckpt_dir)
    return {key: (sh.global_shape, getattr(sh, "dtype", None)) for key, sh in metadata.items()}


def _check_same_checkpoint(checks, first_dir, second_dir):
    """Both checkpoints hold the same tensors and objects, and bitwise-equal tensor data."""
    first, second = _checkpoint_layout(first_dir), _checkpoint_layout(second_dir)
    differ = sorted(k for k in first.keys() | second.keys() if first.get(k) != second.get(k))
    checks.check(not differ, f"checkpoint entries differ: {differ}")
    tensors = [
        {k: v for k, v in load_plain_tensors(d).items() if isinstance(v, torch.Tensor)}
        for d in (first_dir, second_dir)
    ]
    for key in sorted(tensors[0].keys() & tensors[1].keys()):
        checks.equal(f"checkpoint tensor {key}", tensors[1][key], tensors[0][key])


class TestMoonEPCheckpoint:
    def teardown_method(self, method):
        teardown_moonep()

    def test_sharded_keys_match_sequential(self):
        """The shards of SequentialMLP once the SwiGLU factories are built: keys, global shapes,
        offsets, fragmentation and replica ids, in bf16. Every expert entry holds the parameter
        itself, which the distributed optimizer maps its state by. Runs at EP=world/2 where
        possible, so the replica ids vary over expert-DP."""
        ep = _expert_dp_ep()
        _init(ep)
        checks = RankChecks()
        moon = _layer("moonep", ep, seed=1)
        base = _layer("sequential", ep, seed=1)
        experts = moon.experts
        holders = {}
        for sh in nested_values(moon.sharded_state_dict(prefix=PREFIX)):
            if isinstance(sh, (ShardedTensor, ShardedTensorFactory)):
                holders.setdefault(id(sh.data), []).append(sh)
        for name, param in experts.named_parameters():
            found = holders.get(id(param), [])
            ok = len(found) == 1 and found[0].data is param
            checks.check(ok, f"{name}: {len(found)} entries hold the parameter itself")
        moon_shards = _shards(moon.sharded_state_dict(prefix=PREFIX))
        base_shards = _shards(base.sharded_state_dict(prefix=PREFIX))
        only_moon = sorted(map(str, moon_shards - base_shards))
        only_base = sorted(map(str, base_shards - moon_shards))
        mismatch = f"only MoonEP: {only_moon}; only SequentialMLP: {only_base}"
        checks.check(not only_moon and not only_base, mismatch)
        # This rank's experts spelled out: fc1 as its gate and up halves, bf16, one replica per
        # expert-DP rank.
        epn, ep_rank = experts.num_local_experts, experts.ep_group.rank()
        replica = (0, 0, experts.expt_dp_group.rank())
        e, bf16 = NUM_EXPERTS, torch.bfloat16
        half = (bf16, (FFN, HIDDEN), (e, 2 * FFN, HIDDEN), (e, 2, 1), 1, replica)
        fc2 = (bf16, (HIDDEN, FFN), (e, HIDDEN, FFN), (e, 1, 1), 1, replica)
        for g in range(ep_rank * epn, (ep_rank + 1) * epn):
            expected = [
                (FC1_KEY, (g, 0, 0), half),  # gate rows
                (FC1_KEY, (g, FFN, 0), half),  # up rows
                (FC2_KEY, (g, 0, 0), fc2),
            ]
            expected += [(key, (g,), ((e,), replica)) for key in EXTRA_STATE_KEYS]
            for shard in expected:
                checks.check(moon_shards[shard] == 1, f"expert {g}: no shard {shard}")
        free_moonep_models()
        checks.assert_all()

    @pytest.mark.parametrize(
        "src, dst",
        [("moonep", "moonep"), ("sequential", "moonep"), ("moonep", "sequential")],
        ids=["moonep-to-moonep", "sequential-to-moonep", "moonep-to-sequential"],
    )
    def test_round_trip(self, src, dst, tmp_path_dist_ckpt):
        """Bitwise round trips between bf16 MoonEP and SequentialMLP layers: the loaded expert
        weights equal the saved ones, a re-save of the loaded layer reproduces the checkpoint,
        and the loaded layer computes what the saved one does (bit for bit between MoonEP
        layers: the next dispatch stages the loaded weights, not the ones it staged before)."""
        ep = Utils.world_size
        _init(ep)
        checks = RankChecks()
        name = f"moonep_round_trip_{src}_to_{dst}"
        with (
            TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / f"{name}_resaved", sync=True) as resaved_dir,
        ):
            source = _layer(src, ep, seed=1)
            target = _layer(dst, ep, seed=2)
            hidden = _hidden()
            _eval_forward(target, hidden)  # a MoonEP target stages its initial weights
            want = _expert_weights(source)
            initial = _expert_weights(target)[0][0]
            checks.check(not torch.equal(initial, want[0][0]), "seeds should differ")
            save(source.sharded_state_dict(prefix=PREFIX), ckpt_dir)
            result = _load_into(target, ckpt_dir)
            checks.check(not result.missing_keys and not result.unexpected_keys, f"{result}")
            for i, (got, expected) in enumerate(zip(_expert_weights(target), want)):
                checks.equal(f"expert {i} fc1", got[0], expected[0])
                checks.equal(f"expert {i} fc2", got[1], expected[1])
            checks.equal("router", target.router.weight, source.router.weight)
            save(target.sharded_state_dict(prefix=PREFIX), resaved_dir)
            _check_same_checkpoint(checks, ckpt_dir, resaved_dir)
            output, want_output = _eval_forward(target, hidden), _eval_forward(source, hidden)
            if src == dst:
                checks.equal("output", output, want_output)
            else:
                checks.close("output", output, want_output, TOL)
        free_moonep_models()
        checks.assert_all()

    def test_expert_parallel_reshard(self, tmp_path_dist_ckpt):
        """Save at EP=world, load at EP=world/2 (two expert-DP replicas): every global expert
        lands on the ranks that own it at the new EP size, and the layer computes what it did."""
        src_ep = Utils.world_size
        dst_ep = src_ep // 2
        if dst_ep < 2 or NUM_EXPERTS % dst_ep:
            pytest.skip("needs at least 4 GPUs")
        _init(src_ep)
        checks = RankChecks()
        with TempNamedDir(tmp_path_dist_ckpt / "moonep_ep_reshard", sync=True) as ckpt_dir:
            source = _layer("moonep", src_ep, seed=7)
            all_fc1, all_fc2 = _global_experts(source)
            router = source.router.weight.detach().clone()
            hidden = _hidden()
            want_output = _eval_forward(source, hidden)
            save(source.sharded_state_dict(prefix=PREFIX), ckpt_dir)
            del source
            teardown_moonep()

            _init(dst_ep)
            target = _layer("moonep", dst_ep, seed=8)
            experts = target.experts
            first = experts.ep_group.rank() * experts.num_local_experts
            initial = _expert_weights(target)[0][0]
            checks.check(not torch.equal(initial, all_fc1[first]), "seeds should differ")
            result = _load_into(target, ckpt_dir)
            checks.check(not result.missing_keys and not result.unexpected_keys, f"{result}")
            for i, (fc1, fc2) in enumerate(_expert_weights(target)):
                checks.equal(f"expert {first + i} fc1", fc1, all_fc1[first + i])
                checks.equal(f"expert {first + i} fc2", fc2, all_fc2[first + i])
            checks.equal("router", target.router.weight, router)
            checks.close("output", _eval_forward(target, hidden), want_output, TOL)
        free_moonep_models()
        checks.assert_all()


def _ckpt_metadata(sharding_type):
    """The sharded_state_dict metadata megatron/training/checkpointing.py passes to the model and
    the optimizer of torch_dist checkpoints with the distributed optimizer:
    ``--dist-ckpt-optim-fully-reshardable`` selects fully_reshardable, else dp_reshardable."""
    args = Namespace(
        use_distributed_optimizer=True,
        ckpt_format="torch_dist",
        dist_ckpt_optim_fully_reshardable=sharding_type == "fully_reshardable",
        distrib_optim_fully_reshardable_mem_efficient=False,
    )
    return _build_sharded_state_dict_metadata(args)


def _gpt(ep, seed):
    """GPTModel (every layer MoE on MoonEP) -> Float16Module -> DDP with the distributed
    optimizer, as megatron/training builds it. Returns ``(model, gpt, optimizer)``."""
    config = TransformerConfig(
        **moe_config_kwargs(
            ep,
            num_layers=NUM_LAYERS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="moonep",
            moe_moonep_max_tokens_per_rank=MAX_TOKENS,
        )
    )
    _set_random_seed(seed)
    gpt = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(num_experts=NUM_EXPERTS),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
    ).cuda()
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, grad_reduce_in_fp32=True, check_for_nan_in_grad=True
    )
    model = DDP(config, ddp_config, Float16Module(config, gpt))
    config.finalize_model_grads_func = finalize_model_grads
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.01,
            clip_grad=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
        ),
        [model],
    )
    config.grad_scale_func = optimizer.scale_loss
    return model, gpt, optimizer


def _train_step(model, optimizer, seed):
    """One iteration in megatron/training/training.py:train_step order. Returns this rank's
    ``[loss sum, tokens]``, the grad norm and whether the step succeeded."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + dist.get_rank())
    tokens = torch.randint(0, VOCAB, (MBS, SEQ), device="cuda", generator=gen)
    labels = torch.roll(tokens, shifts=-1, dims=1)
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(MBS, -1)
    mask = torch.triu(torch.ones(SEQ, SEQ, device="cuda", dtype=torch.bool), diagonal=1)

    def forward_step(data_iterator, chunk):
        output = chunk(tokens, position_ids, mask[None, None], labels=labels)

        def loss_func(output_tensor):
            loss = output_tensor.float().sum()
            num_tokens = torch.tensor(output_tensor.numel(), device="cuda", dtype=torch.int)
            reported = torch.stack([loss.detach(), num_tokens.float()])
            return loss, num_tokens, {"lm loss": reported}

        return output, loss_func

    model.zero_grad_buffer()
    optimizer.zero_grad()
    losses = forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=None,
        model=model,
        num_microbatches=1,
        seq_length=SEQ,
        micro_batch_size=MBS,
    )
    success, grad_norm, _ = optimizer.step()
    return losses[0]["lm loss"], grad_norm, success


def _expert_optimizers(gpt, optimizer):
    """The distributed optimizers of the chain whose buffers hold MoonEP expert weights."""
    moonep = {id(p) for experts in iter_moonep_experts(gpt) for p in experts.parameters()}
    return [
        dist_opt
        for dist_opt in optimizer.chained_optimizers
        if any(id(p) in moonep for buffer in dist_opt.buffers for p in buffer.param_index_map)
    ]


def _optimizer_shards(gpt, optimizer):
    """This rank's shards of the fp32 main weights and optimizer states (``param``,
    ``exp_avg``, ...), by (index of the distributed optimizer in the chain, parameter name)."""
    names = {id(param): name for name, param in gpt.named_parameters()}
    return {
        (index, names[id(param)]): dist_opt._get_main_param_and_optimizer_states(param)
        for index, dist_opt in enumerate(optimizer.chained_optimizers)
        for param in dist_opt.model_param_group_index_map
    }


def _group_steps(optimizer):
    """Step count of every param group of the chain (TE's FusedAdam counts per group)."""
    return [
        [None if g.get("step") is None else float(g["step"]) for g in o.optimizer.param_groups]
        for o in optimizer.chained_optimizers
    ]


def _check_same_training_state(checks, what, a, b):
    """``a`` and ``b`` are ``(gpt, optimizer)``: every parameter, and this rank's shards of the
    main weights and optimizer states of every distributed optimizer, bit for bit."""
    (gpt_a, optim_a), (gpt_b, optim_b) = a, b
    params_b = dict(gpt_b.named_parameters())
    for name, param in gpt_a.named_parameters():
        checks.equal(f"{what}: {name}", params_b[name], param)
    shards_a, shards_b = _optimizer_shards(gpt_a, optim_a), _optimizer_shards(gpt_b, optim_b)
    checks.check(shards_a.keys() == shards_b.keys(), f"{what}: different optimizer shards")
    for key in shards_a.keys() & shards_b.keys():
        states_a, states_b = shards_a[key], shards_b[key]
        checks.check(states_a.keys() == states_b.keys(), f"{what}: {key} has {sorted(states_b)}")
        for state in states_a.keys() & states_b.keys():
            checks.equal(f"{what}: {key} {state}", states_b[state], states_a[state])
    moonep = {id(p) for experts in iter_moonep_experts(gpt_a) for p in experts.parameters()}
    moonep_names = {name for name, param in gpt_a.named_parameters() if id(param) in moonep}
    compared = sum(name in moonep_names for _, name in shards_a)
    checks.check(compared > 0, f"{what}: no MoonEP expert weight has optimizer state here")
    steps_a, steps_b = _group_steps(optim_a), _group_steps(optim_b)
    checks.check(steps_a == steps_b, f"{what}: param-group steps {steps_b} vs {steps_a}")


def _check_gpt_expert_entries(checks, ckpt_dir, sharding_type):
    """The checkpoint stores the MoonEP weights like SequentialMLP's, bf16 behind a layer and an
    expert axis; in fully_reshardable format their optimizer states follow the same layout."""
    tensors = load_tensors_metadata(ckpt_dir)
    states = ("param", "exp_avg", "exp_avg_sq") if sharding_type == "fully_reshardable" else ()
    for linear, shape in (("linear_fc1", (2 * FFN, HIDDEN)), ("linear_fc2", (HIDDEN, FFN))):
        key = f"decoder.layers.mlp.experts.experts.{linear}.weight"
        global_shape = (NUM_LAYERS, NUM_EXPERTS, *shape)
        entries = [(key, torch.bfloat16)]
        entries += [(f"optimizer.state.{state}.{key}", torch.float32) for state in states]
        for entry, dtype in entries:
            sh = tensors.get(entry)
            ok = sh is not None and sh.global_shape == global_shape and sh.dtype == dtype
            checks.check(ok, f"checkpoint entry {entry}: {sh}")


class TestMoonEPOptimizerCheckpoint:
    def teardown_method(self, method):
        destroy_moe_metrics_tracker()
        teardown_moonep()

    @pytest.mark.parametrize("sharding_type", ["dp_reshardable", "fully_reshardable"])
    def test_model_and_optimizer_round_trip(self, sharding_type, tmp_path_dist_ckpt):
        """A GPT on MoonEP with expert-DP > 1 (EP=4 on 8 GPUs, so the expert optimizer state is
        sharded) under DDP and the distributed optimizer, saved after one step: the model in
        torch_dist, the optimizer in ``sharding_type`` (dp_reshardable is the training default,
        fully_reshardable is --dist-ckpt-optim-fully-reshardable). A fresh model and optimizer
        load it with the weights, fp32 main weights and Adam moments of the saved ones, and
        both copies take the next step bit for bit identically."""
        ep = _expert_dp_ep()
        _init(ep)
        checks = RankChecks()
        metadata = _ckpt_metadata(sharding_type)
        checks.check(metadata["distrib_optim_sharding_type"] == sharding_type, f"{metadata}")
        ckpt_name = f"moonep_optimizer_{sharding_type}"
        with TempNamedDir(tmp_path_dist_ckpt / ckpt_name, sync=True) as ckpt_dir:
            model_a, gpt_a, optim_a = _gpt(ep, seed=11)
            layers = len(list(iter_moonep_experts(gpt_a)))
            checks.check(layers == NUM_LAYERS, f"{layers} MoonEP layers")
            sizes = [o.data_parallel_group.size() for o in _expert_optimizers(gpt_a, optim_a)]
            checks.check(sizes == [Utils.world_size // ep], f"expert optimizer DP sizes {sizes}")
            checks.check(_train_step(model_a, optim_a, seed=1)[2], "step 1 failed")
            model_sd = gpt_a.sharded_state_dict(metadata=metadata)
            optim_sd = optim_a.sharded_state_dict(model_sd, metadata=metadata)
            save({"model": model_sd, "optimizer": optim_sd}, ckpt_dir)
            _check_gpt_expert_entries(checks, ckpt_dir, sharding_type)

            model_b, gpt_b, optim_b = _gpt(ep, seed=12)
            model_sd = gpt_b.sharded_state_dict(metadata=metadata)
            optim_sd = optim_b.sharded_state_dict(model_sd, is_loading=True, metadata=metadata)
            loaded = load({"model": model_sd, "optimizer": optim_sd}, ckpt_dir)
            result = gpt_b.load_state_dict(loaded["model"], strict=False)
            checks.check(not result.missing_keys and not result.unexpected_keys, f"{result}")
            optim_b.load_state_dict(loaded["optimizer"])
            a, b = (gpt_a, optim_a), (gpt_b, optim_b)
            _check_same_training_state(checks, "after load", a, b)

            # Both copies keep training identically.
            loss_a, norm_a, ok_a = _train_step(model_a, optim_a, seed=2)
            loss_b, norm_b, ok_b = _train_step(model_b, optim_b, seed=2)
            checks.check(ok_a and ok_b, "step 2 failed")
            checks.equal("step 2 loss", loss_b, loss_a)
            checks.check(norm_b == norm_a, f"step 2 grad norm {norm_b} vs {norm_a}")
            _check_same_training_state(checks, "after one more step", a, b)
        free_moonep_models()
        checks.assert_all()
