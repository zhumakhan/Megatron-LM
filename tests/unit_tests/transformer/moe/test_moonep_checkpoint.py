# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP backend: distributed checkpoints interchangeable with SequentialMLP.

Run with ``torch.distributed.run --nproc-per-node <GPUs of one NVLink domain>``, without ``-x``:
checks are gathered with ``RankChecks`` so every rank fails together.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import load, save
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

SEQ, MBS, VOCAB = 64, 2, 1024
MAX_TOKENS = SEQ * MBS
PREFIX = "decoder.layers.0.mlp."
PROJECTIONS = ("gate", "up", "down")


def _init(ep):
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)
    _set_random_seed(123)


def _layer(backend, ep, seed):
    _set_random_seed(seed)
    if backend == "moonep":
        return build_moe_layer(moonep_config(ep, MAX_TOKENS))
    return build_moe_layer(alltoall_config(ep))


def _load_into(layer, ckpt_dir):
    loaded = load(layer.sharded_state_dict(prefix=PREFIX), ckpt_dir)
    return layer.load_state_dict({k.removeprefix(PREFIX): v for k, v in loaded.items()})


def _masters(layer):
    experts = layer.experts
    return [getattr(experts, f"weight_{n}").detach().clone() for n in PROJECTIONS]


def _check_pools(checks, layer, where):
    experts, storage = layer.experts, layer.experts.storage
    for name in PROJECTIONS:
        checks.equal(
            f"{where}: {name} pool == master.bf16()",
            storage.w[name][storage.lo : storage.hi],
            getattr(experts, f"weight_{name}").bfloat16(),
        )
    problems = experts.check_invariants()
    checks.check(problems == [], f"{where}: {problems}")


class TestMoonEPCheckpoint:
    def teardown_method(self, method):
        teardown_moonep()

    def test_sharded_keys_match_sequential(self):
        """Same keys, global shapes and fragment layout as SequentialMLP."""
        ep = Utils.world_size
        _init(ep)
        checks = RankChecks()
        moon = _layer("moonep", ep, seed=1).sharded_state_dict(prefix=PREFIX)
        fc1 = f"{PREFIX}experts.experts.linear_fc1.weight"
        fc2 = f"{PREFIX}experts.experts.linear_fc2.weight"
        gate, up, down = (moon[f"{PREFIX}experts.weight_{n}"] for n in PROJECTIONS)
        checks.check(gate.key == fc1 and up.key == fc1 and down.key == fc2, "sharded keys")
        checks.check(gate.global_shape == (NUM_EXPERTS, 2 * FFN, HIDDEN), f"{gate.global_shape}")
        checks.check(down.global_shape == (NUM_EXPERTS, HIDDEN, FFN), f"{down.global_shape}")
        epn, ep_rank = NUM_EXPERTS // ep, dist.get_rank()
        checks.check(gate.global_offset == (ep_rank * epn, 0, 0), f"gate {gate.global_offset}")
        checks.check(up.global_offset == (ep_rank * epn, FFN, 0), f"up {up.global_offset}")
        checks.check(gate.data.dtype == torch.float32, "the masters are what is saved")
        base_keys = {
            sh.key
            for sh in _layer("alltoall", ep, seed=1).sharded_state_dict(prefix=PREFIX).values()
        }
        moon_keys = {sh.key for sh in moon.values()}
        checks.check(
            moon_keys == base_keys,
            f"only MoonEP: {sorted(moon_keys - base_keys)}, "
            f"only SequentialMLP: {sorted(base_keys - moon_keys)}",
        )
        free_moonep_models()
        checks.assert_all()

    def test_moonep_round_trip(self, tmp_path_dist_ckpt):
        ep = Utils.world_size
        _init(ep)
        checks = RankChecks()
        with TempNamedDir(tmp_path_dist_ckpt / "moonep_round_trip", sync=True) as ckpt_dir:
            source = _layer("moonep", ep, seed=1)
            save(source.sharded_state_dict(prefix=PREFIX), ckpt_dir)
            target = _layer("moonep", ep, seed=2)
            checks.check(
                not torch.equal(_masters(source)[0], _masters(target)[0]), "seeds should differ"
            )
            result = _load_into(target, ckpt_dir)
            checks.check(not result.missing_keys and not result.unexpected_keys, f"{result}")
            for name, got, want in zip(PROJECTIONS, _masters(target), _masters(source)):
                checks.equal(f"master {name}", got, want)
            _check_pools(checks, target, "after load")
            checks.equal("router", target.router.weight, source.router.weight)
        free_moonep_models()
        checks.assert_all()

    def test_sequential_to_moonep(self, tmp_path_dist_ckpt):
        """bf16 SequentialMLP weights load into fp32 masters exactly."""
        ep = Utils.world_size
        _init(ep)
        checks = RankChecks()
        with TempNamedDir(tmp_path_dist_ckpt / "sequential_to_moonep", sync=True) as ckpt_dir:
            source = _layer("alltoall", ep, seed=3)
            save(source.sharded_state_dict(prefix=PREFIX), ckpt_dir)
            target = _layer("moonep", ep, seed=4)
            _load_into(target, ckpt_dir)
            experts = target.experts
            for i, expert in enumerate(source.experts.local_experts):
                fc1 = expert.linear_fc1.weight.float()
                checks.equal(f"gate[{i}]", experts.weight_gate[i], fc1[:FFN])
                checks.equal(f"up[{i}]", experts.weight_up[i], fc1[FFN:])
                checks.equal(f"down[{i}]", experts.weight_down[i], expert.linear_fc2.weight.float())
            _check_pools(checks, target, "after load")
        free_moonep_models()
        checks.assert_all()

    def test_moonep_to_sequential(self, tmp_path_dist_ckpt):
        """fp32 masters load into bf16 SequentialMLP weights as master.bf16()."""
        ep = Utils.world_size
        _init(ep)
        checks = RankChecks()
        with TempNamedDir(tmp_path_dist_ckpt / "moonep_to_sequential", sync=True) as ckpt_dir:
            source = _layer("moonep", ep, seed=5)
            # Give the masters sub-bf16 precision so the rounding on load is exercised.
            with torch.no_grad():
                for master in source.experts.moonep_master_parameters():
                    master.add_(torch.randn_like(master) * 1e-4)
                source.experts.sync_compute_weights()
            save(source.sharded_state_dict(prefix=PREFIX), ckpt_dir)
            target = _layer("alltoall", ep, seed=6)
            result = _load_into(target, ckpt_dir)
            checks.check(not result.missing_keys and not result.unexpected_keys, f"{result}")
            experts = source.experts
            for i, expert in enumerate(target.experts.local_experts):
                fc1 = torch.cat((experts.weight_gate[i], experts.weight_up[i])).bfloat16()
                checks.equal(f"fc1[{i}]", expert.linear_fc1.weight, fc1)
                checks.equal(
                    f"fc2[{i}]", expert.linear_fc2.weight, experts.weight_down[i].bfloat16()
                )
        free_moonep_models()
        checks.assert_all()

    def test_expert_parallel_reshard(self, tmp_path_dist_ckpt):
        """Save with EP=world, load with EP=world/2 (two expert-DP replicas)."""
        src_ep = Utils.world_size
        dst_ep = src_ep // 2
        if dst_ep < 2 or NUM_EXPERTS % dst_ep:
            pytest.skip("needs at least 4 GPUs")
        checks = RankChecks()
        with TempNamedDir(tmp_path_dist_ckpt / "moonep_ep_reshard", sync=True) as ckpt_dir:
            _init(src_ep)
            source = _layer("moonep", src_ep, seed=7)
            full = []
            for master in source.experts.moonep_master_parameters():
                parts = [torch.empty_like(master) for _ in range(src_ep)]
                dist.all_gather(parts, master.detach(), group=source.experts.ep_group)
                full.append(torch.cat(parts))
            save(source.sharded_state_dict(prefix=PREFIX), ckpt_dir)
            del source
            teardown_moonep()

            _init(dst_ep)
            target = _layer("moonep", dst_ep, seed=8)
            _load_into(target, ckpt_dir)
            storage = target.experts.storage
            for name, got, want in zip(PROJECTIONS, _masters(target), full):
                checks.equal(f"master {name}", got, want[storage.lo : storage.hi])
            _check_pools(checks, target, "after resharded load")
        free_moonep_models()
        checks.assert_all()


def _gpt(ep, seed):
    config = TransformerConfig(
        **moe_config_kwargs(
            ep,
            num_layers=1,
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
    model.zero_moe_grads()
    forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=None,
        model=model,
        num_microbatches=1,
        seq_length=SEQ,
        micro_batch_size=MBS,
    )
    success, _, _ = optimizer.step()
    return success


class TestMoonEPOptimizerCheckpoint:
    def teardown_method(self, method):
        destroy_moe_metrics_tracker()
        teardown_moonep()

    def test_model_and_optimizer_round_trip(self, tmp_path_dist_ckpt):
        """Masters, pools and MoonEPOptimizer Adam states survive save/load."""
        ep = Utils.world_size
        _init(ep)
        checks = RankChecks()
        metadata = {"distrib_optim_sharding_type": "dp_reshardable"}
        with TempNamedDir(tmp_path_dist_ckpt / "moonep_optimizer", sync=True) as ckpt_dir:
            model_a, gpt_a, optim_a = _gpt(ep, seed=11)
            checks.check(_train_step(model_a, optim_a, seed=1), "step A failed")
            model_sd = gpt_a.sharded_state_dict()
            save(
                {
                    "model": model_sd,
                    "optimizer": optim_a.sharded_state_dict(model_sd, metadata=metadata),
                },
                ckpt_dir,
            )

            model_b, gpt_b, optim_b = _gpt(ep, seed=12)
            model_sd = gpt_b.sharded_state_dict()
            loaded = load(
                {
                    "model": model_sd,
                    "optimizer": optim_b.sharded_state_dict(
                        model_sd, is_loading=True, metadata=metadata
                    ),
                },
                ckpt_dir,
            )
            result = gpt_b.load_state_dict(loaded["model"])
            checks.check(not result.missing_keys, f"missing keys {result.missing_keys}")
            optim_b.load_state_dict(loaded["optimizer"])

            moon_a = optim_a.chained_optimizers[-1].optimizer
            moon_b = optim_b.chained_optimizers[-1].optimizer
            layers_a = list(iter_moonep_experts(gpt_a))
            layers_b = list(iter_moonep_experts(gpt_b))
            checks.check(len(layers_a) == len(layers_b) == 1, "one MoonEP layer each")
            for ex_a, ex_b in zip(layers_a, layers_b):
                pairs = zip(ex_a.moonep_master_parameters(), ex_b.moonep_master_parameters())
                for pa, pb in pairs:
                    checks.equal("master", pb, pa)
                    for key in ("exp_avg", "exp_avg_sq"):
                        checks.equal(key, moon_b.state[pb][key], moon_a.state[pa][key])
                    if "step" in moon_a.state[pa]:  # torch Adam; TE FusedAdam steps per group
                        checks.check(
                            float(moon_b.state[pb]["step"]) == float(moon_a.state[pa]["step"]),
                            "per-parameter step",
                        )
                storage = ex_b.storage
                for name in PROJECTIONS:
                    checks.equal(
                        f"{name} pool",
                        storage.w[name][storage.lo : storage.hi],
                        getattr(ex_b, f"weight_{name}").bfloat16(),
                    )
            steps_a = [g.get("step") for g in moon_a.param_groups]
            steps_b = [g.get("step") for g in moon_b.param_groups]
            checks.check(steps_a == steps_b, f"param-group steps {steps_a} vs {steps_b}")
            # Both copies keep training identically.
            checks.check(_train_step(model_a, optim_a, seed=2), "step A2 failed")
            checks.check(_train_step(model_b, optim_b, seed=2), "step B2 failed")
            for ex_a, ex_b in zip(layers_a, layers_b):
                pairs = zip(ex_a.moonep_master_parameters(), ex_b.moonep_master_parameters())
                for pa, pb in pairs:
                    checks.equal("master after one more step", pb, pa)
        free_moonep_models()
        checks.assert_all()
