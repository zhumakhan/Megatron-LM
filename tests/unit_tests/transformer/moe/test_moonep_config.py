# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static validation of the MoonEP flex dispatcher backend (CPU only: no GPU, no process group).

Model-level rules: ``TransformerConfig._validate_moonep`` at config construction. Training-level
rules: the MoonEP block of ``megatron/training/arguments.py:validate_args``, which admits the
data-parallel wrappers that keep the expert weights whole -- Megatron DDP (optionally with the
distributed optimizer, ZeRO-1) and Megatron-FSDP v1 with ``optim`` (ZeRO-1) or ``optim_grads``
(ZeRO-2) -- and rejects everything that shards parameters.
"""

import importlib.util
import re
import sys

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.moonep_backend import uses_moonep
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import arguments

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(importlib.util.find_spec("moonep") is None, reason="moonep not installed"),
]

PREFIX = "moe_flex_dispatcher_backend='moonep' requires "
BLOCK_RECOMPUTE = dict(
    recompute_granularity="full", recompute_method="block", recompute_num_layers=2
)


def moonep_config_kwargs(**overrides):
    """A valid phase-1 MoonEP config (EP=8, TP=1, bf16, SwiGLU, no bias, fp32 router)."""
    kwargs = dict(
        num_layers=2,
        hidden_size=1024,
        num_attention_heads=8,
        num_moe_experts=16,
        moe_ffn_hidden_size=1024,
        moe_router_topk=4,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="moonep",
        bf16=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        moe_router_dtype="fp32",
        moe_moonep_max_tokens_per_rank=512,
        use_cpu_initialization=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_config():
    config = TransformerConfig(**moonep_config_kwargs())
    assert uses_moonep(config)
    assert config.moe_moonep_token_padding == 128


def test_other_backends_unaffected():
    config = TransformerConfig(
        **moonep_config_kwargs(
            moe_flex_dispatcher_backend="deepep", add_bias_linear=True, moe_router_dtype=None
        )
    )
    assert not uses_moonep(config)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(moe_token_dispatcher_type="alltoall"), "moe_token_dispatcher_type='flex'"),
        # The generic rules reject num_moe_experts=None with EP > 1 or with moe_ffn_hidden_size set.
        (
            dict(num_moe_experts=None, moe_ffn_hidden_size=None, expert_model_parallel_size=1),
            "num_moe_experts to be set",
        ),
        (dict(expert_model_parallel_size=1), "1 < expert_model_parallel_size <= 128"),
        (
            dict(expert_model_parallel_size=129, num_moe_experts=129),
            r"1 < expert_model_parallel_size <= 128 .*, got 129\.",
        ),
        (dict(num_moe_experts=12), "num_moe_experts divisible by expert_model_parallel_size"),
        (dict(expert_tensor_parallel_size=2), "expert_tensor_parallel_size == 1"),
        (
            dict(expert_tensor_parallel_num_weight_shards=2),
            "gtp_weight_remat_size == expert_gtp_weight_remat_size == 1",
        ),
        (
            dict(tensor_parallel_num_weight_shards=2),
            r"gtp_weight_remat_size == expert_gtp_weight_remat_size == 1 .*, got 2 and 1\.",
        ),
        (
            dict(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16),
            "pipeline_model_parallel_size == 1",
        ),
        (dict(virtual_pipeline_model_parallel_size=2), "virtual_pipeline_model_parallel_size None"),
        (dict(bf16=False, params_dtype=torch.float32), "bf16=True and fp16=False"),
        (dict(bf16=False, fp16=True, params_dtype=torch.float16), "bf16=True and fp16=False"),
        (dict(gated_linear_unit=False), "SwiGLU experts"),
        (dict(activation_func=F.gelu), "SwiGLU experts"),
        (dict(glu_linear_offset=1.0), "a plain SwiGLU"),
        (dict(activation_func_clamp_value=7.0), "a plain SwiGLU"),
        (
            dict(activation_func_tanh_clamp_scale=7.0),
            r"a plain SwiGLU .*activation_func_tanh_clamp_scale=7\.0",
        ),
        (dict(moe_mlp_glu_interleave_size=32), r"a plain SwiGLU .*moe_mlp_glu_interleave_size=32"),
        (dict(add_bias_linear=True), "add_bias_linear=False"),
        (dict(moe_expert_capacity_factor=1.0), "dropless routing"),
        (
            dict(moe_expert_capacity_factor=1.0, moe_pad_expert_input_to_capacity=True),
            "dropless routing",
        ),
        # Quantization padding needs fp8 or fp4 (a generic rule); the dropless rule runs first.
        (
            dict(moe_router_padding_for_quantization=True, fp8="e4m3"),
            r"dropless routing .*moe_router_padding_for_quantization=True",
        ),
        (
            dict(moe_pad_experts_for_cuda_graph_inference=True),
            r"dropless routing .*moe_pad_experts_for_cuda_graph_inference=True",
        ),
        (dict(moe_router_dtype=None), "moe_router_dtype='fp32'"),
        (dict(fp8="e4m3"), r"bf16 expert compute \(no fp8/fp4\)"),
        (dict(fp4="nvfp4"), r"bf16 expert compute \(no fp8/fp4\), got fp8=None, fp4=nvfp4"),
        (dict(moe_latent_size=256), "no MoE latent projections"),
        (
            dict(moe_apply_probs_on_input=True),
            "no MoE latent projections, .*moe_apply_probs_on_input=True",
        ),
        (
            dict(moe_num_hash_layers=1, hash_moe_vocab_size=1024),
            r"no MoE latent projections, .*moe_num_hash_layers=1\.",
        ),
        (
            dict(moe_shared_expert_intermediate_size=1024, moe_shared_expert_overlap=True),
            "moe_shared_expert_overlap=False",
        ),
        (dict(overlap_moe_expert_parallel_comm=True), "no EP communication overlap"),
        (
            dict(overlap_dispatch_backward_with_experts_wgrad=True),
            "no EP communication overlap .*overlap_dispatch_backward_with_experts_wgrad=True",
        ),
        (dict(cuda_graph_impl="local"), "cuda_graph_impl='none'"),
        (
            dict(cuda_graph_impl="transformer_engine"),
            "cuda_graph_impl='none' .*, got 'transformer_engine'",
        ),
        (dict(cuda_graph_impl="full_iteration"), "cuda_graph_impl='none' .*, got 'full_iteration'"),
        # The deprecated flags set cuda_graph_impl 'local' and 'transformer_engine'.
        (dict(enable_cuda_graph=True), "cuda_graph_impl='none' .*, got 'local'"),
        (dict(external_cuda_graph=True), "cuda_graph_impl='none' .*, got 'transformer_engine'"),
        (
            dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1),
            "recompute_method='block' with full recompute",
        ),
        (
            dict(recompute_granularity="selective", recompute_modules=["moe"]),
            "no selective 'moe' or 'moe_act' recompute",
        ),
        (dict(moe_layer_recompute=True), "no selective 'moe' or 'moe_act' recompute"),
        (
            dict(
                recompute_granularity="selective",
                recompute_modules=["moe_act"],
                moe_grouped_gemm=True,
            ),
            "no selective 'moe' or 'moe_act' recompute",
        ),
        (dict(variable_seq_lengths=True), "the same token count on every EP rank"),
        (
            dict(hybrid_context_parallel=True),
            "the same token count on every EP rank .*hybrid_context_parallel=True",
        ),
        (dict(mlp_chunks_for_training=2), "one call per MoE layer per micro-batch"),
        (dict(mtp_num_layers=1, mtp_use_repeated_layer=True), "one call per MoE layer"),
        (
            dict(fine_grained_activation_offloading=True, offload_modules=["expert_fc1"]),
            "no fine-grained activation offloading",
        ),
        (
            dict(cpu_offloading=True, cpu_offloading_num_layers=1),
            "cpu_offloading only together with full recompute and recompute_method='block'",
        ),
        (dict(init_model_with_meta_device=True), "GPU initialization at construction"),
        (dict(use_cpu_initialization=True), "GPU initialization at construction"),
        # The inference-optimized layers need RMSNorm (a generic rule).
        (
            dict(transformer_impl="inference_optimized", normalization="RMSNorm"),
            "GPU initialization at construction .*transformer_impl='inference_optimized'",
        ),
        (dict(hidden_size=1000), "hidden_size and moe_ffn_hidden_size to be multiples of 128"),
        (dict(moe_ffn_hidden_size=1000), "hidden_size and moe_ffn_hidden_size to be multiples"),
        (dict(num_moe_experts=64, moe_router_topk=33), r"1 <= moe_router_topk <= 32"),
        (dict(moe_moonep_max_tokens_per_rank=0), "moe_moonep_max_tokens_per_rank None or > 0"),
        (dict(moe_moonep_token_padding=0), "moe_moonep_token_padding >= 1"),
    ],
)
def test_forbidden_overrides(overrides, match):
    with pytest.raises(ValueError, match=PREFIX + match):
        TransformerConfig(**moonep_config_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(BLOCK_RECOMPUTE, id="block-recompute"),
        pytest.param(
            dict(BLOCK_RECOMPUTE, cpu_offloading=True, cpu_offloading_num_layers=1),
            id="block-recompute-offload",
        ),
        pytest.param(
            dict(recompute_granularity="selective", recompute_modules=["core_attn"]),
            id="selective-core_attn",
        ),
        pytest.param(
            dict(recompute_granularity="selective", recompute_modules=["layernorm"]),
            id="selective-layernorm",
        ),
        pytest.param(
            dict(recompute_granularity="selective", recompute_modules=["mlp"]), id="selective-mlp"
        ),
        pytest.param(
            dict(
                recompute_granularity="selective",
                recompute_modules=["shared_experts"],
                moe_shared_expert_intermediate_size=1024,
            ),
            id="selective-shared_experts",
        ),
        pytest.param(
            dict(mtp_num_layers=1, freeze_base_model_for_mtp=True), id="freeze-base-model-for-mtp"
        ),
        pytest.param(
            dict(tensor_model_parallel_size=2, sequence_parallel=True),
            id="tp2-sequence-parallel-etp1",
        ),
        pytest.param(
            dict(moe_shared_expert_intermediate_size=1024, moe_shared_expert_overlap=False),
            id="shared-experts-no-overlap",
        ),
    ],
)
def test_allowed_overrides(overrides):
    """Full block-wise recompute, also with CPU offloading: the recompute replays the
    checkpointed forward's plan instead of pushing the experts again. Selective recompute outside
    the experts (attention, layernorms, dense MLPs, shared experts). freeze_base_model_for_mtp:
    the expert weights are ordinary parameters, frozen or trained per layer like any other. TP > 1
    with sequence parallelism and whole experts (ETP=1 from the base config), and shared experts
    without the side-stream overlap."""
    assert uses_moonep(TransformerConfig(**moonep_config_kwargs(**overrides)))


def test_moe_enable_deepep_conflict():
    """The deprecated flag would otherwise silently rewrite the backend to 'deepep'."""
    with pytest.raises(ValueError, match="moe_enable_deepep forces the 'deepep' flex backend"):
        TransformerConfig(**moonep_config_kwargs(moe_enable_deepep=True))


def test_missing_package(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "moonep":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(ValueError, match=PREFIX + "the moonep package"):
        TransformerConfig(**moonep_config_kwargs())


# ---- validate_args: the data-parallel wrappers MoonEP trains with ----------------------------

# A MoonEP training command line whose model also passes TransformerConfig._validate_moonep
# (EP=8, TP=1, bf16, SwiGLU, no bias, fp32 router); the cells add their flags.
MOONEP_TRAINING_ARGS = (
    "--num-layers 2 --hidden-size 1024 --num-attention-heads 8 --seq-length 512 "
    "--max-position-embeddings 512 --micro-batch-size 1 --num-experts 16 "
    "--moe-ffn-hidden-size 1024 --moe-router-topk 4 --expert-model-parallel-size 8 "
    "--expert-tensor-parallel-size 1 --moe-token-dispatcher-type flex "
    "--moe-flex-dispatcher-backend moonep --moe-router-dtype fp32 --bf16 --swiglu "
    "--disable-bias-linear"
).split()
# Megatron-FSDP requires the fsdp_dtensor checkpoint format.
MFSDP = "--use-megatron-fsdp --ckpt-format fsdp_dtensor"
MFSDP_ZERO2 = MFSDP + " --data-parallel-sharding-strategy optim_grads"
MFSDP_ERROR = "MoonEP with Megatron-FSDP "
HYBRID_ERROR = MFSDP_ERROR + "does not support hybrid sharding "
DDP_CKPT_ERROR = "MoonEP with Megatron DDP supports only --ckpt-format torch_dist"


def _validate_moonep_args(monkeypatch, flags, **env):
    """Run ``validate_args`` on the MoonEP training command line plus ``flags`` as a rank of an
    8-GPU job, with the extra environment variables ``env``."""
    # Megatron-FSDP and torch FSDP2 reject CUDA_DEVICE_MAX_CONNECTIONS=1, which other tests may
    # leave set in this process.
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    # parse_args takes the world size from the launcher's environment.
    monkeypatch.setenv("WORLD_SIZE", "8")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    # validate_args reads the GPU architecture to pick Megatron-FSDP's high-priority
    # communication streams and, with TP or CP > 1, to require CUDA_DEVICE_MAX_CONNECTIONS=1
    # before Blackwell; pin Hopper so these cells run without a GPU.
    monkeypatch.setattr(arguments, "get_device_arch_version", lambda: 9)
    argv = ["test_moonep_config.py", *MOONEP_TRAINING_ARGS, *flags.split()]
    monkeypatch.setattr(sys, "argv", argv)
    return arguments.validate_args(arguments.parse_args())


def _strategies_error(*got):
    """The Megatron-FSDP sharding-strategy rule's message for the sorted strategies ``got``."""
    return (
        MFSDP_ERROR + r"supports the 'optim' \(ZeRO-1\) and 'optim_grads' \(ZeRO-2\) sharding "
        r"strategies only.*, got " + re.escape(str(list(got)))
    )


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(
            "--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather "
            "--ckpt-format torch_dist",
            id="ddp-distopt-overlap",
        ),
        pytest.param(MFSDP + " --data-parallel-sharding-strategy optim", id="mfsdp-optim"),
        pytest.param(MFSDP_ZERO2, id="mfsdp-optim_grads"),
        pytest.param(
            MFSDP_ZERO2 + " --expert-data-parallel-sharding-strategy optim",
            id="mfsdp-optim_grads-expert-optim",
        ),
    ],
)
def test_validate_args_accepts_zero1_and_zero2(monkeypatch, flags):
    """ZeRO-1 and ZeRO-2: Megatron DDP with the distributed optimizer and overlapped grad
    reduce and param gather, and Megatron-FSDP v1 'optim' / 'optim_grads' (also mixed between
    dense and expert parameters)."""
    args = _validate_moonep_args(monkeypatch, flags)
    # The MoonEP block ran: it sized the token buffer as seq_length * micro_batch_size.
    assert args.moe_moonep_max_tokens_per_rank == 512
    # Megatron-FSDP v1 runs on the distributed optimizer too.
    assert args.use_distributed_optimizer
    if args.use_megatron_fsdp:
        assert args.megatron_fsdp_version == 1 and args.ckpt_format == "fsdp_dtensor"
    else:
        assert args.overlap_grad_reduce and args.overlap_param_gather
        assert args.ckpt_format == "torch_dist"


@pytest.mark.parametrize(
    "flags",
    [
        # The CLI defaults: Megatron DDP without the distributed optimizer, torch_dist checkpoints.
        pytest.param("", id="ddp"),
        pytest.param(
            "--use-distributed-optimizer --num-distributed-optimizer-instances 2",
            id="ddp-distopt-instances",
        ),
    ],
)
def test_validate_args_accepts_ddp(monkeypatch, flags):
    """Megatron DDP keeps the parameters whole without the distributed optimizer, and with it
    over several instances (ZeRO-1 within each, replicated across them): MoonEP rejects hybrid
    sharding with Megatron-FSDP only."""
    args = _validate_moonep_args(monkeypatch, flags)
    assert args.moe_moonep_max_tokens_per_rank == 512
    assert not args.use_megatron_fsdp and args.ckpt_format == "torch_dist"
    assert args.use_distributed_optimizer == ("--use-distributed-optimizer" in flags)


@pytest.mark.parametrize(
    "flags, match",
    [
        # The CLI default of --data-parallel-sharding-strategy is optim_grads_params.
        pytest.param(MFSDP, _strategies_error("optim_grads_params"), id="mfsdp-default"),
        pytest.param(
            MFSDP + " --data-parallel-sharding-strategy optim_grads_params "
            "--expert-data-parallel-sharding-strategy optim_grads",
            _strategies_error("optim_grads", "optim_grads_params"),
            id="mfsdp-dense-optim_grads_params",
        ),
        pytest.param(
            MFSDP_ZERO2 + " --expert-data-parallel-sharding-strategy optim_grads_params",
            _strategies_error("optim_grads", "optim_grads_params"),
            id="mfsdp-expert-optim_grads_params",
        ),
        pytest.param(
            MFSDP + " --data-parallel-sharding-strategy no_shard",
            _strategies_error("no_shard"),
            id="mfsdp-no_shard",
        ),
        pytest.param(
            MFSDP_ZERO2 + " --megatron-fsdp-version 2",
            "MoonEP supports Megatron-FSDP v1 only",
            id="mfsdp-v2",
        ),
        pytest.param(
            MFSDP_ZERO2 + " --num-distributed-optimizer-instances 2",
            HYBRID_ERROR + r"\(--num-distributed-optimizer-instances > 1\)",
            id="mfsdp-optimizer-instances",
        ),
        pytest.param(
            MFSDP_ZERO2 + " --outer-dp-sharding-strategy optim",
            HYBRID_ERROR + r"\(--outer-dp-sharding-strategy\)",
            id="mfsdp-outer-dp-optim",
        ),
        pytest.param(
            MFSDP_ZERO2 + " --megatron-fsdp-enable-fine-grained-param-gather",
            MFSDP_ERROR + "needs whole-layer parameter gathering",
            id="mfsdp-fine-grained-param-gather",
        ),
        # The generic rule against torch FSDP2 with expert parallelism fires first for every
        # MoonEP model (EP > 1); the MoonEP rule backs it up.
        pytest.param(
            "--use-torch-fsdp2 --no-gradient-accumulation-fusion "
            "--untie-embeddings-and-output-weights",
            "MoonEP does not support torch FSDP2|"
            "--use-torch-fsdp2 is not supported with expert parallelism",
            id="torch-fsdp2",
        ),
        pytest.param(
            "--use-distributed-optimizer --ckpt-format torch_dcp",
            DDP_CKPT_ERROR,
            id="ddp-torch_dcp",
        ),
        pytest.param(
            "--use-distributed-optimizer --ckpt-format torch", DDP_CKPT_ERROR, id="ddp-torch"
        ),
        pytest.param(
            "--use-distributed-optimizer --ckpt-format fsdp_dtensor",
            DDP_CKPT_ERROR,
            id="ddp-fsdp_dtensor",
        ),
        pytest.param(
            "--inprocess-restart",
            r"MoonEP \(phase 1\) does not support --inprocess-restart",
            id="inprocess-restart",
        ),
    ],
)
def test_validate_args_rejects_beyond_zero2(monkeypatch, flags, match):
    """Everything but ZeRO-1/2: parameter sharding (Megatron-FSDP optim_grads_params, also as
    the CLI default, Megatron-FSDP v2, torch FSDP2), Megatron-FSDP no_shard, hybrid sharding,
    fine-grained parameter gathering, DDP checkpoint formats other than torch_dist, and
    in-process restart."""
    with pytest.raises(AssertionError, match=match):
        _validate_moonep_args(monkeypatch, flags)


# ---- validate_args: the MoonEP token buffer --------------------------------------------------


@pytest.mark.parametrize(
    "flags, max_tokens",
    [
        pytest.param("--context-parallel-size 2", 512 // 2, id="context-parallel"),
        pytest.param(
            "--tensor-model-parallel-size 2 --sequence-parallel", 512 // 2, id="sequence-parallel"
        ),
        # The eval global batch must be a multiple of the eval micro-batch size times DP (8).
        pytest.param(
            "--eval-micro-batch-size 4 --eval-global-batch-size 32", 512 * 4, id="eval-micro-batch"
        ),
        pytest.param("--moe-moonep-max-tokens-per-rank 1000", 1000, id="explicit"),
    ],
)
def test_validate_args_max_tokens_per_rank(monkeypatch, flags, max_tokens):
    """The token buffer holds a rank's largest micro-batch: seq_length (512) times the larger of
    the training and evaluation micro-batch sizes, divided by the context-parallel size and, under
    sequence parallelism, by the tensor-parallel size. --moe-moonep-max-tokens-per-rank sets it
    instead."""
    # Megatron DDP with TP or CP > 1 requires CUDA_DEVICE_MAX_CONNECTIONS=1 before Blackwell.
    args = _validate_moonep_args(monkeypatch, flags, CUDA_DEVICE_MAX_CONNECTIONS="1")
    assert args.moe_moonep_max_tokens_per_rank == max_tokens
