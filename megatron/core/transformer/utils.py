# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utilities for transformer layers."""

import copy
import gc
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedStateDict, StateDict
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.utils import (
    get_pg_rank,
    get_tensor_model_parallel_group_if_none,
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)

if TYPE_CHECKING:
    from megatron.core.transformer import TransformerConfig

logger = logging.getLogger(__name__)


def set_attention_backend(config: 'TransformerConfig') -> None:
    """Configure Transformer Engine attention backends from a transformer config."""
    attention_backend = config.attention_backend
    if isinstance(attention_backend, str):
        try:
            attention_backend = AttnBackend[attention_backend]
        except KeyError as exc:
            choices = ", ".join(backend.name for backend in AttnBackend)
            raise ValueError(
                f"Unknown attention backend {attention_backend!r}; expected one of: {choices}"
            ) from exc

    if config.batch_invariant_mode:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            assert_te_supports_batch_invariant_attention,
        )

        assert_te_supports_batch_invariant_attention()

    def check_and_set_env_variable(
        env_variable_name: str, expected_value: int, attn_type: AttnBackend
    ) -> None:
        current_value = os.getenv(env_variable_name)
        assert current_value is None or current_value == str(expected_value), (
            f'{env_variable_name} is set to {current_value}, but attention_backend='
            f'{attn_type.name!r} and flash_attention_version='
            f'{config.flash_attention_version!r} require {expected_value}. Transformer Engine '
            'attention backend and FlashAttention version controls are process-wide; all models '
            'constructed in one process must use compatible settings. Unset the NVTE attention '
            'environment variables before model construction to let the configs set them.'
        )
        os.environ[env_variable_name] = str(expected_value)

    backend_flags = {
        AttnBackend.local: (0, 0, 0),
        AttnBackend.flash: (1, 0, 0),
        AttnBackend.fused: (0, 1, 0),
        AttnBackend.unfused: (0, 0, 1),
        AttnBackend.auto: (1, 1, 1),
    }
    flash, fused, unfused = backend_flags[attention_backend]
    check_and_set_env_variable('NVTE_FLASH_ATTN', flash, attention_backend)
    check_and_set_env_variable('NVTE_FUSED_ATTN', fused, attention_backend)
    check_and_set_env_variable('NVTE_UNFUSED_ATTN', unfused, attention_backend)

    if config.flash_attention_version is not None:
        for version in (2, 3, 4):
            check_and_set_env_variable(
                f'NVTE_FLASH_ATTN_V{version}',
                int(version == config.flash_attention_version),
                attention_backend,
            )


def get_linear_layer(rows, columns, init_method, perform_initialization=True):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if perform_initialization:  # Take from modelparallel config
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


def get_default_causal_mask(sq: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


def get_sliding_window_causal_mask(sq, skv, window_size):
    """Create the equivalent attention mask for SWA in [sq, skv] shape"""
    m = torch.ones(sq, skv, dtype=torch.bool, device="cuda")
    mu = torch.triu(m, diagonal=skv - sq - window_size[0])
    ml = torch.tril(mu, diagonal=skv - sq + window_size[1])
    ml = ~ml

    return ml


# pylint: disable=missing-function-docstring
def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def check_key_padding_mask(
    attention_mask: torch.Tensor, batch_size: Optional[int] = None, key_length: Optional[int] = None
) -> None:
    """Check that ``attention_mask`` is a ``[b, 1, 1, sk]`` boolean key-padding mask.

    Args:
        attention_mask (Tensor): the mask, True = ignore that key.
        batch_size (int, optional): batch size of the attention inputs. The mask must have this
            batch size, or batch size 1 for one mask shared by every sample.
        key_length (int, optional): number of keys of the attention inputs, which the mask
            must cover.

    Raises:
        ValueError: if the mask has another layout or dtype, or does not match the inputs.
    """
    if not (
        attention_mask.dim() == 4
        and attention_mask.shape[1] == 1
        and attention_mask.shape[2] == 1
        and attention_mask.dtype == torch.bool
    ):
        raise ValueError(
            "AttnMaskType.key_padding expects a [b, 1, 1, sk] bool attention mask, got "
            f"{tuple(attention_mask.shape)} {attention_mask.dtype}"
        )
    if batch_size is not None and attention_mask.shape[0] not in (1, batch_size):
        raise ValueError(
            f"key_padding mask has batch size {attention_mask.shape[0]}, but the attention "
            f"inputs have batch size {batch_size} (expected {batch_size} or 1)"
        )
    if key_length is not None and attention_mask.shape[-1] != key_length:
        raise ValueError(
            f"key_padding mask covers {attention_mask.shape[-1]} keys, but the attention inputs "
            f"have {key_length} keys"
        )


def check_key_padding_window(sliding_window: bool) -> None:
    """Reject AttnMaskType.key_padding on a layer with sliding window attention.

    Args:
        sliding_window (bool): whether the layer applies sliding window attention.
    """
    if sliding_window:
        raise ValueError(
            "AttnMaskType.key_padding does not support sliding window attention (window_size): "
            "the varlen kernel would apply the window to compacted key positions. Use "
            "AttnMaskType.arbitrary with a dense mask on windowed layers."
        )


def key_padding_safe_mask(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make every row of a key-padding mask attend to at least one key.

    Args:
        attention_mask (Tensor): ``[b, 1, 1, sk]`` boolean key-padding mask, True = ignore key.

    Returns:
        Tuple of the mask with key 0 unmasked for samples without any valid key, and the
        ``[b, 1, 1, 1]`` boolean flags of those samples. Their attention output must be zeroed
        by the caller.
    """
    empty = attention_mask.all(dim=-1, keepdim=True)
    first = torch.arange(attention_mask.shape[-1], device=attention_mask.device) == 0
    return attention_mask & ~(empty & first), empty


@dataclass(frozen=True)
class KeyPaddingMetadata:
    """Varlen (THD) layout of a ``[b, 1, 1, sk]`` key-padding mask.

    Every query is kept and the keys/values of each sample are compacted to its valid positions.
    Samples without any valid key keep key 0 (see :func:`key_padding_safe_mask`); their output
    rows must be zeroed with :attr:`empty`.
    """

    sbhd_kv_index: torch.Tensor
    """int64 ``[nnz]``: kept rows of the ``[sk * b, ...]`` view of an ``[sk, b, ...]`` key or
    value tensor, in sample-major (THD) order."""

    bshd_kv_index: torch.Tensor
    """int64 ``[nnz]``: kept rows of the ``[b * sk, ...]`` view of a ``[b, sk, ...]`` key or
    value tensor, in sample-major (THD) order."""

    cu_seqlens_q: torch.Tensor
    """int32 ``[b + 1]``: cumulative query lengths, ``arange(b + 1) * sq``."""

    cu_seqlens_kv: torch.Tensor
    """int32 ``[b + 1]``: cumulative numbers of kept keys."""

    empty: torch.Tensor
    """bool ``[b, 1, 1, 1]``: samples without any valid key."""


_KEY_PADDING_METADATA_CACHE_SIZE = 8
_key_padding_metadata_cache: "OrderedDict[tuple, Tuple[torch.Tensor, KeyPaddingMetadata]]" = (
    OrderedDict()
)


def clear_key_padding_metadata_cache() -> None:
    """Drop all cached key-padding metadata and the masks the cache keeps alive.

    Call it after writing a cached mask in a way that does not bump its version counter (see
    :func:`get_key_padding_metadata`), e.g. once per step when a persistent mask buffer is
    refilled by a collective.
    """
    _key_padding_metadata_cache.clear()


def _build_key_padding_metadata(
    attention_mask: torch.Tensor, query_length: int
) -> KeyPaddingMetadata:
    batch, key_length = attention_mask.shape[0], attention_mask.shape[-1]
    safe_mask, empty = key_padding_safe_mask(attention_mask)
    valid = ~safe_mask[:, 0, 0]
    lengths = valid.sum(dim=-1, dtype=torch.int32)
    cu_seqlens_kv = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    cu_seqlens_q = (
        torch.arange(batch + 1, device=attention_mask.device, dtype=torch.int32) * query_length
    )
    # The only host synchronization: nonzero() needs the number of kept keys.
    bshd_kv_index = valid.reshape(-1).nonzero().squeeze(1)
    sbhd_kv_index = (bshd_kv_index % key_length) * batch + torch.div(
        bshd_kv_index, key_length, rounding_mode="floor"
    )
    return KeyPaddingMetadata(
        sbhd_kv_index=sbhd_kv_index,
        bshd_kv_index=bshd_kv_index,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        empty=empty,
    )


def get_key_padding_metadata(attention_mask: torch.Tensor, query_length: int) -> KeyPaddingMetadata:
    """Return the varlen (THD) metadata of a key-padding mask, computed once per mask.

    The metadata is cached on the mask storage (data pointer, version counter, shape, strides,
    device), so every layer that receives the same mask, and the activation-checkpoint
    recompute (which receives detached aliases of the same storage), reuse a single
    ``nonzero`` and its device-to-host synchronization. A cached mask is kept alive, so its
    storage cannot be reused by another tensor while the entry exists.

    Like autograd's checks of saved tensors, the cache relies on the mask's version counter:
    in-place PyTorch operations on the mask (or on a view of it) bump the version and
    invalidate the entry, but writes that bypass the version counter go unnoticed and would
    return stale metadata. Such writes include ``torch.distributed`` collectives into an
    existing mask buffer (``broadcast``, ``all_gather_into_tensor``, ...), custom CUDA or
    Triton kernels, and writes through ``.data`` or ``untyped_storage()``. After such a write,
    call ``torch.autograd.graph.increment_version(mask)`` or
    :func:`clear_key_padding_metadata_cache`, or pass a new mask tensor instead.

    Masks that are inference tensors, and calls under ``torch.inference_mode()``, bypass the
    cache: inference tensors have no version counter, and metadata built in inference mode
    could not be saved for backward by a later training forward.

    Args:
        attention_mask (Tensor): ``[b, 1, 1, sk]`` boolean mask, True = ignore that key.
        query_length (int): number of queries per sample.
    """
    check_key_padding_mask(attention_mask)
    if attention_mask.is_inference() or torch.is_inference_mode_enabled():
        return _build_key_padding_metadata(attention_mask, query_length)
    key = (
        attention_mask.data_ptr(),
        attention_mask._version,
        tuple(attention_mask.shape),
        attention_mask.stride(),
        attention_mask.device,
        query_length,
    )
    entry = _key_padding_metadata_cache.get(key)
    if entry is not None:
        _key_padding_metadata_cache.move_to_end(key)
        return entry[1]
    metadata = _build_key_padding_metadata(attention_mask, query_length)
    _key_padding_metadata_cache[key] = (attention_mask.detach(), metadata)
    while len(_key_padding_metadata_cache) > _KEY_PADDING_METADATA_CACHE_SIZE:
        _key_padding_metadata_cache.popitem(last=False)
    return metadata


def key_padding_pack(tensor: torch.Tensor, kv_index: torch.Tensor) -> torch.Tensor:
    """Gather the kept rows of an sbhd or bshd key or value tensor into THD ``[nnz, h, d]``.

    Args:
        tensor (Tensor): key or value tensor, ``[sk, b, h, d]`` or ``[b, sk, h, d]``.
        kv_index (Tensor): the matching :attr:`KeyPaddingMetadata.sbhd_kv_index` or
            :attr:`KeyPaddingMetadata.bshd_kv_index` of the key-padding mask.
    """
    return tensor.reshape(-1, *tensor.shape[2:]).index_select(0, kv_index)


def ulysses_inner_config(
    config: 'TransformerConfig', cp_group: torch.distributed.ProcessGroup
) -> 'TransformerConfig':
    """Config of the attention kernel that runs inside ``cp_comm_type='ulysses'``.

    The kernel sees the full sequence and ``1 / cp`` of the (tensor-parallel local) heads, so it
    is built with ``num_attention_heads`` and ``num_query_groups`` divided by the context-parallel
    size and with context parallelism disabled.

    Args:
        config (TransformerConfig): config of the attention layer.
        cp_group (ProcessGroup): the context-parallel group the sequence is sharded over.

    Raises:
        ValueError: without a context-parallel group, or if the heads or query groups cannot be
            split across the tensor- and context-parallel ranks.
    """
    if cp_group is None:
        raise ValueError("cp_comm_type='ulysses' needs an initialized context-parallel group")
    cp_size = cp_group.size()
    shards = config.tensor_model_parallel_size * cp_size
    for name in ("num_attention_heads", "num_query_groups"):
        if getattr(config, name) % shards != 0:
            raise ValueError(
                f"cp_comm_type='ulysses' needs {name} ({getattr(config, name)}) to be divisible "
                f"by tensor_model_parallel_size * context_parallel_size ({shards})"
            )
    inner = copy.copy(config)
    inner.num_attention_heads //= cp_size
    inner.num_query_groups //= cp_size
    inner.context_parallel_size = 1
    return inner


def check_ulysses_attn_mask_type(attn_mask_type: AttnMaskType) -> None:
    """Reject the mask types that ``cp_comm_type='ulysses'`` does not support.

    Ulysses needs the sequence sharded contiguously across the context-parallel group.
    Megatron's own context-parallel data and RoPE sharding is load-balanced (zig-zag) for
    causal attention, and a causal mask over a gathered zig-zag sequence silently gives wrong
    attention, so causal mask types are rejected.

    Args:
        attn_mask_type (AttnMaskType): mask type of the attention call.
    """
    if attn_mask_type in (
        AttnMaskType.causal,
        AttnMaskType.padding_causal,
        AttnMaskType.causal_bottom_right,
    ):
        raise ValueError(
            f"cp_comm_type='ulysses' does not support {attn_mask_type}: it needs contiguous "
            "context-parallel sequence shards, while Megatron's context-parallel data and RoPE "
            "sharding for causal attention is load-balanced (zig-zag). Use cp_comm_type 'p2p', "
            "'all_gather' or 'a2a' for causal attention."
        )


def check_ulysses_attention_type(attention_type: str) -> None:
    """Reject the attention types that ``cp_comm_type='ulysses'`` does not support.

    Ulysses exchanges keys and values over the context-parallel group exactly like the queries,
    so they must be sharded along the sequence like the queries, which holds for self-attention
    only. The keys and values of cross-attention come from a context that is not sharded that way
    (the keys and values of a replicated context would be duplicated ``cp`` times, and masks
    sized to the context would not match).

    Args:
        attention_type (str): attention type of the layer, "self" or "cross".
    """
    if attention_type != "self":
        raise ValueError(
            "cp_comm_type='ulysses' supports self-attention only, got "
            f"attention_type={attention_type!r}: it gathers the keys and values over the "
            "context-parallel group like the queries, so they must be sharded along the sequence "
            "like the queries."
        )


def ulysses_attention(
    attention_fn: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run ``attention_fn`` on the full sequence with heads split across the CP group (Ulysses).

    Contract: the sequence dimension (dim 0) of ``query``/``key``/``value`` is sharded
    contiguously across ``cp_group``: with ``s`` the length of the full sequence and ``cp`` the
    size of ``cp_group``, rank ``r`` holds positions ``[r * s / cp, (r + 1) * s / cp)``. Every
    tensor argument of ``attention_fn`` that covers the sequence, such as the attention mask, is
    CP-global.

    Args:
        attention_fn (Callable): attention kernel called as
            ``attention_fn(query, key, value, *args, **kwargs)`` on ``[s, b, h / cp, d]`` inputs;
            it returns ``[s, b, h / cp * d]``.
        query, key, value (Tensor): ``[s / cp, b, h, d]`` local sequence shards. The number of
            heads ``h`` may differ between the query and the key/value (grouped-query attention)
            and must be divisible by ``cp``.
        cp_group (ProcessGroup): context-parallel group.

    Returns:
        Tensor: ``[s / cp, b, h * d]`` attention output of the local sequence shard.
    """

    cp_size = cp_group.size()

    def scatter_heads(tensor: torch.Tensor) -> torch.Tensor:
        # [s / cp, b, h, d] -> [s, b, h / cp, d]: gather the sequence, keep 1 / cp of the heads.
        heads, head_dim = tensor.shape[-2:]
        if heads % cp_size != 0:
            raise ValueError(f"{heads} heads cannot be split across {cp_size} CP ranks")
        exchanged = all_to_all_sp2hp(tensor.flatten(-2), group=cp_group)
        return exchanged.view(
            tensor.shape[0] * cp_size, *tensor.shape[1:-2], heads // cp_size, head_dim
        )

    output = attention_fn(
        scatter_heads(query), scatter_heads(key), scatter_heads(value), *args, **kwargs
    )
    # [s, b, h / cp * d] -> [s / cp, b, h * d]
    return all_to_all_hp2sp(output, group=cp_group).view(*query.shape[:-2], -1)


@jit_fuser
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


# pylint: disable=missing-function-docstring
def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with
# type hints for ONNX exporter
# pylint: disable=missing-function-docstring
@jit_fuser
def erf_gelu(x):
    return (
        x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
    )


@torch.no_grad()
def cat_with_oom_fallback(sub_state_dict):
    """Merge sharded tensor pieces, falling back to CPU if device-side cat OOMs."""
    try:
        return torch.cat(sub_state_dict)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.warning(
            f"CUDA OutOfMemoryError encountered during tensors merging."
            f" Switching to CPU merge. (Error: {e})"
        )
        merged_sub_state_dict = torch.cat([t.cpu() for t in sub_state_dict])
        gc.collect()
        torch.cuda.empty_cache()
        return merged_sub_state_dict


def make_sharded_tensors_for_checkpoint(
    state_dict: StateDict,
    prefix: str,
    tensor_parallel_layers_axis_map: Optional[Dict[str, int]] = None,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    extra_state_suffix: str = '_extra_state',
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Wraps tensors from transformer layers with ShardedTensor or ShardedObject.

    For a given `state_dict`, wraps:
    - all _extra_states with ShardedObject
    - all tensors specified in tensor_parallel_layers_axis_map with TP and DP sharded ShardedTensor
    - other values with DP sharded ShardedTensor

    Args:
        state_dict (StateDict): state_dict to convert
        prefix (str): prefix appended to keys in final state dict
        tensor_parallel_layers_axis_map (Dict[str, int], optional): dict mapping layer
            names to the axis for TP sharding
        sharded_offsets (Iterable[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related), passed along to ShardedTensor
        extra_state_suffix (str, default = '_extra_state'): layers with this
            suffix will be wrapped with ShardedObject instead of ShardedTensor.
        tp_group (Optional[torch.distributed.ProcessGroup], optional): tensor parallel group.
            If None, defaults to parallel_state.get_tensor_model_parallel_group()
        dp_cp_group (Optional[torch.distributed.ProcessGroup], optional): data parallel group
            with context parallel. If None, defaults to
            parallel_state.get_data_parallel_group(with_context_parallel=True)

    """

    if tensor_parallel_layers_axis_map is None:
        tensor_parallel_layers_axis_map = {}

    if tp_group is None and dp_cp_group is None:
        tp_group = get_tensor_model_parallel_group_if_none(tp_group)
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    # GTP-sharded weights need the GTP axis layered onto the TP/DP offsets. The GTP helper
    # is a no-op for non-GTP state_dicts, but importing it eagerly would be circular, so
    # gate on HAVE_GTP and the presence of a GTP param before delegating.
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

    if HAVE_GTP:
        from megatron.core.tensor_parallel.gtp_api import (
            is_gtp_param,
            make_sharded_tensors_for_checkpoint_with_gtp_remat,
        )

        if any(is_gtp_param(t) for t in state_dict.values()):
            return make_sharded_tensors_for_checkpoint_with_gtp_remat(
                state_dict,
                prefix,
                tensor_parallel_layers_axis_map,
                sharded_offsets,
                extra_state_suffix=extra_state_suffix,
                tp_group=tp_group,
                dp_cp_group=dp_cp_group,
            )

    sharded_state_dict = {}
    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{prefix}{layer_name}'

        if layer_name.endswith(extra_state_suffix):
            # Compute replica_id when groups are provided
            replica_id = (0, get_pg_rank(tp_group), get_pg_rank(dp_cp_group))

            sharded_state_dict[layer_key] = make_sharded_object_for_checkpoint(
                tensor, layer_key, sharded_offsets, replica_id=replica_id
            )

        elif layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
                tensor,
                layer_key,
                tp_axis,
                prepend_offsets=sharded_offsets,
                tp_group=tp_group,
                dp_cp_group=dp_cp_group,
            )

        else:
            sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                tensor,
                layer_key,
                prepend_offsets=sharded_offsets,
                tp_group=tp_group,
                dp_cp_group=dp_cp_group,
            )

    return sharded_state_dict


def make_sharded_object_for_checkpoint(
    obj: Any,
    key: str,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    replica_id: Union[None, int, Tuple[int, ...]] = None,
    **kwargs,
):
    """Helper for instantiating a non-sharded ShardedObject (replicated across TP and DP group).

    Args:
        obj (object): any object to be sharded
        key (str): unique identifier of the object
        sharded_offsets (Iterable[Tuple[int, int, int]]): offsets normally
            prepended to ShardedTensors, will be used as global offsets for
            ShardedObject
        replica_id (Union[None, int, Tuple[int, ...]]): replica id
    """
    if replica_id is None:
        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

    return ShardedObject(key, obj, *_get_extra_state_offsets(sharded_offsets), replica_id, **kwargs)


def _get_extra_state_offsets(
    sharded_offsets: Iterable[Tuple[int, int, int]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Turns ShardedTensor offsets into offsets suitable for ShardedObject."""
    if sharded_offsets:
        sharded_offsets = sorted(sharded_offsets, key=itemgetter(0))  # sort by axis
        axis, extra_state_offset, extra_state_shape = zip(*sharded_offsets)
        assert list(axis) == list(
            range(len(axis))
        ), f'Expected contiguous axis for offsets: {sharded_offsets}'
    else:
        extra_state_shape = (1,)
        extra_state_offset = (0,)
    return extra_state_shape, extra_state_offset


def ensure_metadata_has_dp_cp_group(metadata: Optional[dict]) -> dict:
    """Ensure `metadata` is a dict containing `dp_cp_group` entry.

    If `metadata` is None, a new dict is returned with `dp_cp_group` set.
    If `metadata` is a dict and missing `dp_cp_group`, it is updated in-place.
    Otherwise, asserts that `dp_cp_group` exists.
    """
    if metadata is None:
        return {'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)}
    assert isinstance(metadata, dict), "metadata must be a dict with dp_cp_group as key"
    if 'dp_cp_group' not in metadata:
        metadata['dp_cp_group'] = parallel_state.get_data_parallel_group(with_context_parallel=True)
    return metadata


def sharded_state_dict_default(
    module: torch.nn.Module,
    prefix: str = '',
    sharded_offsets: Tuple[Tuple[int, int, int]] = (),
    metadata: Optional[dict] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> ShardedStateDict:
    """Provides implementation for sharded_state_dict method for non-MegatronModules.

    Tries to call `module.sharded_state_dict` when possible,
    otherwise uses regular state dict and assumes tensors are replicated across TP and DP.

    `keep_vars=True` is passed to module.state_dict so that optimizer states
    can be sharded later on.

    Args:
        module (torch.nn.Module): module which sharded state dict we want to obtain
        prefix (str): prefix for the state dict keys
        sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
        metadata (dict, optional): metadata passed to module sharded_state_dict method
        tp_group (Optional[torch.distributed.ProcessGroup], optional): tensor parallel group.
            If None, defaults to parallel_state.get_tensor_model_parallel_group()

    Returns:
        dict: dictionary of state dict keys mapped to ShardedTensors
    """

    # Guard for cases metadata is not provided
    metadata = ensure_metadata_has_dp_cp_group(metadata)

    if hasattr(module, 'sharded_state_dict'):
        module_sharded_sd = module.sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
    else:
        module_sd = module.state_dict(prefix='', keep_vars=True)
        module_sharded_sd = make_sharded_tensors_for_checkpoint(
            module_sd,
            prefix,
            {},
            sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )
    return module_sharded_sd


# Initialize cache for sequence parallel modules
_sequence_parallel_attr_cache = None


def set_model_config_attribute(model: Any, attribute: str, value: Any) -> None:
    """Set a config attribute on a model and all distinct child-module configs.

    Some models give individual layers separate config objects. Runtime model-wide
    toggles must update those configs just as they did when every layer shared the
    model's root config.

    Args:
        model: Model whose configs should be updated.
        attribute: Config attribute to set.
        value: Value to assign. The same value object is assigned to every config.
    """
    root_config = model.config
    setattr(root_config, attribute, value)
    updated_config_ids = {id(root_config)}

    module_root = model
    visited_wrapper_ids = set()
    while not isinstance(module_root, torch.nn.Module) or not hasattr(module_root, "_modules"):
        visited_wrapper_ids.add(id(module_root))
        module_root = getattr(module_root, "module", None)
        if module_root is None or id(module_root) in visited_wrapper_ids:
            return

    for module in module_root.modules():
        config = getattr(module, "config", None)
        if (
            config is not None
            and id(config) not in updated_config_ids
            and hasattr(config, attribute)
        ):
            setattr(config, attribute, value)
            updated_config_ids.add(id(config))


def _init_sequence_parallel_cache(model, exclude_modules):
    """
    Initialize the cache of modules with sequence parallel attributes.
    Only needs to be called once, subsequent calls have no effect.

    Args:
        model: model to change sequence parallelism attributes
        exclude_modules: Modules to exclude from changing sequence parallelism
    """
    global _sequence_parallel_attr_cache
    model_id = id(model)
    if _sequence_parallel_attr_cache is not None and model_id in _sequence_parallel_attr_cache:
        return  # Cache already initialized

    # Attributes for sequence parallel
    sequence_parallel_attrs = [
        "sequence_parallel",
        "scatter_to_sequence_parallel",
        "reduce_scatter_embeddings",
    ]

    if model.position_embedding_type == "learned_absolute":
        sequence_parallel_attrs.remove("reduce_scatter_embeddings")

    # Initialize dictionary to hold attributes -> list of modules
    if _sequence_parallel_attr_cache is None:
        _sequence_parallel_attr_cache = {}
    _sequence_parallel_attr_cache[model_id] = {attr: [] for attr in sequence_parallel_attrs}

    # Get the model
    model_modules = model

    # Recursive function to find all modules with our target attributes
    def find_modules_with_attrs(module):
        if exclude_modules is None or module not in exclude_modules:
            # Check if this module has any of our target attributes
            for attr in sequence_parallel_attrs:
                if hasattr(module, attr):
                    _sequence_parallel_attr_cache[model_id][attr].append(module)

            # Check all children modules recursively
            for child in module._modules.values():
                if child is not None:
                    find_modules_with_attrs(child)

    # Start the search from each major component
    find_modules_with_attrs(model_modules)


def set_model_to_sequence_parallel(model, set_to=False, exclude_modules=None):
    """
    Set sequence parallel attributes for the model.

    Args:
        set_to: Value to set for sequence_parallel attributes
        exclude_modules: Modules to exclude from changing sequence parallelism
    """
    global _sequence_parallel_attr_cache
    model_id = id(model)

    # Initialize cache if needed
    if _sequence_parallel_attr_cache is None or model_id not in _sequence_parallel_attr_cache:
        _init_sequence_parallel_cache(model, exclude_modules)

    set_model_config_attribute(model, "sequence_parallel", set_to)

    # Set all cached attributes to desired value
    for attr, modules in _sequence_parallel_attr_cache[model_id].items():
        for module in modules:
            setattr(module, attr, set_to)


# Initialize cache for modules
cuda_graph_attr_cache = None


def init_cuda_graph_cache(model):
    """
    Initialize the cache of modules for cuda graphs
    """
    global cuda_graph_attr_cache
    model_id = id(model)
    if cuda_graph_attr_cache is not None and model_id in cuda_graph_attr_cache:
        return  # Cache already initialized

    cuda_graph_attrs = ["cuda_graph_impl", "flash_decode", "cudagraph_manager"]

    # Special case handling for activation recomputation
    if model.config.recompute_granularity is not None:
        cuda_graph_attrs.append("recompute_granularity")

    # Initialize dictionary to hold attributes -> list of modules
    if cuda_graph_attr_cache is None:
        cuda_graph_attr_cache = {}

    cuda_graph_attr_cache[model_id] = {attr: [] for attr in cuda_graph_attrs}

    # Get the model
    model_modules = model

    # Recursive function to find all modules with our target attributes
    def find_modules_with_attrs(module):
        # Check if this module has any of our target attributes
        for attr in ["cuda_graph_impl", "flash_decode"]:
            if hasattr(module, attr) and isinstance(getattr(module, attr), bool):
                cuda_graph_attr_cache[model_id][attr].append(module)

            # Check for config variables
            if hasattr(module, "config"):
                if hasattr(module.config, attr):
                    cuda_graph_attr_cache[model_id][attr].append(module.config)

        # Specific caching for cuda graph managers
        if hasattr(module, "cudagraph_manager"):
            cuda_graph_attr_cache[model_id]["cudagraph_manager"].append(
                [module, module.cudagraph_manager]
            )

        # Specific caching for recompute granularity
        if hasattr(module, "recompute_granularity"):
            cuda_graph_attr_cache[model_id]["recompute_granularity"].append(
                [module, module.recompute_granularity]
            )

        # Check all children modules recursively
        for child in module._modules.values():
            if child is not None:
                find_modules_with_attrs(child)

    # Start the search from each major component
    find_modules_with_attrs(model_modules)


def toggle_cuda_graphs(model, set_to="none"):
    """
    Toggle CUDA graph-related attributes for the model and its modules.

    Args:
        set_to (str): Value to set for CUDA graph-related attributes.
    """
    global cuda_graph_attr_cache
    model_id = id(model)

    # Initialize cache if needed
    if cuda_graph_attr_cache is None or model_id not in cuda_graph_attr_cache:
        init_cuda_graph_cache(model)

    assert set_to in ["none", "local"], f"Invalid CUDA graph implementation: {set_to}"
    set_model_config_attribute(model, "cuda_graph_impl", set_to)

    # Collect all modules that have any of the CUDA graph attributes
    for attribute, modules in cuda_graph_attr_cache[model_id].items():
        if attribute == "cuda_graph_impl":
            for module in modules:
                setattr(module, attribute, set_to)
        elif attribute == "recompute_granularity":
            for module in modules:
                if set_to == "local":
                    # If we are turning on cuda graphs we need to turn of activation recomputation
                    setattr(module[0], attribute, None)
                else:
                    # If we are turning off cuda graphs we can set it to the cached value
                    setattr(module[0], attribute, module[1])
        # Cuda Graph manager case
        elif attribute == "cudagraph_manager":
            for module in modules:
                if set_to == "local":
                    # If we are not resetting cuda graphs we set it to its cached cuda graph
                    setattr(module[0], attribute, module[1])
                else:
                    for module in modules:
                        # If we are deleting the cuda graph, we delete its attribute
                        if hasattr(module[0], "cudagraph_manager"):
                            delattr(module[0], "cudagraph_manager")


def transition_moe_cudagraphs(model, scope: str):
    """
    Switch MoE layers to the given cudagraph scope. Flips between 'partial' and 'full'.

    Args:
        model: The model with MoE layers which will be transitioned.
        scope: 'partial' for training (router + postprocess captured, expert dispatch eager)
               or 'full' for inference (full-layer graph capture).
    """
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    for module in model.modules():
        if isinstance(module, MoETransformerLayer):
            module.transition_cudagraph_scope(scope)


def is_layer_window_attention(
    window_size: Optional[Tuple[int, int]], window_attn_skip_freq: int | list, layer_number: int
) -> bool:
    # layer_number is 1-indexed
    if not window_size:
        return False
    if window_attn_skip_freq is None:
        return True
    if isinstance(window_attn_skip_freq, int):
        return layer_number % window_attn_skip_freq != 0
    if isinstance(window_attn_skip_freq, list):
        return bool(window_attn_skip_freq[layer_number - 1])

    raise ValueError(
        f"Invalid `window_attn_skip_freq`: {type(window_attn_skip_freq)}, "
        f"{window_attn_skip_freq}"
    )
