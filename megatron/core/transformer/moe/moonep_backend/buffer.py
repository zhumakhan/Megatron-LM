# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The process-global ``moonep.Buffer`` and expert pools shared by every MoonEP MoE layer.

One Buffer serves every layer and micro-batch: fresh dispatches allocate their own plan
tensors, plan-reuse paths build nothing, and outputs are fresh copies (``zero_copy=False``), so
each (layer, micro-batch) only has to keep its own plan until its last use. All layers must
therefore agree on ``(S, H, K, E, token_padding, num_sms)``. Likewise one set of expert pools
(``storage.MoonEPExpertPools``) serves every layer, so all layers share ``(H, Hi)`` too.
"""

from typing import Optional

import torch.distributed as dist

from megatron.core.transformer.moe.moonep_backend.autograd import reset_first_use_alignment

_BUFFER = None
_BUFFER_KEY: Optional[tuple] = None
_POOLS = None
_POOLS_KEY: Optional[tuple] = None


def get_moonep_buffer(
    group: dist.ProcessGroup,
    *,
    max_tokens_per_rank: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    num_sms: Optional[int],
    token_padding: int,
):
    """Return the process-global ``moonep.Buffer``, creating it on first use.

    Creation is collective over ``group`` (it allocates symmetric NVLink memory and barriers).
    """
    global _BUFFER, _BUFFER_KEY
    key = (
        tuple(dist.get_process_group_ranks(group)),
        max_tokens_per_rank,
        hidden_size,
        topk,
        num_experts,
        num_sms,
        token_padding,
    )
    if _BUFFER is not None:
        if _BUFFER_KEY != key:
            raise RuntimeError(
                f"MoonEP: all MoE layers must share one moonep.Buffer, but a layer requested {key} "
                f"while the live Buffer was built for {_BUFFER_KEY}. Call destroy_moonep_buffer() "
                "(collective) first."
            )
        return _BUFFER

    from moonep import Buffer

    ep_size = group.size()
    _BUFFER = Buffer(
        max_tokens_per_rank,
        hidden_size,
        topk,
        num_experts,
        ep_size,
        num_sms=num_sms,
        token_padding=token_padding,
        group=group,
        explicitly_destroy=True,
    )
    _BUFFER_KEY = key
    return _BUFFER


def get_moonep_expert_pools(
    buffer, group: dist.ProcessGroup, hidden_size: int, ffn_hidden_size: int
):
    """Return the process-global expert pools, creating them on first use.

    Creation is collective over ``group`` (VMM allocation, handle exchange and a barrier); every
    rank builds its MoE layers in the same order, so the first layer creates them everywhere.
    """
    global _POOLS, _POOLS_KEY
    key = (tuple(dist.get_process_group_ranks(group)), hidden_size, ffn_hidden_size)
    if _POOLS is not None:
        if _POOLS_KEY != key:
            raise RuntimeError(
                f"MoonEP: all MoE layers must share one expert pool set, but a layer requested "
                f"{key} while the live pools were built for {_POOLS_KEY}. Call "
                "destroy_moonep_buffer() (collective) first."
            )
        return _POOLS

    from megatron.core.transformer.moe.moonep_backend.storage import MoonEPExpertPools

    _POOLS = MoonEPExpertPools(buffer, group, hidden_size, ffn_hidden_size)
    _POOLS_KEY = key
    return _POOLS


def destroy_moonep_buffer() -> None:
    """Destroy the process-global expert pools and Buffer. Collective over the EP group."""
    global _BUFFER, _BUFFER_KEY, _POOLS, _POOLS_KEY
    if _POOLS is not None:
        _POOLS.destroy()
    _POOLS, _POOLS_KEY = None, None
    if _BUFFER is not None:
        _BUFFER.destroy()
    _BUFFER, _BUFFER_KEY = None, None
    reset_first_use_alignment()


def release_moonep_buffer() -> None:
    """Drop the references to the pools and Buffer without the collective teardown; never
    blocks.

    For ``destroy_model_parallel`` and fault paths, where peers may be gone.
    """
    global _BUFFER, _BUFFER_KEY, _POOLS, _POOLS_KEY
    _BUFFER, _BUFFER_KEY, _POOLS, _POOLS_KEY = None, None, None, None
    reset_first_use_alignment()


def end_moonep_pools_step() -> None:
    """Forget the pools' per-step state: which layer's copies they hold and whose
    gradients they wait to reduce (once per step, see ``finalize_moonep_step``)."""
    if _POOLS is not None:
        _POOLS.end_step()


def moonep_buffer_is_live() -> bool:
    """Whether a process-global Buffer currently exists."""
    return _BUFFER is not None
