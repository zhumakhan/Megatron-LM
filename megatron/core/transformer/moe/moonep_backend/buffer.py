# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The process-global ``moonep.Buffer`` shared by every MoonEP MoE layer.

One Buffer serves every layer and micro-batch: fresh dispatches allocate their own plan
tensors, plan-reuse paths build nothing, and outputs are fresh copies (``zero_copy=False``), so
each (layer, micro-batch) only has to keep its own plan until its last use. All layers must
therefore agree on ``(S, H, K, E, B, token_padding, num_sms)``.
"""

from typing import Optional

import torch.distributed as dist

_BUFFER = None
_BUFFER_KEY: Optional[tuple] = None


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
        B=num_experts // ep_size,
        group=group,
        explicitly_destroy=True,
    )
    _BUFFER_KEY = key
    return _BUFFER


def destroy_moonep_buffer() -> None:
    """Destroy the process-global Buffer. Collective over the Buffer's EP group."""
    global _BUFFER, _BUFFER_KEY
    if _BUFFER is not None:
        _BUFFER.destroy()
    _BUFFER, _BUFFER_KEY = None, None


def release_moonep_buffer() -> None:
    """Drop the reference to the Buffer without the collective teardown; never blocks.

    For ``destroy_model_parallel`` and fault paths, where peers may be gone.
    """
    global _BUFFER, _BUFFER_KEY
    _BUFFER, _BUFFER_KEY = None, None


def moonep_buffer_is_live() -> bool:
    """Whether a process-global Buffer currently exists."""
    return _BUFFER is not None
