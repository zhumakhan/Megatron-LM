# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hand-written DiT attention used as the bitwise reference for native Megatron features.

Before ``AttnMaskType.key_padding``, ``cp_comm_type='ulysses'``, ``rope_freqs_per_token`` and
``attention_output_gate_in_fp32`` existed, a video DiT built on Megatron implemented them by
subclassing ``TEDotProductAttention`` and ``SelfAttention``. The two classes below are copies of
that code (up to line wrapping, the class names and the docstrings), reduced to the default path
that the tests exercise: Transformer Engine attention kernels and real-valued rotary
frequencies. The mask helper was added. Tests check that the native features reproduce the
classes bitwise.
"""

from copy import copy
from functools import partial

import torch

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType


class DiTMaskedAttention(TEDotProductAttention):
    """TE attention with a hand-written key-padding mask and Ulysses context parallelism.

    The mask is ``[b, 1, 1, sk]`` bool, True = ignore that key, with holes anywhere. fp16/bf16
    inputs run as a TE THD varlen call that keeps every query and compacts the keys and values
    of each sample to its valid positions; fp32 inputs run the dense mask. A sample without a
    valid key attends to key 0, and its output rows are zeroed. Packed (THD) inputs run with
    TE's ``padding`` mask. With context parallelism the kernel is built for ``1 / cp`` of the
    heads, and :meth:`forward` exchanges q/k/v and the output with all-to-alls over the
    context-parallel group around :meth:`attend`.
    """

    def __init__(self, attention_type, config, pg_collection, **kwargs):
        context_group = pg_collection.cp if config.context_parallel_size > 1 else None
        if context_group is not None:
            config = copy(config)
            config.num_attention_heads //= context_group.size()
            config.num_query_groups //= context_group.size()
            config.context_parallel_size = 1
        super().__init__(
            attention_type="cross", config=config, pg_collection=pg_collection, **kwargs
        )
        self.context_group = context_group

    def forward(self, query, key, value, attention_mask, **kwargs):
        """Attention of the local sequence shard; Ulysses all-to-alls around ``attend``."""
        group = self.context_group
        if group is None:
            return self.attend(query, key, value, attention_mask, **kwargs)
        length, batch, heads, width = query.shape
        exchanged = [
            all_to_all_sp2hp(tensor.flatten(2), group=group).view(
                length * group.size(), batch, heads // group.size(), width
            )
            for tensor in (query, key, value)
        ]
        output = self.attend(*exchanged, attention_mask, **kwargs)
        return all_to_all_hp2sp(output, group=group).view(length, batch, -1)

    def attend(self, query, key, value, attention_mask, **kwargs):
        """TE attention over the full sequence with the key-padding mask."""
        if kwargs.get("packed_seq_params") is not None:
            kwargs["attn_mask_type"] = AttnMaskType.padding
            return super().forward(query, key, value, None, **kwargs)
        if attention_mask is None:
            return super().forward(query, key, value, attention_mask, **kwargs)
        empty = attention_mask.all(dim=-1, keepdim=True)
        first = torch.arange(key.shape[0], device=key.device) == 0
        safe_mask = attention_mask & ~(empty & first)
        if query.dtype == torch.float32:
            output = super().forward(query, key, value, safe_mask, **kwargs)
            return output.masked_fill(empty[:, 0].transpose(0, 1), 0)
        length, batch, heads, width = query.shape
        valid = ~safe_mask[:, 0, 0]
        lengths = valid.sum(dim=-1, dtype=torch.int32)
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.arange(batch + 1, device=query.device, dtype=torch.int32) * length,
            cu_seqlens_kv=offsets,
            max_seqlen_q=length,
            max_seqlen_kv=key.shape[0],
        )
        kwargs["attn_mask_type"] = AttnMaskType.padding
        kwargs["packed_seq_params"] = packed
        output = super().forward(
            query.transpose(0, 1).reshape(-1, heads, width).contiguous(),
            key.transpose(0, 1)[valid].contiguous(),
            value.transpose(0, 1)[valid].contiguous(),
            None,
            **kwargs,
        )
        output = output.reshape(batch, length, -1).transpose(0, 1)
        return output.masked_fill(empty[:, 0].transpose(0, 1), 0)


class DiTSelfAttention(SelfAttention):
    """SelfAttention with the DiT's initialization, packed (THD) branch and output gate.

    Packed inputs (``packed_seq_params`` with ``[t, 1, h]`` hidden states) rotate the queries and
    keys with one rotary frequency row per packed token as an sbhd ``[t, 1, np, hn]`` tensor, so
    positions do not restart at sequence starts, and run the core attention on the THD tokens.
    Other inputs take the stock SelfAttention forward. The output gate ``x * sigmoid(gate)`` is
    computed in the input dtype.
    """

    def __init__(self, config, **kwargs):
        config = copy(config)
        width = config.kv_channels * config.num_attention_heads
        initialize = partial(torch.nn.init.normal_, std=(2 / (config.hidden_size + width)) ** 0.5)
        config.init_method = config.output_layer_init_method = initialize
        super().__init__(config=config, **kwargs)

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None, **kwargs):
        """The hand-written branch for packed inputs, the stock forward for the others."""
        packed = kwargs.get("packed_seq_params")
        if packed is None:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                **kwargs,
            )
        # Frequencies already contain each image's ORIGINAL coordinates.
        # Do not let THD RoPE reset/reindex an already per-token frequency map.
        qkv = self.get_query_key_value_tensors(
            hidden_states, output_gate=self.config.attention_output_gate
        )
        query, key, value = qkv[:3]
        if rotary_pos_emb is not None:
            query = apply_rotary_pos_emb(
                query, rotary_pos_emb, self.config, cp_group=self.pg_collection.cp
            )
            key = apply_rotary_pos_emb(
                key, rotary_pos_emb, self.config, cp_group=self.pg_collection.cp
            )
        output = self.core_attention(
            query.squeeze(1),
            key.squeeze(1),
            value.squeeze(1),
            None,
            packed_seq_params=packed,
            attn_mask_type=AttnMaskType.padding,
        )
        output = output.reshape(hidden_states.shape[0], 1, -1)
        if self.config.attention_output_gate:
            output = self._apply_output_gate(output, qkv[3])
        return self.linear_proj(output)

    def _apply_output_gate(self, x, gate):
        """Output gate in the dtype of ``x``: sigmoid(gate) is rounded before the product."""
        return x * torch.sigmoid(gate.reshape_as(x))


def dit_key_padding_mask(caption_lengths, registers, caption_slots, video_tokens, device="cuda"):
    """Key mask of a DiT sequence ``[registers | caption | caption padding | video]``.

    Returns a ``[b, 1, 1, registers + caption_slots + video_tokens]`` bool mask, True = ignore
    that key: the caption padding forms a hole between the valid caption and the video tokens.
    """
    caption = (
        torch.arange(caption_slots, device=device)[None, :]
        < torch.tensor(caption_lengths, device=device)[:, None]
    )
    batch = caption.shape[0]
    valid = torch.cat(
        (
            torch.ones(batch, registers, dtype=torch.bool, device=device),
            caption,
            torch.ones(batch, video_tokens, dtype=torch.bool, device=device),
        ),
        dim=1,
    )
    return ~valid[:, None, None, :]
