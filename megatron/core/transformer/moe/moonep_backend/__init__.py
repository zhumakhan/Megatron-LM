# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP expert-parallel backend of the flex token dispatcher.

MoonEP (https://github.com/zhumakhan/MoonEP) moves tokens over NVLink with symmetric-memory
dispatch/combine kernels and balances load by prefetching the weights of hot remote experts into
local slots. Enable it with ``--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend
moonep``.

Nothing in this package imports ``moonep`` at module level, so Megatron keeps importing on
machines without it; the modules that need it import it inside the functions that use it.
"""

from typing import Iterator

import torch


def uses_moonep(config) -> bool:
    """Whether ``config`` selects the MoonEP flex dispatcher backend."""
    return (
        getattr(config, "moe_token_dispatcher_type", None) == "flex"
        and getattr(config, "moe_flex_dispatcher_backend", None) == "moonep"
    )


def iter_moonep_experts(module: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """Yield every ``MoonEPGroupedMLP`` inside ``module`` (wrappers such as DDP included)."""
    for submodule in module.modules():
        if getattr(submodule, "_is_moonep_experts", False):
            yield submodule
