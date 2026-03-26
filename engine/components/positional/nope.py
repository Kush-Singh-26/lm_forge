"""
engine/components/positional/nope.py

No positional encoding — all PEOutput fields are None.

Useful as an ablation baseline to measure how much PE contributes.
Some recent work (e.g. "NoPE" — Kazemnejad et al., 2023) shows that
decoder-only models can acquire implicit length generalisation without
any explicit PE.

type key: "none"
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from engine.components.positional import PEOutput, register
from engine.config.schema import PositionalConfig


@register("none")
class NoPE(nn.Module):
    """Pass-through — returns an empty PEOutput."""

    def __init__(
        self,
        config: PositionalConfig,
        hidden_size: int = 0,
        num_heads: int = 0,
    ) -> None:
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> PEOutput:
        return PEOutput()
