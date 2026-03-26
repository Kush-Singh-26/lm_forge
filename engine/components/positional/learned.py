"""
engine/components/positional/learned.py

Learned absolute positional embeddings — Vaswani et al. / GPT-2 style.

A simple nn.Embedding(max_seq_len, hidden_size) where each position
gets its own learned vector.  Added directly to token embeddings before
any attention, so attention modules need no special handling.

Limitations:
  • Hard cap at max_seq_len — cannot extrapolate beyond training length
  • Position vectors are trained independently (no inductive bias)

type key: "learned"
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from engine.components.positional import PEOutput, register
from engine.config.schema import PositionalConfig


@register("learned")
class LearnedAbsolutePE(nn.Module):
    """
    Learned positional embedding table.

    Returns PEOutput(hidden_states=...) — the hidden states with position
    vectors added in.  Downstream attention sees plain input + position
    information baked into the activations.
    """

    def __init__(
        self,
        config: PositionalConfig,
        hidden_size: int = 512,
        num_heads: int = 0,
    ) -> None:
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(config.max_seq_len, hidden_size)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> PEOutput:
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds learned PE max {self.max_seq_len}. "
            "Increase positional.max_seq_len in your config."
        )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        pos_emb = self.embedding(position_ids)   # (B or 1, S, D)
        return PEOutput(hidden_states=hidden_states + pos_emb)
