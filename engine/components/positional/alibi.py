"""
engine/components/positional/alibi.py

Attention with Linear Biases — Press et al., 2022.

Instead of rotating q/k, ALiBi adds a head-specific linear penalty to
attention logits:
    score(i,j) = q_i · k_j / sqrt(d) - m_h * (i - j)

where m_h is a geometric sequence of slopes, one per head.

Key properties:
  • No learned parameters
  • Strong length extrapolation (trained on 1K, infers on 4K+)
  • Does NOT interact with q/k — purely an additive bias on attention logits

type key: "alibi"
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from engine.components.positional import PEOutput, register
from engine.config.schema import PositionalConfig


def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute the per-head slope sequence defined in the ALiBi paper.

    For H heads, the slopes are:
        m_h = 2^{-8h/H}  for h = 1..H

    If H is not a power of 2, the sequence is padded to the next power of 2
    and then truncated.
    """

    def _slopes_pow2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(_slopes_pow2(num_heads), dtype=torch.float32)

    # Pad to next power of 2, interleave, truncate
    n_pow2 = int(2 ** math.ceil(math.log2(num_heads)))
    slopes = _slopes_pow2(n_pow2)
    # Take every other slope to halve, then fill the rest from the full set
    extra = _slopes_pow2(n_pow2 * 2)[0::2][: num_heads - n_pow2 // 2]
    return torch.tensor(slopes[: n_pow2 // 2] + extra, dtype=torch.float32)


@register("alibi")
class ALiBi(nn.Module):
    """
    ALiBi positional bias.

    Returns PEOutput(attn_bias=...) — an additive (B, H, S, S) tensor
    that attention modules add to their logits before softmax.
    """

    def __init__(
        self,
        config: PositionalConfig,
        hidden_size: int = 0,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        slopes = _get_alibi_slopes(num_heads)  # (H,)
        self.register_buffer("slopes", slopes)
        self._num_heads = num_heads
        self._cached_len: int = 0
        self.register_buffer("_bias_cache", torch.empty(0), persistent=False)

    def _build(self, seq_len: int) -> None:
        # Relative position matrix: positions[i, j] = i - j
        # For causal models, we only care about i >= j.
        # i - j is the distance from the query to the key.
        device = self.slopes.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        bias = -self.slopes.unsqueeze(-1).unsqueeze(-1) * pos.abs().float()
        self._bias_cache = bias.unsqueeze(0)  # (1, H, S, S)
        self._cached_len = seq_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> PEOutput:
        req_len = seq_len
        if position_ids is not None:
            req_len = max(req_len, int(position_ids.max().item()) + 1)
            
        if req_len > self._cached_len:
            self._build(max(req_len * 2, 2048))

        bias = self._bias_cache[:, :, :req_len, :req_len].to(hidden_states.device)
        return PEOutput(attn_bias=bias)
