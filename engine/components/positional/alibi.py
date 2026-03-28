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

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> PEOutput:
        """
        Produce ALiBi bias.
        seq_len: total length (including past)
        """
        B, S, _ = hidden_states.shape
        kv_len = seq_len
        device = hidden_states.device
        dtype = hidden_states.dtype

        if position_ids is None:
            # Queries are the last S positions
            # (1, S)
            q_pos = torch.arange(kv_len - S, kv_len, device=device).unsqueeze(0)
        else:
            q_pos = position_ids  # (B, S)

        # Keys are always all positions from 0 to kv_len-1
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)  # (1, kv_len)

        # Distance matrix (B, S, kv_len)
        # ALiBi is calculated as: -(i - j).abs() * slope
        # Optimized broadcasting to avoid large intermediates where possible
        dist = (q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)).abs()

        # Apply slopes: (1, H, 1, 1) * (B, 1, S, kv_len) -> (B, H, S, kv_len)
        # Cast to model's primary dtype early to avoid high-precision intermediates
        bias = -self.slopes.view(1, -1, 1, 1).to(dtype) * dist.to(dtype)

        return PEOutput(attn_bias=bias)
