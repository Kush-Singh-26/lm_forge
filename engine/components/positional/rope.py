"""
engine/components/positional/rope.py

Rotary Positional Embeddings — Su et al., 2023.

Encodes position by rotating query/key pairs in 2-D subspaces.
Does NOT modify hidden_states — returns cos/sin tables for use inside
attention via apply_rope().

Key properties:
  • Relative positions emerge naturally from dot-product rotation
  • Extrapolates to lengths unseen at training (esp. with NTK scaling)
  • Cache is extended lazily so you never need to set max_seq_len upfront

type key: "rope"
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from engine.components.positional import PEOutput, register
from engine.config.schema import PositionalConfig


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate q and k using precomputed cos/sin tables.

    Args:
        q, k : (B, H, S, head_dim)
        cos  : (1, 1, S, head_dim)
        sin  : (1, 1, S, head_dim)
    """

    def _rot(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    return (q * cos + _rot(q) * sin).to(q.dtype), (k * cos + _rot(k) * sin).to(k.dtype)


@register("rope")
class RoPE(nn.Module):
    """
    Precomputed + cached RoPE tables.

    Returns PEOutput(cos=..., sin=...) — the rest of PEOutput is None.
    Attention modules call apply_rope() with these tables.
    """

    def __init__(
        self,
        config: PositionalConfig,
        hidden_size: int = 0,  # unused; kept for uniform factory signature
        num_heads: int = 0,
        head_dim: int = 0,
    ) -> None:
        super().__init__()
        self.theta = config.theta
        self.scaling_type = config.scaling_type
        self.factor = config.factor
        self._head_dim: int = head_dim  # set by model after construction
        self._cached_len: int = 0
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

    def set_head_dim(self, head_dim: int) -> None:
        """Called by the model after it knows head_dim."""
        self._head_dim = head_dim

    def _build(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cpu")

        theta = self.theta
        if self.scaling_type == "ntk":
            # Dynamic NTK scaling: adjust theta based on sequence length extension
            # Simplified static version: scale base by factor
            theta = self.theta * (
                self.factor ** (self._head_dim / (self._head_dim - 2))
            )

        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, self._head_dim, 2, dtype=torch.float32, device=device)
                / self._head_dim
            )
        )
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (S, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (S, D)
        self._cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
        self._sin = emb.sin().unsqueeze(0).unsqueeze(0)
        self._cached_len = seq_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> PEOutput:
        if (
            seq_len > self._cached_len
            or self._cos.device.type != hidden_states.device.type
        ):
            self._build(max(seq_len * 2, 2048), device=hidden_states.device)

        if position_ids is None:
            cos = self._cos[:, :, :seq_len]
            sin = self._sin[:, :, :seq_len]
        else:
            cos = self._cos[0, 0][position_ids].unsqueeze(1)
            sin = self._sin[0, 0][position_ids].unsqueeze(1)

        return PEOutput(cos=cos, sin=sin)
