"""
engine/components/attention/mha.py

Standard Multi-Head Attention (Vaswani et al., 2017).

  • Equal number of Q, K, V heads
  • RoPE support via pe_out.cos/sin
  • ALiBi support via pe_out.attn_bias
  • KV cache for autoregressive generation
  • Falls back to PyTorch SDPA (auto-selects Flash / Math kernel)

type key: "mha"
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.components.attention import BaseAttention, register
from engine.components.positional import PEOutput
from engine.components.positional.rope import apply_rope
from engine.config.schema import ModelConfig


@register("mha")
class MultiHeadAttention(BaseAttention):
    """
    Vanilla MHA — all heads see the full K/V set.

    LLaMA-style projection names are kept so PEFT LoRA requires no changes:
        q_proj  k_proj  v_proj  o_proj
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        attn = cfg.attention
        self.num_heads = attn.num_heads
        self.head_dim  = cfg.head_dim
        self.scale     = 1.0 / math.sqrt(self.head_dim)
        self.dropout   = attn.dropout

        d = cfg.hidden_size
        h = self.num_heads * self.head_dim

        # ── LLaMA-style names (PEFT target_modules works unchanged) ──────
        self.q_proj = nn.Linear(d, h, bias=False)
        self.k_proj = nn.Linear(d, h, bias=False)
        self.v_proj = nn.Linear(d, h, bias=False)
        self.o_proj = nn.Linear(h, d, bias=False)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _split(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, n_heads, head_dim).transpose(1, 2)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe_out: PEOutput,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, S, _ = hidden_states.shape

        q = self._split(self.q_proj(hidden_states), self.num_heads, self.head_dim)
        k = self._split(self.k_proj(hidden_states), self.num_heads, self.head_dim)
        v = self._split(self.v_proj(hidden_states), self.num_heads, self.head_dim)

        # ── Positional encoding ───────────────────────────────────────────
        if pe_out.cos is not None:
            cos, sin = pe_out.cos, pe_out.sin
            if past_key_value is not None:
                q, k = apply_rope(q, k, cos[:, :, -S:], sin[:, :, -S:])
            else:
                q, k = apply_rope(q, k, cos, sin)

        # ── KV cache ─────────────────────────────────────────────────────
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        # ── ALiBi bias ────────────────────────────────────────────────────
        attn_mask = attention_mask
        if pe_out.attn_bias is not None:
            bias = pe_out.attn_bias.to(q.dtype)
            attn_mask = (attn_mask + bias) if attn_mask is not None else bias

        # ── Scaled dot-product attention ──────────────────────────────────
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attn_mask is None),
            scale=self.scale,
        )   # (B, H, S, D)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), present
