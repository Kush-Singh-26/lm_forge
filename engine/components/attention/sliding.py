"""
engine/components/attention/sliding.py

Sliding Window Attention — Beltagy et al. (Longformer), Jiang et al. (Mistral).

Each token only attends to the W most recent tokens (+ itself), where W is
the window size.  This reduces attention complexity from O(S²) to O(S·W),
enabling much longer sequences at the same compute budget.

Implementation strategy:
  • For training (full sequences): mask out tokens outside the window by
    setting their attention logits to -inf before softmax.
  • For inference (with KV cache): the cache is naturally bounded because
    we only keep the last W positions.

This version uses GQA as the underlying attention mechanism.
type key: "sliding"
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


def _sliding_window_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """
    Build an additive (S, S) mask where positions outside the window are -inf.

    Position j is visible from position i if  i - window < j <= i.
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    # keep: j > i - window  AND  j <= i  (causal constraint)
    in_window = (j > i - window) & (j <= i)
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(~in_window, float("-inf"))
    return mask   # (S, S)


@register("sliding")
class SlidingWindowAttention(BaseAttention):
    """
    GQA-based attention with a local sliding window.

    LLaMA-style names (PEFT-safe):
        q_proj, k_proj, v_proj, o_proj
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        attn = cfg.attention
        self.num_heads    = attn.num_heads
        self.num_kv_heads = attn.num_kv_heads
        self.kv_groups    = self.num_heads // self.num_kv_heads
        self.head_dim     = cfg.head_dim
        self.window_size  = attn.window_size
        self.scale        = 1.0 / math.sqrt(self.head_dim)
        self.dropout      = attn.dropout

        d = cfg.hidden_size
        q_dim  = self.num_heads    * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d, q_dim,  bias=False)
        self.k_proj = nn.Linear(d, kv_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, d,  bias=False)

    @staticmethod
    def _split(x: torch.Tensor, n: int, d: int) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, n, d).transpose(1, 2)

    @staticmethod
    def _expand_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
        if groups == 1:
            return x
        B, H_kv, S, D = x.shape
        return x.unsqueeze(2).expand(B, H_kv, groups, S, D).reshape(B, H_kv * groups, S, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe_out: PEOutput,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, S, _ = hidden_states.shape

        q = self._split(self.q_proj(hidden_states), self.num_heads,    self.head_dim)
        k = self._split(self.k_proj(hidden_states), self.num_kv_heads, self.head_dim)
        v = self._split(self.v_proj(hidden_states), self.num_kv_heads, self.head_dim)

        # ── RoPE ─────────────────────────────────────────────────────────
        if pe_out.cos is not None:
            cos, sin = pe_out.cos, pe_out.sin
            offset_cos = cos[:, :, -S:] if past_key_value is not None else cos
            offset_sin = sin[:, :, -S:] if past_key_value is not None else sin
            q, k = apply_rope(q, k, offset_cos, offset_sin)

        # ── KV cache (bounded to window_size) ────────────────────────────
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
            # Evict oldest entries beyond window
            if k.shape[2] > self.window_size:
                k = k[:, :, -self.window_size:]
                v = v[:, :, -self.window_size:]
        present = (k, v) if use_cache else None

        k_exp = self._expand_kv(k, self.kv_groups)
        v_exp = self._expand_kv(v, self.kv_groups)

        # ── Sliding window mask (training only; inference uses bounded cache) ─
        kv_len = k_exp.shape[2]
        win_mask = _sliding_window_mask(kv_len, self.window_size, hidden_states.device)
        win_mask = win_mask[-S:, :]    # for the current query positions
        win_mask = win_mask.unsqueeze(0).unsqueeze(0).to(q.dtype)  # (1,1,S,S_kv)

        # Merge with padding mask if provided
        if attention_mask is not None:
            win_mask = win_mask + attention_mask

        # ── ALiBi bias ────────────────────────────────────────────────────
        if pe_out.attn_bias is not None:
            bias = pe_out.attn_bias.to(q.dtype)
            kv_len = k_exp.shape[2]
            bias = bias[:, :, -S:, -kv_len:]
            win_mask = win_mask + bias

        out = F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=win_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,   # mask already handles causality
            scale=self.scale,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), present
