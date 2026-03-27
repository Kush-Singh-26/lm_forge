"""
engine/components/attention/gqa.py

Grouped-Query Attention (GQA) — Ainslie et al., 2023.
Also covers MQA (Multi-Query) when num_kv_heads = 1.

GQA reduces KV-cache memory and memory-bandwidth at inference by sharing
K/V heads across groups of Q heads.

    num_kv_heads = num_heads  → MHA   (same as mha.py, kept here for clarity)
    num_kv_heads = 1          → MQA   (extreme sharing, best KV-cache savings)
    1 < num_kv_heads < num_heads → GQA (LLaMA-2 70B, Mistral, Gemma default)

type keys: "gqa", "mqa"
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


@register("gqa")
class GroupedQueryAttention(BaseAttention):
    """
    GQA with full RoPE + ALiBi + KV-cache support.

    Flash Attention 2::

        pip install flash-attn --no-build-isolation
        # then set attention.flash_attn: true in config.yaml

    LLaMA-style names (PEFT-safe):
        q_proj, k_proj, v_proj, o_proj
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        attn = cfg.attention
        self.num_heads = attn.num_heads
        self.num_kv_heads = attn.num_kv_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = attn.dropout
        self.use_flash = getattr(attn, "flash_attn", False)

        # Validate flash_attn availability at construction time
        if self.use_flash:
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                raise ImportError(
                    "flash_attn not installed. "
                    "Run: pip install flash-attn --no-build-isolation\n"
                    "Or set attention.flash_attn: false in config.yaml."
                )

        d = cfg.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d, q_dim, bias=False)
        self.k_proj = nn.Linear(d, kv_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, d, bias=False)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _split(x: torch.Tensor, n: int, d: int) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, n, d).transpose(1, 2)

    @staticmethod
    def _expand_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
        """(B, H_kv, S, D) → (B, H_q, S, D) by repeating KV heads."""
        if groups == 1:
            return x
        B, H_kv, S, D = x.shape
        return (
            x.unsqueeze(2).expand(B, H_kv, groups, S, D).reshape(B, H_kv * groups, S, D)
        )

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
        k = self._split(self.k_proj(hidden_states), self.num_kv_heads, self.head_dim)
        v = self._split(self.v_proj(hidden_states), self.num_kv_heads, self.head_dim)

        # ── RoPE ─────────────────────────────────────────────────────────
        if pe_out.cos is not None:
            cos, sin = pe_out.cos, pe_out.sin
            offset_cos = cos[:, :, -S:] if past_key_value is not None else cos
            offset_sin = sin[:, :, -S:] if past_key_value is not None else sin
            q, k = apply_rope(q, k, offset_cos, offset_sin)

        # ── KV cache ─────────────────────────────────────────────────────
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        # ── GQA head expansion ────────────────────────────────────────────
        k_expanded = self._expand_kv(k, self.kv_groups)
        v_expanded = self._expand_kv(v, self.kv_groups)

        # ── ALiBi bias ────────────────────────────────────────────────────
        attn_mask = attention_mask
        if pe_out.attn_bias is not None:
            bias = pe_out.attn_bias.to(q.dtype)
            kv_len = k_expanded.shape[2]
            bias = bias[:, :, -S:, -kv_len:]
            attn_mask = (attn_mask + bias) if attn_mask is not None else bias

        # ── Attention kernel — Flash2 if available, else PyTorch SDPA ────
        # NOTE: FlashAttention 2 doesn't support arbitrary additive bias (ALiBi).
        # Fall back to SDPA if ALiBi is enabled.
        if self.use_flash and pe_out.attn_bias is None:
            from flash_attn import flash_attn_func

            q_fa = q.transpose(1, 2)
            k_fa = k_expanded.transpose(1, 2)
            v_fa = v_expanded.transpose(1, 2)
            out = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                dropout_p=self.dropout if self.training else 0.0,
                causal=(attn_mask is None),
                softmax_scale=self.scale,
            )
            out = out.reshape(B, S, -1)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k_expanded,
                v_expanded,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(attn_mask is None),
                scale=self.scale,
            )
            out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), present
