"""
engine/components/attention/gqa.py

Grouped-Query Attention (GQA) — Ainslie et al., 2023.
Also covers MQA (Multi-Query) when num_kv_heads = 1.
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

        if self.use_flash:
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                raise ImportError(
                    "flash_attn not installed. "
                    "Run: pip install flash-attn --no-build-isolation"
                )

        d = cfg.hidden_size
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(d, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, d, bias=False)

    @staticmethod
    def _split(x: torch.Tensor, n: int, d: int) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, n, d).transpose(1, 2)

    @staticmethod
    def _expand_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
        if groups == 1:
            return x
        B, H_kv, S, D = x.shape
        # use expand + reshape. Note: reshape might trigger a copy,
        # but is necessary for SDPA which expects (B, H, S, D).
        return (
            x.unsqueeze(2).expand(B, H_kv, groups, S, D).reshape(B, H_kv * groups, S, D)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe_out: PEOutput,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, S, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = self._split(q, self.num_heads, self.head_dim)
        k = self._split(k, self.num_kv_heads, self.head_dim)
        v = self._split(v, self.num_kv_heads, self.head_dim)

        # ── RoPE ─────────────────────────────────────────────────────────
        if pe_out.cos is not None:
            cos, sin = pe_out.cos, pe_out.sin
            # For incremental decoding, S=1. We need the cos/sin at the correct position.
            if past_key_value is not None:
                # Determine current length for RoPE offset
                if hasattr(past_key_value, "get_seq_length"):
                    kv_seq_len = past_key_value.get_seq_length(layer_idx)
                else:
                    kv_seq_len = past_key_value[0].shape[2]

                if cos.shape[2] >= kv_seq_len + S:
                    offset_cos = cos[:, :, kv_seq_len : kv_seq_len + S]
                    offset_sin = sin[:, :, kv_seq_len : kv_seq_len + S]
                else:
                    offset_cos = cos
                    offset_sin = sin
                q, k = apply_rope(q, k, offset_cos, offset_sin)
            else:
                if cos.shape[2] >= S:
                    q, k = apply_rope(q, k, cos[:, :, :S], sin[:, :, :S])
                else:
                    q, k = apply_rope(q, k, cos, sin)

        # ── KV cache ─────────────────────────────────────────────────────
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                k, v = past_key_value.update(k, v, layer_idx)
            else:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        # ── ALiBi bias ────────────────────────────────────────────────────
        attn_mask = attention_mask
        if pe_out.attn_bias is not None:
            bias = pe_out.attn_bias.to(q.dtype)
            kv_len = k.shape[2]
            if bias.shape[2] > S or bias.shape[3] > kv_len:
                bias = bias[:, :, -S:, -kv_len:]
            attn_mask = (attn_mask + bias) if attn_mask is not None else bias

        if self.use_flash and pe_out.attn_bias is None:
            from flash_attn import flash_attn_func

            # Flash Attention natively supports GQA by broadcasting KV heads.
            # We pass q, k, v in (B, S, H, D) format.
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)

            out = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                dropout_p=self.dropout if self.training else 0.0,
                causal=(attn_mask is None and S > 1),
                softmax_scale=self.scale,
            )
            out = out.reshape(B, S, -1)
        else:
            # GQA head expansion for SDPA
            k_expanded = self._expand_kv(k, self.kv_groups)
            v_expanded = self._expand_kv(v, self.kv_groups)

            out = F.scaled_dot_product_attention(
                q,
                k_expanded,
                v_expanded,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(attn_mask is None and S > 1),
                scale=self.scale,
            )
            out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), present
        return self.o_proj(out), present
