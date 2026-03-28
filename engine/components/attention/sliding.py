"""
engine/components/attention/sliding.py

Sliding Window Attention — Beltagy et al. (Longformer), Jiang et al. (Mistral).
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


def _sliding_window_mask(
    seq_len: int, window: int, device: torch.device
) -> torch.Tensor:
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    in_window = (j > i - window) & (j <= i)
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(~in_window, float("-inf"))
    return mask


@register("sliding")
class SlidingWindowAttention(BaseAttention):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        attn = cfg.attention
        self.num_heads = attn.num_heads
        self.num_kv_heads = attn.num_kv_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.head_dim
        self.window_size = attn.window_size
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
            if past_key_value is not None:
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
                # transformers.Cache (v5 compatible)
                k, v = past_key_value.update(k, v, layer_idx)

                # Manual eviction for standard Cache objects that don't handle sliding windows
                # (Standard DynamicCache just keeps growing).
                if k.shape[2] > self.window_size:
                    # Note: Slicing a Cache object is tricky, but here we only
                    # care about the tensors for the current forward pass.
                    k = k[:, :, -self.window_size :]
                    v = v[:, :, -self.window_size :]
            else:
                # legacy tuple (k, v)
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
                # Evict oldest entries beyond window
                if k.shape[2] > self.window_size:
                    k = k[:, :, -self.window_size :]
                    v = v[:, :, -self.window_size :]
        present = (k, v) if use_cache else None

        if self.use_flash and pe_out.attn_bias is None:
            from flash_attn import flash_attn_func

            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)

            out = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                window_size=(self.window_size, self.window_size),
                softmax_scale=self.scale,
            )
            out = out.reshape(B, S, -1)
        else:
            k_exp = self._expand_kv(k, self.kv_groups)
            v_exp = self._expand_kv(v, self.kv_groups)

            # ── Sliding window mask ──────────────────────────────────────────
            kv_len = k_exp.shape[2]
            # Use the mask cached in the model if available (passed through attention_mask)
            # or build a local one (fallback).
            if attention_mask is not None and attention_mask.shape[-2:] == (S, kv_len):
                win_mask = attention_mask
            else:
                win_mask = _sliding_window_mask(
                    kv_len, self.window_size, hidden_states.device
                )
                win_mask = win_mask[-S:, :]
                win_mask = win_mask.unsqueeze(0).unsqueeze(0).to(q.dtype)

            if pe_out.attn_bias is not None:
                bias = pe_out.attn_bias.to(q.dtype)
                if bias.shape[2] > S or bias.shape[3] > kv_len:
                    bias = bias[:, :, -S:, -kv_len:]
                win_mask = win_mask + bias

            out = F.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=win_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
            out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), present
        return self.o_proj(out), present
