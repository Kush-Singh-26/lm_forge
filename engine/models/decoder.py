"""
engine/models/decoder.py

Decoder-only Causal LM — assembled from registry components.

The model tree is identical to LLaMA regardless of which attention / PE / FFN
combination is chosen, because all sub-modules use the same attribute names.

    CausalLM
    ├── model                       ← DecoderModel
    │   ├── embed_tokens
    │   ├── pe                      ← any BasePE (rope/alibi/learned/none)
    │   ├── layers[i]               ← DecoderLayer
    │   │   ├── input_layernorm
    │   │   ├── self_attn           ← any BaseAttention (mha/gqa/mqa/sliding)
    │   │   │   ├── q_proj
    │   │   │   ├── k_proj
    │   │   │   ├── v_proj
    │   │   │   └── o_proj
    │   │   ├── post_attention_layernorm
    │   │   └── mlp                 ← any FFN (swiglu/geglu/classic)
    │   │       ├── gate_proj / fc1
    │   │       ├── up_proj
    │   │       └── down_proj / fc2
    │   └── norm
    └── lm_head

PEFT LoRA with default LLaMA target_modules still works because projection
names never change.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from engine.config.schema import ModelConfig
from engine.components.norm import build_norm
from engine.components.positional import build_pe
from engine.components.attention import build_attention
from engine.components.ffn import build_ffn
from engine.models.base import BaseLM


# ─────────────────────────────────────────────────────────────────────────────
# Single layer
# ─────────────────────────────────────────────────────────────────────────────


class DecoderLayer(nn.Module):
    """
    Pre-norm transformer layer:  attn → residual → FFN → residual.

    Named to match LLaMA for PEFT.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Norms
        self.input_layernorm = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.post_attention_layernorm = build_norm(
            cfg.norm.type, cfg.hidden_size, cfg.norm.eps
        )
        # Attention (any registered type)
        self.self_attn = build_attention(cfg)
        # FFN (any registered type)
        self.mlp = build_ffn(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe_out,  # PEOutput
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden_states),
            pe_out=pe_out,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present


# ─────────────────────────────────────────────────────────────────────────────
# Trunk (no LM head)
# ─────────────────────────────────────────────────────────────────────────────


class DecoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        # Positional encoding — shared across all layers
        self.pe = build_pe(
            cfg.positional,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.attention.num_heads,
        )

        # Wire head_dim into RoPE after it's been computed by ModelConfig
        if hasattr(self.pe, "set_head_dim"):
            self.pe.set_head_dim(cfg.head_dim)

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.gradient_checkpointing = False
        self._causal_mask_cache: dict = {}

    def reset_cache(self) -> None:
        """Clear cached causal masks (e.g. after device change)."""
        self._causal_mask_cache.clear()

    def _get_causal_mask(
        self, S: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or build a cached causal mask."""
        if len(self._causal_mask_cache) > 16:
            self._causal_mask_cache.clear()
        norm_device = str(torch.device(device.type, device.index or 0))
        key = (S, dtype, norm_device)
        if key not in self._causal_mask_cache:
            mask = torch.full(
                (S, S),
                torch.finfo(dtype).min / 2,
                device=device,
                dtype=dtype,
            )
            mask = torch.triu(mask, diagonal=1)
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def enable_gradient_checkpointing(self) -> None:
        """
        Recompute activations during backward instead of storing them.
        Cuts peak memory ~sqrt(L) at ~33% extra compute cost.
        Essential for training on Colab T4 with seq_len >= 512.
        """
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        B, S = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # ── Compute actual sequence length (including KV cache) ────────────
        kv_len = S
        if past_key_values is not None:
            kv_len += past_key_values[0][0].shape[2]

        # PE — may modify hidden_states or produce cos/sin/bias
        pe_out = self.pe(hidden_states, seq_len=kv_len, position_ids=position_ids)
        if pe_out.hidden_states is not None:
            hidden_states = pe_out.hidden_states

        # ── Masking ───────────────────────────────────────────────────────
        # Every step in SDPA needs a (B, 1, S, S) additive mask.
        # It must combine:
        #  1. Causal mask (if S > 1 and not incremental decoding)
        #  2. Padding mask (if provided)
        #  3. Positional bias (ALiBi) is added later inside the attention module

        additive = None
        if S > 1 and past_key_values is None:
            # Causal mask: upper triangle is -inf (cached for performance)
            additive = self._get_causal_mask(
                S, hidden_states.device, hidden_states.dtype
            )
            additive = additive.view(1, 1, S, S)

        if attention_mask is not None:
            # Padding mask: 0 in attention_mask means -inf
            kv_len = attention_mask.shape[-1]
            pad_mask = (attention_mask == 0).view(B, 1, 1, kv_len)
            fill_value = torch.finfo(hidden_states.dtype).min / 2
            if additive is not None:
                additive = additive.masked_fill(pad_mask, fill_value)
            else:
                additive = torch.zeros(
                    B,
                    1,
                    S,
                    kv_len,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                additive = additive.masked_fill(pad_mask, fill_value)
        # ──────────────────────────────────────────────────────────────────

        presents: list = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training and pkv is None:
                # torch.utils.checkpoint rewrites the forward graph so that
                # intermediate activations are not stored — they're recomputed
                # during backward instead.  use_cache must be False.
                pe_out_captured = pe_out

                def _inner(hs, mask, layer=layer, pe_out_captured=pe_out_captured):
                    out, _ = layer(
                        hs,
                        pe_out=pe_out_captured,
                        attention_mask=mask,
                        past_key_value=None,
                        use_cache=False,
                    )
                    return out

                hidden_states = torch.utils.checkpoint.checkpoint(
                    _inner,
                    hidden_states,
                    additive,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = layer(
                    hidden_states,
                    pe_out=pe_out,
                    attention_mask=additive,
                    past_key_value=pkv,
                    use_cache=use_cache,
                )
            if use_cache:
                presents.append(present)

        return self.norm(hidden_states), presents if use_cache else None


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────


class CausalLM(BaseLM):
    """
    Decoder-only LM.  The attention / PE / FFN types are fully determined by
    the ModelConfig — no code changes required to switch variants.

    PEFT::

        lora = LoraConfig(target_modules=["q_proj", "v_proj"])  # LLaMA defaults
        model = get_peft_model(model, lora)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.model = DecoderModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        hidden, presents = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache
        )
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, presents

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self.eval()
        past: list = []
        for _ in range(max_new_tokens):
            cur = input_ids if not past else input_ids[:, -1:]
            logits, _, past = self(cur, past_key_values=past or None, use_cache=True)
            next_logits = logits[:, -1] / max(temperature, 1e-8)
            if top_k > 0:
                vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < vals[:, -1:]] = float("-inf")
            tok = torch.multinomial(
                torch.softmax(next_logits, -1), 1, generator=generator
            )
            input_ids = torch.cat([input_ids, tok], -1)
            if eos_token_id is not None and (tok == eos_token_id).all():
                break
        return input_ids
