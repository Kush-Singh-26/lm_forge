"""
engine/models/encoder.py

Encoder-only Masked Language Model (BERT / RoBERTa style).

Uses the same component registries as the decoder, so all attention types,
PE types, FFN types, and norm types work identically — just configured for
bidirectional (non-causal) attention.

Module tree (BERT-compatible naming for PEFT):

    MaskedLM
    ├── encoder                     ← EncoderModel
    │   ├── embeddings
    │   │   ├── word_embeddings     ← nn.Embedding
    │   │   └── pe                  ← any PE module
    │   ├── layers[i]               ← EncoderLayer
    │   │   ├── attention
    │   │   │   ├── self            ← BertSelfAttention (q/k/v/o_proj names)
    │   │   │   └── output.LayerNorm
    │   │   └── intermediate + output (FFN + norm)
    │   └── norm
    ├── cls_head                    ← MLM prediction head
    │   ├── dense
    │   ├── act
    │   └── decoder                 ← Linear vocab projection
    └── (optional) pooler           ← CLS pooler for classification

Key difference from CausalLM:
  • Bidirectional attention — no causal mask applied
  • MLM loss — cross-entropy only on [MASK] positions
  • CLS token pooler for downstream tasks (sentence embeddings, classification)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.config.schema import ModelConfig
from engine.components.norm import build_norm
from engine.components.positional import build_pe
from engine.components.attention import build_attention
from engine.components.ffn import build_ffn
from engine.models.base import BaseLM


# ─────────────────────────────────────────────────────────────────────────────
# Single encoder layer
# ─────────────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Pre-norm encoder layer — same as DecoderLayer but no causal masking.

    Attribute names use BERT convention for PEFT compat:
        attention.self  → contains q_proj / k_proj / v_proj / o_proj
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Wrap self_attn inside an 'attention.self' namespace so that
        # PEFT target_modules=["attention.self.q_proj"] works.
        class _AttnWrapper(nn.Module):
            def __init__(self_, cfg):
                super().__init__()
                self_.self = build_attention(cfg)
        self.attention = _AttnWrapper(cfg)

        self.input_layernorm          = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.post_attention_layernorm = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.mlp = build_ffn(cfg)

    @property
    def self_attn(self):
        """Alias so code that addresses self_attn directly still works."""
        return self.attention.self

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe_out,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Bidirectional attention — pass attention_mask but NOT causal
        residual = hidden_states
        attn_out, _ = self.attention.self(
            self.input_layernorm(hidden_states),
            pe_out=pe_out,
            attention_mask=attention_mask,
            past_key_value=None,
            use_cache=False,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Embedding module
# ─────────────────────────────────────────────────────────────────────────────

class EncoderEmbeddings(nn.Module):
    """Word embeddings + optional positional embeddings for the encoder."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pe = build_pe(cfg.positional, hidden_size=cfg.hidden_size,
                           num_heads=cfg.attention.num_heads)
        if hasattr(self.pe, "set_head_dim"):
            self.pe.set_head_dim(cfg.head_dim)
        self.norm = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.drop = nn.Dropout(cfg.attention.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, object]:
        hidden = self.word_embeddings(input_ids)
        pe_out = self.pe(hidden, seq_len=input_ids.shape[1], position_ids=position_ids)
        if pe_out.hidden_states is not None:
            hidden = pe_out.hidden_states
        return self.drop(self.norm(hidden)), pe_out


# ─────────────────────────────────────────────────────────────────────────────
# Trunk
# ─────────────────────────────────────────────────────────────────────────────

class EncoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embeddings = EncoderEmbeddings(cfg)
        self.layers     = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm       = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        hidden, pe_out = self.embeddings(input_ids, position_ids)

        # Build additive padding mask  (0 = masked, 1 = real)
        if attention_mask is not None:
            additive = torch.zeros(B, 1, S, S, dtype=hidden.dtype, device=hidden.device)
            pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            additive = additive.masked_fill(pad, float("-inf"))
        else:
            additive = None

        for layer in self.layers:
            hidden = layer(hidden, pe_out=pe_out, attention_mask=additive)

        return self.norm(hidden)


# ─────────────────────────────────────────────────────────────────────────────
# MLM prediction head
# ─────────────────────────────────────────────────────────────────────────────

class MLMHead(nn.Module):
    """
    Standard BERT MLM head:
        dense → GELU → norm → linear (vocab)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.dense   = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.act     = nn.GELU()
        self.norm    = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(hidden))))


# ─────────────────────────────────────────────────────────────────────────────
# Full masked LM
# ─────────────────────────────────────────────────────────────────────────────

class MaskedLM(BaseLM):
    """
    Encoder-only masked language model.

    The attention type is set to bidirectional by NOT passing is_causal=True —
    the attention modules already handle this because they only apply the
    causal flag when attention_mask is None and use_cache is False.

    For true bidirectional attention you should pass attention_mask=ones
    (or None) so that SDPA's is_causal=False path is taken.

    Usage::

        cfg   = ModelConfig(attention=AttentionConfig(type="mha"), ...)
        model = MaskedLM(cfg)

        # Forward with MLM collated batch
        logits, loss = model(input_ids, labels=labels, attention_mask=mask)

        # Sentence embeddings (CLS pooling)
        embeddings = model.encode(input_ids, attention_mask=mask)

    PEFT::

        lora = LoraConfig(target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, lora)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.encoder  = EncoderModel(cfg)
        self.cls_head = MLMHead(cfg)
        self.pooler   = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # CLS pooler

        if cfg.tie_word_embeddings:
            self.cls_head.decoder.weight = self.encoder.embeddings.word_embeddings.weight

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids      : (B, S) — tokens, [MASK] positions set by collator.
            labels         : (B, S) — -100 everywhere except masked positions.
            attention_mask : (B, S) — 1=real token, 0=pad.

        Returns:
            (logits, loss)  — loss is None when labels are not provided.
        """
        hidden = self.encoder(input_ids, attention_mask, position_ids)  # (B, S, D)
        logits = self.cls_head(hidden)                                   # (B, S, V)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pool: str = "cls",
    ) -> torch.Tensor:
        """
        Produce sentence embeddings.

        Args:
            input_ids      : (B, S)
            attention_mask : (B, S)
            pool           : "cls" = first token  |  "mean" = masked mean pooling

        Returns:
            (B, hidden_size) embeddings, L2-normalised.
        """
        hidden = self.encoder(input_ids, attention_mask)  # (B, S, D)

        if pool == "cls":
            vec = torch.tanh(self.pooler(hidden[:, 0]))   # (B, D)
        elif pool == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                vec = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            else:
                vec = hidden.mean(dim=1)
        else:
            raise ValueError(f"pool must be 'cls' or 'mean', got '{pool}'")

        return F.normalize(vec, p=2, dim=-1)
