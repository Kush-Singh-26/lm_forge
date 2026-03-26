"""
engine/config/hf_config.py

LMForgeConfig — a PretrainedConfig subclass that wraps ModelConfig.

This is the bridge between lm_forge's nested YAML config system and
HuggingFace's flat PretrainedConfig format.  It means:

  • AutoConfig.from_pretrained()           works
  • model.config.push_to_hub()            works
  • Trainer reads loss / architectures    works
  • config.json on the Hub is recognised  works

Design
──────
All ModelConfig fields are flattened into the PretrainedConfig namespace
(HF does not support nested configs natively).  Nesting is preserved with
double-underscore prefixes:
    model.attention.num_heads  →  attention__num_heads
    model.ffn.type             →  ffn__type

Round-trip:
    LMForgeConfig.from_model_config(model_cfg)  →  LMForgeConfig
    lmforge_cfg.to_model_config()               →  ModelConfig
"""

from __future__ import annotations

from typing import Any

from engine.config.schema import (
    ModelConfig,
    AttentionConfig,
    PositionalConfig,
    FFNConfig,
    NormConfig,
)

try:
    from transformers import PretrainedConfig

    class LMForgeConfig(PretrainedConfig):
        """
        HF-compatible config for all lm_forge CausalLM models.

        Usage::

            cfg = LMForgeConfig.from_model_config(model_cfg)
            cfg.save_pretrained("my_model/")

            # Load back
            cfg2 = LMForgeConfig.from_pretrained("my_model/")
            model_cfg = cfg2.to_model_config()
        """

        model_type = "lm_forge"

        def __init__(
            self,
            # ── shape ──────────────────────────────────────────────────
            vocab_size: int = 32_000,
            hidden_size: int = 512,
            num_layers: int = 6,
            max_seq_len: int = 2048,
            # ── attention ──────────────────────────────────────────────
            attention__type: str = "gqa",
            attention__num_heads: int = 8,
            attention__num_kv_heads: int = 8,
            attention__dropout: float = 0.0,
            attention__window_size: int = 512,
            # ── positional ─────────────────────────────────────────────
            positional__type: str = "rope",
            positional__theta: float = 10_000.0,
            positional__max_seq_len: int = 2048,
            # ── ffn ────────────────────────────────────────────────────
            ffn__type: str = "swiglu",
            ffn__intermediate_size: int = 1376,
            ffn__dropout: float = 0.0,
            ffn__bias: bool = False,
            # ── norm ───────────────────────────────────────────────────
            norm__type: str = "rms",
            norm__eps: float = 1e-5,
            # ── misc ───────────────────────────────────────────────────
            tie_word_embeddings: bool = False,
            initializer_range: float = 0.02,
            # ── HF required ────────────────────────────────────────────
            bos_token_id: int | None = None,
            eos_token_id: int | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )
            # shape
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.max_seq_len = max_seq_len
            self.initializer_range = initializer_range

            # attention
            self.attention__type = attention__type
            self.attention__num_heads = attention__num_heads
            self.attention__num_kv_heads = attention__num_kv_heads
            self.attention__dropout = attention__dropout
            self.attention__window_size = attention__window_size

            # positional
            self.positional__type = positional__type
            self.positional__theta = positional__theta
            self.positional__max_seq_len = positional__max_seq_len

            # ffn
            self.ffn__type = ffn__type
            self.ffn__intermediate_size = ffn__intermediate_size
            self.ffn__dropout = ffn__dropout
            self.ffn__bias = ffn__bias

            # norm
            self.norm__type = norm__type
            self.norm__eps = norm__eps

            # HF standard aliases for generation compatibility
            self.num_hidden_layers = num_layers
            self.num_attention_heads = attention__num_heads
            self.intermediate_size = ffn__intermediate_size

        # ── conversion helpers ────────────────────────────────────────

        @classmethod
        def from_model_config(cls, cfg: ModelConfig) -> "LMForgeConfig":
            """Wrap a lm_forge ModelConfig as an HF PretrainedConfig."""
            return cls(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                max_seq_len=cfg.max_seq_len,
                initializer_range=cfg.initializer_range,
                tie_word_embeddings=cfg.tie_word_embeddings,
                attention__type=cfg.attention.type,
                attention__num_heads=cfg.attention.num_heads,
                attention__num_kv_heads=cfg.attention.num_kv_heads,
                attention__dropout=cfg.attention.dropout,
                attention__window_size=cfg.attention.window_size,
                positional__type=cfg.positional.type,
                positional__theta=cfg.positional.theta,
                positional__max_seq_len=cfg.positional.max_seq_len,
                ffn__type=cfg.ffn.type,
                ffn__intermediate_size=cfg.ffn.intermediate_size,
                ffn__dropout=cfg.ffn.dropout,
                ffn__bias=cfg.ffn.bias,
                norm__type=cfg.norm.type,
                norm__eps=cfg.norm.eps,
            )

        def to_model_config(self) -> ModelConfig:
            """Reconstruct a lm_forge ModelConfig from this HF config."""
            return ModelConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                max_seq_len=self.max_seq_len,
                initializer_range=self.initializer_range,
                tie_word_embeddings=self.tie_word_embeddings,
                attention=AttentionConfig(
                    type=self.attention__type,
                    num_heads=self.attention__num_heads,
                    num_kv_heads=self.attention__num_kv_heads,
                    dropout=self.attention__dropout,
                    window_size=self.attention__window_size,
                ),
                positional=PositionalConfig(
                    type=self.positional__type,
                    theta=self.positional__theta,
                    max_seq_len=self.positional__max_seq_len,
                ),
                ffn=FFNConfig(
                    type=self.ffn__type,
                    intermediate_size=self.ffn__intermediate_size,
                    dropout=self.ffn__dropout,
                    bias=self.ffn__bias,
                ),
                norm=NormConfig(
                    type=self.norm__type,
                    eps=self.norm__eps,
                ),
            )

except ImportError:
    # transformers not installed — define a stub so imports don't explode
    class LMForgeConfig:  # type: ignore[no-redef]
        """Stub — install transformers to use HF features."""

        model_type = "lm_forge"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LMForgeConfig requires transformers. Run: pip install transformers"
            )
