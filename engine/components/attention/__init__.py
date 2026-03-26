"""
engine/components/attention/__init__.py

Registry + BaseAttention for all attention implementations.

Every attention class:
  1. Inherits BaseAttention (which is just nn.Module + documented interface)
  2. Decorates itself with @register("key")
  3. Uses the standard projection names: q_proj, k_proj, v_proj, o_proj
     (LLaMA naming → PEFT LoRA works out of the box)

The forward signature is uniform across all implementations:

    forward(
        hidden_states,
        pe_out,          ← PEOutput from the PE module
        attention_mask,  ← (B,1,S,S) additive mask or None
        past_key_value,  ← (k, v) KV-cache tuple or None
        use_cache,
    ) → (output, present_key_value)
"""

from __future__ import annotations

import torch.nn as nn

from engine.config.schema import AttentionConfig, ModelConfig

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator — register an attention class under a string key."""
    def _inner(cls):
        _REGISTRY[name] = cls
        return cls
    return _inner


class BaseAttention(nn.Module):
    """
    Minimal base class that documents the expected interface.

    Subclasses MUST keep these attribute names for PEFT / LoRA compat:
        self.q_proj
        self.k_proj
        self.v_proj
        self.o_proj
    """

    def forward(self, hidden_states, pe_out, attention_mask=None,
                past_key_value=None, use_cache=False):
        raise NotImplementedError


def build_attention(model_cfg: ModelConfig) -> nn.Module:
    """
    Instantiate the attention module specified by model_cfg.attention.type.

    Passes the full ModelConfig so implementations can read head_dim, etc.
    """
    key = model_cfg.attention.type
    # "mqa" is GQA with num_kv_heads=1 — reuse GQA implementation
    lookup = "gqa" if key == "mqa" else key
    cls = _REGISTRY.get(lookup)
    if cls is None:
        raise ValueError(
            f"Unknown attention type '{key}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return cls(model_cfg)


def list_attention_types() -> list[str]:
    return sorted(_REGISTRY)


# Auto-import so @register decorators fire
from engine.components.attention import mha, gqa, sliding  # noqa: E402,F401
