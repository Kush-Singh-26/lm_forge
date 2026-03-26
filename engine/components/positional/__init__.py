"""
engine/components/positional/__init__.py

Registry + PEOutput dataclass for all positional encoding strategies.

Each PE module produces a PEOutput.  The DecoderLayer reads the fields it
needs and passes them downstream — attention modules check which fields are
populated and apply the appropriate mechanism.

    PEOutput.hidden_states  → Learned absolute: add to embeddings before attn
    PEOutput.cos / .sin     → RoPE: rotate q and k inside attention
    PEOutput.attn_bias      → ALiBi: add to attention logits

This keeps PE logic out of attention modules while keeping all PE work in
one place per layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from engine.config.schema import PositionalConfig

# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class PEOutput:
    """
    Container for whatever the chosen PE strategy produces.
    Attention modules inspect only the fields that are set (not None).
    """
    hidden_states: Optional[torch.Tensor] = None   # modified input (learned PE)
    cos: Optional[torch.Tensor] = None             # RoPE cosine  (B,1,S,D)
    sin: Optional[torch.Tensor] = None             # RoPE sine    (B,1,S,D)
    attn_bias: Optional[torch.Tensor] = None       # ALiBi bias   (B,H,S,S)


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator — register a PE class under a string key."""
    def _inner(cls):
        _REGISTRY[name] = cls
        return cls
    return _inner


def build_pe(config: PositionalConfig, hidden_size: int, num_heads: int) -> nn.Module:
    """
    Instantiate the positional encoding module specified by config.positional.type.

    Args:
        config      : PositionalConfig from the experiment YAML.
        hidden_size : Model dimension (needed by learned PE).
        num_heads   : Number of attention heads (needed by ALiBi).
    """
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unknown positional encoding type '{config.type}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return cls(config, hidden_size=hidden_size, num_heads=num_heads)


def list_pe_types() -> list[str]:
    return sorted(_REGISTRY)


# ── Auto-import all PE modules so decorators fire ────────────────────────────

from engine.components.positional import rope, alibi, learned, nope  # noqa: E402,F401
