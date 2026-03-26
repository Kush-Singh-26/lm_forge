"""
engine/components/ffn/__init__.py

Registry for all FFN (feed-forward network) implementations.

All FFN modules keep the LLaMA-style projection names:
    gate_proj, up_proj, down_proj  (for SwiGLU / GeGLU variants)
    fc1, fc2                        (for classic FFN — mapped to same PEFT keys)

type keys: "swiglu", "geglu", "classic"
"""

from __future__ import annotations

import torch.nn as nn
from engine.config.schema import ModelConfig

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def _inner(cls):
        _REGISTRY[name] = cls
        return cls
    return _inner


def build_ffn(cfg: ModelConfig) -> nn.Module:
    key = cfg.ffn.type
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown FFN type '{key}'. Available: {sorted(_REGISTRY)}"
        )
    return cls(cfg)


def list_ffn_types() -> list[str]:
    return sorted(_REGISTRY)


from engine.components.ffn import swiglu, classic  # noqa: E402,F401
