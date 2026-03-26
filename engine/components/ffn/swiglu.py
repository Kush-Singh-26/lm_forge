"""
engine/components/ffn/swiglu.py

SwiGLU (type key: "swiglu") and GeGLU (type key: "geglu") FFNs.

Both are gated linear units — a gate path and an up-projection are multiplied
element-wise, then projected down.  They consistently outperform the classic
two-layer FFN at the same parameter count.

    SwiGLU:  down( SiLU(gate(x)) ⊙ up(x) )
    GeGLU:   down( GELU(gate(x)) ⊙ up(x) )

LLaMA-style names (PEFT-safe): gate_proj, up_proj, down_proj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.components.ffn import register
from engine.config.schema import ModelConfig


class _GatedFFN(nn.Module):
    """Shared implementation for SwiGLU / GeGLU — only the activation differs."""

    def __init__(self, cfg: ModelConfig, act_fn) -> None:
        super().__init__()
        self.act_fn = act_fn
        d, h = cfg.hidden_size, cfg.ffn.intermediate_size

        # LLaMA-style names
        self.gate_proj = nn.Linear(d, h, bias=cfg.ffn.bias)
        self.up_proj   = nn.Linear(d, h, bias=cfg.ffn.bias)
        self.down_proj = nn.Linear(h, d, bias=cfg.ffn.bias)
        self.drop = nn.Dropout(cfg.ffn.dropout) if cfg.ffn.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


@register("swiglu")
class SwiGLUFFN(_GatedFFN):
    """SwiGLU — SiLU gate.  LLaMA / Mistral / Gemma default."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, F.silu)


@register("geglu")
class GeGLUFFN(_GatedFFN):
    """GeGLU — GELU gate.  PaLM variant."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, F.gelu)
