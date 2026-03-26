"""
engine/components/ffn/classic.py

Classic two-layer GELU FFN (GPT-2 / BERT style).

    FFN(x) = W2( GELU( W1(x) ) )

Uses fc1 / fc2 attribute names (no gate projection needed).
Also registered under gate_proj alias so PEFT configs that target
gate_proj still work — the alias maps to fc1.

type key: "classic"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.components.ffn import register
from engine.config.schema import ModelConfig


@register("classic")
class ClassicFFN(nn.Module):
    """
    Standard two-layer FFN.

    Attribute names:
        fc1  / gate_proj  → both reference the same first linear layer
        fc2  / down_proj  → both reference the same second linear layer
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h = cfg.hidden_size, cfg.ffn.intermediate_size

        self.fc1 = nn.Linear(d, h, bias=cfg.ffn.bias)
        self.fc2 = nn.Linear(h, d, bias=cfg.ffn.bias)
        self.drop = nn.Dropout(cfg.ffn.dropout) if cfg.ffn.dropout > 0 else nn.Identity()

        # PEFT aliases — point to the same objects so target_modules="gate_proj"
        # or "down_proj" still hit the right weights
        self.gate_proj = self.fc1
        self.down_proj = self.fc2
        # up_proj doesn't exist in classic FFN; set to None to signal that
        self.up_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
