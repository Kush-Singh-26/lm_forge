"""
engine/models/base.py

BaseLM — root nn.Module for all engine models.

Provides save_pretrained / from_pretrained (HF-compatible layout),
num_parameters, and the post_init weight initialisation hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from engine.config.schema import ModelConfig


class BaseLM(nn.Module):
    config_class = ModelConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def post_init(self) -> None:
        """Call at end of __init__ in every concrete subclass."""
        self.apply(self._init_weights)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory)
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), directory / "model.safetensors")
        except ImportError:
            torch.save(self.state_dict(), directory / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        config: Optional[ModelConfig] = None,
        map_location: str | torch.device = "cpu",
    ) -> "BaseLM":
        import json
        from dataclasses import asdict
        path = Path(path)
        if config is None:
            raw = json.loads((path / "model_config.json").read_text())
            # Rebuild nested dataclasses
            from engine.config.schema import (
                ModelConfig, AttentionConfig, PositionalConfig, FFNConfig, NormConfig
            )
            raw["attention"]  = AttentionConfig(**{k: v for k, v in raw.get("attention",{}).items()
                                                    if k in AttentionConfig.__dataclass_fields__})
            raw["positional"] = PositionalConfig(**{k: v for k, v in raw.get("positional",{}).items()
                                                     if k in PositionalConfig.__dataclass_fields__})
            raw["ffn"]  = FFNConfig(**{k: v for k, v in raw.get("ffn",{}).items()
                                        if k in FFNConfig.__dataclass_fields__})
            raw["norm"] = NormConfig(**{k: v for k, v in raw.get("norm",{}).items()
                                         if k in NormConfig.__dataclass_fields__})
            config = ModelConfig(**{k: v for k, v in raw.items()
                                     if k in ModelConfig.__dataclass_fields__})

        model = cls(config)

        sf = path / "model.safetensors"
        pt = path / "pytorch_model.bin"
        if sf.exists():
            from safetensors.torch import load_file
            state = load_file(str(sf), device=str(map_location))
        elif pt.exists():
            state = torch.load(pt, map_location=map_location, weights_only=True)
        else:
            raise FileNotFoundError(f"No weights in {path}")

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[BaseLM] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[BaseLM] Unexpected keys ({len(unexpected)})")
        return model
