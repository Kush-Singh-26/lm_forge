"""
engine/components/norm/norms.py

RMSNorm and LayerNorm with a build_norm() factory.

Identical interface — swap by changing config.norm.type = "rms" | "layer".
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Norm (Zhang & Sennrich, 2019).
    Drops the mean-centering step → ~half the compute of LayerNorm.
    Used by LLaMA, Mistral, Gemma.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused native RMSNorm if available (PyTorch 2.4+)
        if hasattr(nn.functional, "rms_norm"):
            return nn.functional.rms_norm(
                x, (self.weight.shape[0],), self.weight, self.eps
            ).to(x.dtype)

        # Fallback with careful upcasting to avoid full FP32 tensor allocation if possible
        dtype = x.dtype
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(dtype)
        return (self.weight.to(dtype) * x) * rms


class LayerNorm(nn.Module):
    """Standard Layer Norm with optional bias (BERT / GPT-2 style)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.eps = eps
        self._shape = (hidden_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(
            x, self._shape, self.weight, self.bias, self.eps
        )


def build_norm(norm_type: str, hidden_size: int, eps: float = 1e-5) -> nn.Module:
    """Factory — matches config.norm.type."""
    match norm_type:
        case "rms":
            return RMSNorm(hidden_size, eps)
        case "layer":
            return LayerNorm(hidden_size, eps)
        case _:
            raise ValueError(
                f"Unknown norm type '{norm_type}'. Choose 'rms' or 'layer'."
            )
