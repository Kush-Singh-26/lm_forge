"""
engine/training/device.py

DeviceManager — owns all device/dtype decisions.

One object, one place.  Model code and training loops never contain
if cuda / if cpu branches — they just call dm.prepare() and dm.to_device().
"""

from __future__ import annotations
from typing import Union
import torch
import torch.nn as nn

from engine.config.schema import TrainConfig


class DeviceManager:
    """
    Resolves backend + dtype from a TrainConfig, prepares models,
    moves batches, and wraps forward passes in autocast.

    CPU → GPU migration:
        Change training.backend = "cuda" and training.dtype = "bfloat16"
        in config.yaml.  Zero code changes.
    """

    _DTYPE = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg    = cfg
        self.device = self._resolve_device()
        self.dtype  = self._resolve_dtype()

    # ── resolution ────────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        b = self.cfg.backend
        if b == "auto":
            if torch.cuda.is_available():    b = "cuda"
            elif torch.backends.mps.is_available(): b = "mps"
            else:                            b = "cpu"
        if b == "cuda":
            if not torch.cuda.is_available():
                print("[DeviceManager] CUDA not available → CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        if b == "mps":
            if not torch.backends.mps.is_available():
                print("[DeviceManager] MPS not available → CPU")
                return torch.device("cpu")
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self) -> torch.dtype:
        dtype = self._DTYPE.get(self.cfg.dtype, torch.float32)
        # fp16 training on CPU is unsupported
        if self.device.type == "cpu" and dtype == torch.float16:
            print("[DeviceManager] float16 on CPU → switching to float32")
            return torch.float32
        return dtype

    # ── public API ────────────────────────────────────────────────────────

    def prepare(self, model: nn.Module) -> nn.Module:
        """Move to device + dtype, optionally torch.compile."""
        model = model.to(device=self.device, dtype=self.dtype)
        if self.cfg.compile:
            if hasattr(torch, "compile"):
                print("[DeviceManager] torch.compile() ...")
                model = torch.compile(model)
            else:
                print("[DeviceManager] torch.compile unavailable (needs PyTorch ≥ 2.0)")
        return model

    def to_device(self, batch: Union[dict, torch.Tensor]) -> Union[dict, torch.Tensor]:
        """Recursively move a batch dict or tensor to self.device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        return batch

    def autocast(self):
        """Context manager for mixed-precision forward passes."""
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=self.dtype)
        if self.device.type == "mps":
            return torch.autocast("mps", dtype=self.dtype)
        return torch.autocast("cpu", dtype=torch.float32, enabled=False)

    def grad_scaler(self) -> torch.cuda.amp.GradScaler | None:
        """Return a GradScaler for fp16/CUDA; None otherwise."""
        if self.device.type == "cuda" and self.dtype == torch.float16:
            return torch.cuda.amp.GradScaler()
        return None

    def build_optimizer(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
        fused: bool = True,
    ) -> torch.optim.AdamW:
        """
        Build AdamW with fused=True on CUDA for ~5-10% speedup.

        Fused AdamW launches a single CUDA kernel per parameter group
        instead of separate kernels for each operation (exp avg, sq avg,
        bias correction, param update). The gain is small per step but
        compounds over 10k+ steps.

        Automatically falls back to standard AdamW on CPU/MPS where
        the fused kernel is not available.

        Usage::

            optimizer = dm.build_optimizer(model, lr=exp.training.lr,
                                           fused=exp.training.fused_adamw)
        """
        use_fused = fused and self.device.type == "cuda"

        # Separate weight-decayed params from non-decayed (norms, biases,
        # embeddings). This is the standard transformer optimizer setup.
        decay_params     = []
        no_decay_params  = []
        no_decay_names   = {"bias", "weight"}   # norm weight also skipped below

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Skip 1-D params (norms, biases) and embedding tables
            if param.ndim == 1 or "embed_tokens" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if use_fused:
            try:
                opt = torch.optim.AdamW(
                    param_groups, lr=lr, betas=betas, fused=True
                )
                return opt
            except (TypeError, RuntimeError):
                # fused= not supported on this PyTorch version
                pass

        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)

    def summary(self) -> str:
        lines = [f"Device: {self.device}  |  Dtype: {self.dtype}  |  Compile: {self.cfg.compile}"]
        if self.device.type == "cuda":
            p = torch.cuda.get_device_properties(self.device)
            lines.append(f"GPU   : {p.name}  ({p.total_memory / 1024**3:.1f} GB)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DeviceManager({self.device}, {self.dtype})"
