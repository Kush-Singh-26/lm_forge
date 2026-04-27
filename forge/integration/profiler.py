"""
forge/integration/profiler.py

Lightweight profiler for tracking throughput, MFU, and memory.
"""

from __future__ import annotations
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
import torch
import torch.nn as nn

@dataclass
class ProfilerStats:
    step_times_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    tokens_per_step: int = 0
    peak_memory_gb: float = 0.0
    model_params: int = 0
    non_embed_params: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    seq_len: int = 0
    
    @property
    def mean_step_ms(self) -> float:
        if not self.step_times_ms:
            return 0.0
        return sum(self.step_times_ms) / len(self.step_times_ms)

    @property
    def tokens_per_sec(self) -> float:
        if self.mean_step_ms == 0:
            return 0.0
        return self.tokens_per_step / (self.mean_step_ms / 1000)

class Profiler:
    """
    Tracks training performance metrics.
    """
    
    _GPU_PEAK_TFLOPS = {
        "Tesla T4": 65.1,
        "Tesla V100": 112.0,
        "NVIDIA A100": 624.0,  # Standard reference for FP16/BF16 MFU
        "NVIDIA H100": 989.0,
        "NVIDIA L4": 242.0,
        "NVIDIA RTX 3090": 142.0,
        "NVIDIA RTX 4090": 330.0,
    }

    def __init__(
        self,
        model: nn.Module,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        warmup_steps: int = 5,
        world_size: int = 1,
    ) -> None:
        self.device = device
        self.warmup_steps = warmup_steps
        self.world_size = world_size
        self._step_count = 0
        self.stats = ProfilerStats(
            tokens_per_step=batch_size * seq_len,
            seq_len=seq_len,
        )
        self._count_params(model)
        self._gpu_name = self._detect_gpu()

    def _count_params(self, model: nn.Module) -> None:
        self.stats.model_params = sum(p.numel() for p in model.parameters())
        cfg = getattr(model, "config", None)
        if cfg:
            self.stats.num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 0))
            self.stats.hidden_size = getattr(cfg, "hidden_size", 0)
        
        embed_params = 0
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                embed_params = max(embed_params, module.weight.numel())
        self.stats.non_embed_params = self.stats.model_params - embed_params

    def _detect_gpu(self) -> Optional[str]:
        if self.device.type != "cuda":
            return None
        return torch.cuda.get_device_properties(self.device).name

    def mfu(self) -> Optional[float]:
        if self._gpu_name is None or self.stats.tokens_per_sec == 0:
            return None
        
        peak = None
        for name, tflops in self._GPU_PEAK_TFLOPS.items():
            if name in self._gpu_name:
                peak = tflops
                break
        
        if peak is None:
            return None
            
        # 6N approximation
        flops_per_token = 6.0 * self.stats.non_embed_params
        achieved_tflops = (flops_per_token * self.stats.tokens_per_sec) / 1e12
        return achieved_tflops / peak

    @contextlib.contextmanager
    def step(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._step_count += 1
        if self._step_count > self.warmup_steps:
            self.stats.step_times_ms.append(elapsed_ms)
            if self.device.type == "cuda":
                self.stats.peak_memory_gb = max(self.stats.peak_memory_gb, torch.cuda.max_memory_allocated(self.device) / 1024**3)

    def report(self, step: Optional[int] = None) -> str:
        ws = self.world_size
        tps = self.stats.tokens_per_sec * ws
        ms = self.stats.mean_step_ms
        mfu_v = self.mfu()
        
        parts = [f"step={step or self._step_count}", f"time={ms:.0f}ms", f"tok/s={tps:,.0f}"]
        if mfu_v is not None:
            parts.append(f"MFU={mfu_v * 100:.1f}%")
        return "  ".join(parts)
