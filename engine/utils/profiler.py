"""
engine/utils/profiler.py

Training profiler — MFU, throughput, and memory measurement.

Three things you need to quote metrics on a résumé or paper:

  1. MFU (Model FLOP Utilisation)
     Fraction of theoretical peak GPU FLOP/s actually used for useful work.
     Formula: achieved_FLOP/s / peak_FLOP/s
     Good values: 30-50% for a well-optimised single-GPU setup.
     World-class: ~57% (Chinchilla / PaLM 540B).

     We estimate achieved FLOP/s using the Karpathy / PaLM approximation:
       FLOPs per token ≈ 6 * N  (N = non-embedding parameters)
     This is accurate to within ~5% for standard transformers.

  2. Throughput (tokens/second)
     Tokens processed per wall-clock second during training.
     Multiply by batch_size × seq_len / step_time.

  3. Memory (GB)
     Peak CUDA memory allocated.  Useful for reporting "fits on T4 with
     gradient checkpointing" vs "requires A100".

Usage::

    from engine.utils import Profiler

    profiler = Profiler(model, seq_len=1024, device=dm.device)

    # Wrap the training loop
    for step, batch in enumerate(loader):
        with profiler.step():
            ...
        if step % 50 == 0:
            print(profiler.report())

    # One-shot benchmark (runs N forward+backward passes)
    results = profiler.benchmark(model, dm, sample_input, n_steps=20)
    print(results)
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

    @property
    def flops_per_token(self) -> float:
        """6N approximation (forward + backward ≈ 3× forward, forward ≈ 2N)."""
        return 6.0 * self.non_embed_params

    @property
    def achieved_flops_per_sec(self) -> float:
        return self.flops_per_token * self.tokens_per_sec


class Profiler:
    """
    Lightweight profiler that tracks step time, throughput, and memory.

    Args:
        model       : The model being trained (used to count parameters).
        seq_len     : Sequence length used in training.
        batch_size  : Batch size per step.
        device      : torch.device (used to determine if GPU stats are available).
        warmup_steps: Ignore the first N steps (JIT / cuDNN warmup).
    """

    # Peak theoretical FLOP/s for common GPUs (bfloat16 / float16 Tensor Core)
    _GPU_PEAK_TFLOPS = {
        "Tesla T4": 65.1,
        "Tesla V100": 112.0,
        "NVIDIA A100": 312.0,
        "NVIDIA H100": 989.0,
        "NVIDIA RTX 3090": 142.0,
        "NVIDIA RTX 4090": 330.0,
        "NVIDIA RTX 3080": 119.0,
        "NVIDIA A10G": 125.0,
    }

    def __init__(
        self,
        model: nn.Module,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        warmup_steps: int = 5,
    ) -> None:
        self.device = device
        self.warmup_steps = warmup_steps
        self._step_count = 0
        self._t_start = 0.0
        self.stats = ProfilerStats(tokens_per_step=batch_size * seq_len)
        self._count_params(model)
        self._gpu_name = self._detect_gpu()

    # ── Parameter counting ────────────────────────────────────────────────

    def _count_params(self, model: nn.Module) -> None:
        self.stats.model_params = sum(p.numel() for p in model.parameters())
        # Estimate non-embedding params (subtract embed_tokens if present)
        embed_params = 0
        for name, module in model.named_modules():
            if "embed_tokens" in name and isinstance(module, nn.Embedding):
                embed_params = module.weight.numel()
                break
        self.stats.non_embed_params = self.stats.model_params - embed_params

    def _detect_gpu(self) -> Optional[str]:
        if self.device.type != "cuda":
            return None
        return torch.cuda.get_device_properties(self.device).name

    def _peak_tflops(self) -> Optional[float]:
        if self._gpu_name is None:
            return None
        for name, tflops in self._GPU_PEAK_TFLOPS.items():
            if name in self._gpu_name:
                return tflops
        return None

    # ── Step context manager ─────────────────────────────────────────────

    @contextlib.contextmanager
    def step(self):
        """
        Context manager to time a single training step.

        Usage::

            with profiler.step():
                loss.backward()
                optimizer.step()
        """
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
                mem_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
                self.stats.peak_memory_gb = max(self.stats.peak_memory_gb, mem_gb)

    # ── Reporting ─────────────────────────────────────────────────────────

    def mfu(self) -> Optional[float]:
        """
        Model FLOP Utilisation as a fraction (0-1).

        Returns None if peak FLOP/s is unknown for this GPU.
        """
        peak = self._peak_tflops()
        if peak is None or self.stats.tokens_per_sec == 0:
            return None
        achieved_tflops = self.stats.achieved_flops_per_sec / 1e12
        return achieved_tflops / peak

    def report(self, step: Optional[int] = None) -> str:
        """Return a formatted one-line report string."""
        tps = self.stats.tokens_per_sec
        ms = self.stats.mean_step_ms
        mem = self.stats.peak_memory_gb
        mfu_v = self.mfu()

        parts = [f"step={step or self._step_count}"]
        parts.append(f"step_time={ms:.0f}ms")
        parts.append(f"tok/s={tps:,.0f}")
        if mfu_v is not None:
            parts.append(f"MFU={mfu_v * 100:.1f}%")
        if self.device.type == "cuda" and mem > 0:
            parts.append(f"mem={mem:.2f}GB")

        return "  ".join(parts)

    def summary(self) -> dict:
        """Return a dict of all metrics (for logging to W&B / JSON)."""
        d = {
            "tokens_per_sec": round(self.stats.tokens_per_sec),
            "mean_step_ms": round(self.stats.mean_step_ms, 1),
            "model_params": self.stats.model_params,
            "non_embed_params": self.stats.non_embed_params,
            "flops_per_token": round(self.stats.flops_per_token),
            "achieved_tflops": round(self.stats.achieved_flops_per_sec / 1e12, 3),
            "peak_memory_gb": round(self.stats.peak_memory_gb, 3),
        }
        mfu_v = self.mfu()
        if mfu_v is not None:
            d["mfu_percent"] = round(mfu_v * 100, 2)
            d["gpu"] = self._gpu_name
        return d

    # ── One-shot benchmark ────────────────────────────────────────────────

    @staticmethod
    def benchmark(
        model: nn.Module,
        dm,  # DeviceManager
        seq_len: int,
        batch_size: int,
        n_steps: int = 20,
        vocab_size: Optional[int] = None,
    ) -> dict:
        """
        Run a standalone benchmark: N forward+backward passes, return metrics.

        Args:
            model      : The model to benchmark.
            dm         : DeviceManager (for device + autocast).
            seq_len    : Sequence length.
            batch_size : Batch size.
            n_steps    : Number of steps to average over.
            vocab_size : Needed to generate synthetic input.
                         If None, reads from model.config.vocab_size.

        Returns dict with tok/s, MFU, memory.
        """
        model.train()
        _vocab = vocab_size or getattr(
            getattr(model, "config", None), "vocab_size", 1000
        )
        ids = torch.randint(0, _vocab, (batch_size, seq_len), device=dm.device)

        profiler = Profiler(
            model,
            seq_len=seq_len,
            batch_size=batch_size,
            device=dm.device,
            warmup_steps=3,
        )

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(n_steps):
            with profiler.step():
                with dm.autocast():
                    _, loss = model(ids, labels=ids)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

        return profiler.summary()
