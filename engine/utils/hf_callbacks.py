"""
engine/utils/hf_callbacks.py

HF Trainer Callbacks for lm_forge.
Includes ProfilingCallback for MFU, throughput, and memory tracking.
"""

from __future__ import annotations
import time
from typing import Optional

try:
    import torch
    from transformers import (
        TrainerCallback,
        TrainingArguments,
        TrainerState,
        TrainerControl,
    )
    from engine.utils.profiler import ProfilerStats, Profiler

    class ProfilingCallback(TrainerCallback):
        """
        HF Trainer Callback that tracks MFU, throughput, and memory.
        Results are logged to the Trainer's log_history and printed to console.

        Timing: measures wall-clock between consecutive on_step_end calls,
        which fires once per optimizer step (after all grad_accum micro-batches).
        """

        def __init__(self, warmup_steps: int = 5, seq_len: Optional[int] = None):
            self.warmup_steps = warmup_steps
            self.seq_len_override = seq_len
            self.profiler: Optional[Profiler] = None
            self._prev_step_end_time: Optional[float] = None

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model=None,
            **kwargs,
        ):
            if model is None:
                return

            # Initialize profiler using Trainer's config
            # We use effective_batch_size because on_step_end fires once per
            # optimizer step (after all grad_accum micro-batches).
            batch_size = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )

            # Try to find seq_len in override, model config, or fallback
            seq_len = self.seq_len_override if self.seq_len_override else getattr(model.config, "max_seq_len", 512)

            self.profiler = Profiler(
                model=model,
                seq_len=seq_len,
                batch_size=batch_size,
                device=args.device,
                warmup_steps=self.warmup_steps,
            )
            self._prev_step_end_time = None
            print(f"[Profiler] Initialized. GPU: {self.profiler._gpu_name or 'CPU'}")

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if self.profiler is None:
                return

            if args.device.type == "cuda":
                torch.cuda.synchronize()

            now = time.perf_counter()

            self.profiler._step_count += 1
            if self.profiler._step_count > self.profiler.warmup_steps:
                # Measure time between consecutive optimizer-step completions
                if self._prev_step_end_time is not None:
                    elapsed_ms = (now - self._prev_step_end_time) * 1000
                    self.profiler.stats.step_times_ms.append(elapsed_ms)

                if args.device.type == "cuda":
                    mem_gb = torch.cuda.max_memory_allocated(args.device) / 1024**3
                    self.profiler.stats.peak_memory_gb = max(
                        self.profiler.stats.peak_memory_gb, mem_gb
                    )

            self._prev_step_end_time = now

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if self.profiler and self.profiler._step_count > self.profiler.warmup_steps:
                report = self.profiler.report(step=state.global_step)
                print(f"  ↳ {report}")

                # Inject metrics into state for WandB/Tensorboard
                summary = self.profiler.summary()
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        state.log_history[-1][f"perf/{k}"] = v

except ImportError:

    class ProfilingCallback:
        def __init__(self, *args, **kwargs):
            pass
