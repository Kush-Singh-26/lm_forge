"""
engine/utils/hf_callbacks.py

HF Trainer Callbacks for lm_forge.
Includes ProfilingCallback for MFU, throughput, and memory tracking.
"""

from __future__ import annotations
import time
from typing import Optional

try:
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
        """

        def __init__(self, warmup_steps: int = 5):
            self.warmup_steps = warmup_steps
            self.profiler: Optional[Profiler] = None
            self._step_start_time = 0.0

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
            # We use effective_batch_size because HFTrainer handles grad_accum
            batch_size = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )

            # Try to find seq_len in model config or args
            seq_len = getattr(model.config, "max_seq_len", 512)

            self.profiler = Profiler(
                model=model,
                seq_len=seq_len,
                batch_size=batch_size,
                device=args.device,
                warmup_steps=self.warmup_steps,
            )
            print(f"[Profiler] Initialized. GPU: {self.profiler._gpu_name or 'CPU'}")

        def on_step_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if args.device.type == "cuda":
                torch.cuda.synchronize()
            self._step_start_time = time.perf_counter()

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

            elapsed_ms = (time.perf_counter() - self._step_start_time) * 1000

            self.profiler._step_count += 1
            if self.profiler._step_count > self.profiler.warmup_steps:
                self.profiler.stats.step_times_ms.append(elapsed_ms)

                if args.device.type == "cuda":
                    import torch

                    mem_gb = torch.cuda.max_memory_allocated(args.device) / 1024**3
                    self.profiler.stats.peak_memory_gb = max(
                        self.profiler.stats.peak_memory_gb, mem_gb
                    )

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
