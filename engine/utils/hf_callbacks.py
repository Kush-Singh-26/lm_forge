"""
engine/utils/hf_callbacks.py

HF Trainer Callbacks for lm_forge.
Includes ProfilingCallback and HubCheckpointCallback.
"""

from __future__ import annotations
import time
from pathlib import Path
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
    from engine.utils.hub_checkpoint_utils import HubCheckpointManager
    from engine.config.schema import ExperimentConfig

    class ProfilingCallback(TrainerCallback):
        """
        HF Trainer Callback that tracks MFU, throughput, and memory.
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

            batch_size = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )

            seq_len = (
                self.seq_len_override
                if self.seq_len_override
                else getattr(model.config, "max_seq_len", 512)
            )

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

                summary = self.profiler.summary()
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        state.log_history[-1][f"perf/{k}"] = v

    class HubCheckpointCallback(TrainerCallback):
        """
        HF Trainer Callback that syncs checkpoints to the Hugging Face Hub as folders.
        Keeps only the most recent N checkpoints on the Hub.
        """

        def __init__(self, exp_cfg: ExperimentConfig):
            self.exp_cfg = exp_cfg
            self.manager = HubCheckpointManager(exp_cfg.hub, exp_cfg)

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """Called whenever the Trainer saves a checkpoint locally."""
            if not self.manager._enabled or not self.exp_cfg.hub.use_hub_checkpoints:
                return

            checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint_path.exists():
                print(
                    f"[HubCheckpointCallback] Step {state.global_step}: Uploading FULL STATE to Hub..."
                )
                try:
                    self.manager.upload_checkpoint(checkpoint_path, state.global_step)
                    # Prune remote checkpoints (keep N)
                    self.manager.prune_checkpoints(
                        keep=self.exp_cfg.hub.checkpoint_limit
                    )
                except Exception as e:
                    print(f"[HubCheckpointCallback] ERROR: Hub sync failed: {e}")

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if not self.manager._enabled or not self.exp_cfg.hub.use_hub_checkpoints:
                return

            final_path = Path(args.output_dir) / "final"
            if final_path.exists():
                print(f"[HubCheckpointCallback] Syncing final model to Hub...")
                try:
                    self.manager.push_final(final_path)
                except Exception as e:
                    print(f"[HubCheckpointCallback] ERROR: Final sync failed: {e}")

except ImportError:

    class ProfilingCallback:
        def __init__(self, *args, **kwargs):
            pass

    class HubCheckpointCallback:
        def __init__(self, *args, **kwargs):
            pass
