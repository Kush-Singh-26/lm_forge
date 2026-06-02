"""
forge/integration/hf.py

Hugging Face Trainer Integration for Forge.
Provides ForgeCallback for automated state management and Hub syncing.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Optional

from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from forge.state.hub_manager import HubManager
from forge.config import ForgeConfig, ProfileConfig
from forge.integration.profiler import Profiler
from forge.data.streaming import ForgeDataHelper


class ForgeTrainer(Trainer):
    """
    Automated Trainer for Forge.
    - Automatically injects ForgeCallback.
    - Applies profile-based hyperparameter overrides from forge.yaml.
    - Handles stateful resumption for streaming datasets.
    """
    def __init__(self, *args, config_path: str | Path = "forge.yaml", **kwargs):
        self.config_path = Path(config_path)
        self.forge_cfg: Optional[ForgeConfig] = None
        self.active_profile: Optional[ProfileConfig] = None
        self.hub_manager: Optional[HubManager] = None  # Used by train() for Hub pull
        
        if self.config_path.exists():
            self.forge_cfg = ForgeConfig.load(self.config_path)
            env = self.forge_cfg.detect_env()
            self.active_profile = self.forge_cfg.get_active_profile()

            # Create HubManager on ForgeTrainer so train() can pull checkpoints from Hub
            try:
                self.hub_manager = HubManager(self.forge_cfg.state, self.forge_cfg.name)
            except Exception as e:
                print(f"[Forge] WARNING: HubManager init failed: {e}")
            
            # Enforcement for ephemeral environments
            if env in ["colab", "kaggle"] and not os.environ.get("HF_TOKEN"):
                print(f"[Forge] WARNING: Running in ephemeral {env} without HF_TOKEN. State will be LOST on restart!")

            # Apply Profile Overrides to TrainingArguments
            _FORGE_ONLY_FIELDS = {"probe_memory", "data_cache"}
            if "args" in kwargs and isinstance(kwargs["args"], TrainingArguments):
                if self.active_profile:
                    print(f"[Forge] Applying profile overrides for environment: {env}")
                    for key, value in self.active_profile.model_dump(exclude_none=True).items():
                        if key in _FORGE_ONLY_FIELDS:
                            continue  # Skip internal Forge fields
                        if hasattr(kwargs["args"], key):
                            setattr(kwargs["args"], key, value)
            
            # Inject ForgeCallback
            callback = ForgeCallback(config_path=self.config_path)
            if "callbacks" not in kwargs or kwargs["callbacks"] is None:
                kwargs["callbacks"] = [callback]
            else:
                kwargs["callbacks"].append(callback)

        # Handle Streaming Dataset Resumption
        if "train_dataset" in kwargs and self.forge_cfg:
            import torch.utils.data
            if isinstance(kwargs["train_dataset"], torch.utils.data.IterableDataset):
                output_dir = kwargs["args"].output_dir if "args" in kwargs else "./outputs"
                stats = ForgeDataHelper.get_resume_stats(output_dir)
                skip = stats.get("samples_seen", 0)
                if skip > 0:
                    print(f"[Forge] Resuming: skipping {skip:,} already-seen samples...")
                    # Note: SequencePacker wraps an HF IterableDataset which supports .skip()
                    inner = getattr(kwargs["train_dataset"], "dataset", None)
                    if inner is not None and hasattr(inner, "skip"):
                        kwargs["train_dataset"].dataset = inner.skip(skip)
                    else:
                        print("[Forge] WARNING: Could not fast-forward dataset — dataset type does not support .skip()")

        # TRIGGER DATA CACHE if requested — must happen BEFORE super().__init__() so
        # HF_HOME is redirected before the Trainer initializes any dataset caches.
        if self.active_profile and self.active_profile.data_cache:
            ForgeDataHelper.setup_fast_cache()

        super().__init__(*args, **kwargs)

        # TRIGGER PROBER if requested
        if self.active_profile and self.active_profile.probe_memory:
            from forge.integration.prober import ForgeMemoryProber
            target_batch = (
                self.args.per_device_train_batch_size * 
                self.args.gradient_accumulation_steps * 
                self.args.world_size
            )
            new_per_device, new_accum = ForgeMemoryProber.probe_batch_size(
                model=self.model,
                train_dataset=self.train_dataset,
                data_collator=self.data_collator,
                target_total_batch=target_batch,
                fp16=self.args.fp16,
                bf16=self.args.bf16,
            )
            self.args.per_device_train_batch_size = new_per_device
            self.args.gradient_accumulation_steps = new_accum


    def train(self, *args, resume_from_checkpoint: Optional[str | bool] = None, **kwargs):
        """
        Overridden train to automatically detect and resume from the latest checkpoint.

        Resume priority:
          1. Explicit checkpoint path passed by caller
          2. Local checkpoint in output_dir (works on Modal where disk persists across calls)
          3. Hub checkpoint (critical for Kaggle: local disk is empty on every session start)
        """
        if resume_from_checkpoint is None:
            output_dir = Path(self.args.output_dir)

            # --- Priority 2: local checkpoint ---
            if output_dir.exists():
                checkpoints = []
                for item in output_dir.iterdir():
                    if item.is_dir() and "checkpoint-" in item.name:
                        step_str = item.name.split("checkpoint-")[-1]
                        if step_str.isdigit():
                            checkpoints.append((int(step_str), item))

                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, key=lambda x: x[0])[-1][1]
                    print(f"[Forge] Auto-detect: Resuming from local {latest_checkpoint.name}")
                    resume_from_checkpoint = str(latest_checkpoint)

            # --- Priority 3: Hub checkpoint (Kaggle / fresh machine) ---
            if resume_from_checkpoint is None and self.hub_manager is not None:
                print("[Forge] No local checkpoint found. Checking Hub for latest checkpoint...")
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    pulled = self.hub_manager.pull_latest(output_dir)
                    if pulled is not None:
                        print(f"[Forge] Pulled checkpoint from Hub: {pulled.name}")
                        resume_from_checkpoint = str(pulled)
                    else:
                        print("[Forge] No Hub checkpoint found. Starting fresh training.")
                except Exception as e:
                    print(f"[Forge] WARNING: Hub pull failed ({e}). Starting fresh.")

        return super().train(*args, resume_from_checkpoint=resume_from_checkpoint, **kwargs)


class ForgeCallback(TrainerCallback):
    """
    HF Trainer Callback for Nomad Training.
    - Saves data_state.json for stateful resumption.
    - Syncs checkpoints to the HF Hub.
    - Profiles training performance.
    """

    def __init__(self, config_path: str | Path = "forge.yaml"):
        self.config_path = Path(config_path)
        self.forge_cfg: Optional[ForgeConfig] = None
        self.hub_manager: Optional[HubManager] = None
        self.profiler: Optional[Profiler] = None
        self.samples_seen = 0
        
        if self.config_path.exists():
            try:
                self.forge_cfg = ForgeConfig.load(self.config_path)
                self.hub_manager = HubManager(self.forge_cfg.state, self.forge_cfg.name)
            except Exception as e:
                print(f"[Forge] WARNING: Failed to load config from {config_path}: {e}")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        # Initialize samples_seen from previous state if resuming
        stats = ForgeDataHelper.get_resume_stats(args.output_dir)
        self.samples_seen = stats.get("samples_seen", 0)

        if model is not None:
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            # seq_len detection
            seq_len = getattr(model.config, "max_position_embeddings", 512)
            if hasattr(model.config, "max_seq_len"):
                seq_len = model.config.max_seq_len
            
            self.profiler = Profiler(
                model=model,
                seq_len=seq_len,
                batch_size=batch_size,
                device=args.device,
                world_size=args.world_size
            )
            if args.process_index == 0:
                print(f"[Forge] Profiler initialized (Rank {args.process_index})")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Increment samples_seen based on actual current batch size
        self.samples_seen += (
            args.per_device_train_batch_size * 
            args.gradient_accumulation_steps * 
            args.world_size
        )
        if self.profiler:
            # Note: We can't use the 'with' context here easily in a callback
            # so we just increment step count manually if needed, or handle it via time
            pass 

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.process_index == 0:
            # Update live status for 'forge dash'
            status_file = Path(args.output_dir) / "forge_status.json"
            status_data = {
                "step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": state.epoch,
                "loss": state.log_history[-1].get("loss") if state.log_history else None,
                "learning_rate": state.log_history[-1].get("learning_rate") if state.log_history else None,
                "timestamp": time.time(),
                "throughput": state.log_history[-1].get("train_samples_per_second") if state.log_history else None,
                "mfu": self.profiler.mfu() if self.profiler else None,
                "samples_seen": self.samples_seen,
            }
            try:
                # Ensure directory exists
                status_file.parent.mkdir(parents=True, exist_ok=True)
                with open(status_file, "w") as f:
                    json.dump(status_data, f, indent=2)
            except Exception as e:
                print(f"[Forge] WARNING: Failed to update forge_status.json: {e}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Saves data state and syncs to Hub on Trainer save.
        """
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_path.exists():
            return

        # 1. Save Data State (Using cumulative counter)
        data_state = {
            "samples_seen": self.samples_seen,
            "global_step": state.global_step,
            "world_size": args.world_size,
        }

        with open(checkpoint_path / "data_state.json", "w") as f:
            json.dump(data_state, f, indent=2)

        # 1.5 Save Metadata for 'forge ship'
        metadata = {
            "project_name": self.forge_cfg.name if self.forge_cfg else "unknown",
            "global_step": state.global_step,
            "total_samples": self.samples_seen,
            "loss": state.log_history[-1].get("loss") if state.log_history else None,
            "learning_rate": state.log_history[-1].get("learning_rate") if state.log_history else None,
            "epoch": state.epoch,
            "timestamp": time.time(),
            "hardware": self.forge_cfg.detect_env() if self.forge_cfg else "unknown"
        }
        with open(checkpoint_path / "forge_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if args.process_index == 0:
            print(f"[Forge] Saved progress: {self.samples_seen:,} samples seen.")

            # 2. Sync to Hub
            if self.hub_manager:
                print(f"[Forge] Syncing checkpoint-{state.global_step} to Hub...")
                try:
                    self.hub_manager.upload_checkpoint(checkpoint_path, state.global_step)
                    self.hub_manager.prune_checkpoints()
                except Exception as e:
                    print(f"[Forge] ERROR: Hub sync failed: {e}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.hub_manager and args.process_index == 0 and args.should_save:
            final_path = Path(args.output_dir) / "final"
            if final_path.exists():
                print("[Forge] Training complete. Final model sync not yet implemented as auto-push.")
                # We could add push_final here if desired
