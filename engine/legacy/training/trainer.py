"""
engine/training/trainer.py

Trainer — orchestrates the training loop for the "Colab Nomad" pattern.

[NOTE] This trainer is legacy. Native HF Trainer (transformers.Trainer)
       is now the recommended way to train lm_forge models.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from engine.config.schema import ExperimentConfig
from engine.legacy.training.device import DeviceManager
from engine.legacy.training.hub import HubSync


class Trainer:
    """
    Training loop for lm_forge models.

    Args:
        model       : nn.Module that returns (logits, loss) from forward().
        optimizer   : Any torch Optimizer.
        dm          : DeviceManager — owns device/dtype/autocast/scaler.
        exp_cfg     : Full ExperimentConfig (model + training + hub).
        hub         : HubSync instance (pass HubSync with empty repo_id for no-op).
        scheduler   : Optional LR scheduler.  Step is called every opt step.
        resume_from : Path to a checkpoint directory to resume from.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dm: DeviceManager,
        exp_cfg: ExperimentConfig,
        hub: Optional[HubSync] = None,
        scheduler: Optional[LRScheduler] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dm = dm
        self.cfg = exp_cfg.training
        self.exp_cfg = exp_cfg
        self.hub = hub
        self.scheduler = scheduler

        self._scaler = dm.grad_scaler()
        self.global_step: int = 0
        self.epoch: int = 0

        self._output_root = Path("checkpoints") / exp_cfg.name

        if resume_from:
            self._load_checkpoint(resume_from)

    # ── training loop ─────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ) -> None:
        cfg = self.cfg
        print(f"\n{'━' * 60}")
        print(f"  lm_forge Trainer  |  {self.exp_cfg.name}")
        print(f"  {self.dm.summary()}")
        print(
            f"  Steps: {cfg.max_steps}  |  Batch: {cfg.batch_size}×{cfg.grad_accum}"
            f"  (effective {cfg.effective_batch_size})"
        )
        print(f"{'━' * 60}\n")

        max_steps = cfg.max_steps
        self.model.train()
        running_loss = 0.0
        t0 = time.perf_counter()

        resume_micro_count = getattr(self, "_micro_count", 0)
        if resume_micro_count > 0:
            print(f"[Trainer] Resuming... skipping {resume_micro_count} micro-batches.")

        total_micro_batches_seen = 0
        while self.global_step < max_steps:
            self.epoch += 1
            for batch in train_loader:
                total_micro_batches_seen += 1
                if total_micro_batches_seen <= resume_micro_count:
                    continue

                if self.global_step >= max_steps:
                    break

                batch = self.dm.to_device(batch)

                # ── forward ──────────────────────────────────────────────
                with self.dm.autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch.get("labels", batch["input_ids"]),
                        attention_mask=batch.get("attention_mask"),
                    )
                    # Support both (logits, loss) and (logits, loss, presents)
                    loss = outputs[1]
                    loss = loss / cfg.grad_accum

                # ── backward ─────────────────────────────────────────────
                if self._scaler:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item()

                # ── optimiser step ────────────────────────────────────────
                # We always step at the end of every micro-batch and track
                # accumulated steps with a simple modulo so the loop never
                # needs an internal counter.
                # Use batch_idx trick: increment before checking modulo.
                # Instead track via global_step parity — accumulate until
                # we've done grad_accum backward passes.
                self._micro_count = getattr(self, "_micro_count", 0) + 1
                if self._micro_count % cfg.grad_accum == 0:
                    self._opt_step()
                    self.global_step += 1

                    # ── log ──────────────────────────────────────────────
                    if self.global_step % cfg.log_every == 0:
                        elapsed = time.perf_counter() - t0
                        lr = self.optimizer.param_groups[0]["lr"]
                        print(
                            f"step {self.global_step:>7} | "
                            f"loss {running_loss:.4f} | "
                            f"lr {lr:.2e} | "
                            f"{elapsed:.1f}s"
                        )
                        running_loss = 0.0
                        t0 = time.perf_counter()

                    # ── eval ─────────────────────────────────────────────
                    if (
                        cfg.eval_every > 0
                        and self.global_step % cfg.eval_every == 0
                        and eval_loader is not None
                    ):
                        eval_loss = self.evaluate(eval_loader)
                        print(f"  ↳ eval loss: {eval_loss:.4f}")
                        self.model.train()

                    # ── save + push ───────────────────────────────────────
                    if cfg.save_every > 0 and self.global_step % cfg.save_every == 0:
                        ckpt_path = self._save_checkpoint()
                        if self.hub:
                            self.hub.push_checkpoint(ckpt_path, self.global_step)

                    if self.global_step >= max_steps:
                        break

        # ── final save + push ─────────────────────────────────────────────
        print("\nTraining complete.")
        final_path = self._output_root / "final"
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(final_path)
        else:
            final_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), final_path / "pytorch_model.bin")
        if self.hub:
            self.hub.push_final(final_path)
        print(f"Final model at: {final_path}")

    # ── eval ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in eval_loader:
            batch = self.dm.to_device(batch)
            with self.dm.autocast():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch.get("labels", batch["input_ids"]),
                    attention_mask=batch.get("attention_mask"),
                )
                loss = outputs[1]
            total += loss.item()
            n += 1
        return total / max(n, 1)

    # ── helpers ───────────────────────────────────────────────────────────

    def _opt_step(self) -> None:
        """Unscale, clip, step, zero — one optimiser update."""
        if self._scaler:
            self._scaler.unscale_(self.optimizer)
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )
        if self._scaler:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _save_checkpoint(self) -> Path:
        ckpt = self._output_root / f"step_{self.global_step:07d}"
        ckpt.mkdir(parents=True, exist_ok=True)

        # Model weights + config
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(ckpt)
        else:
            self.exp_cfg.model.save(ckpt)
            torch.save(self.model.state_dict(), ckpt / "pytorch_model.bin")

        # Training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "micro_count": getattr(self, "_micro_count", 0),
            "optimizer": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state().numpy(),
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        if self._scaler:
            state["scaler"] = self._scaler.state_dict()
        torch.save(state, ckpt / "trainer_state.pt")
        print(f"  ↳ checkpoint saved → {ckpt}")
        return ckpt

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        state_path = path / "trainer_state.pt"
        if not state_path.exists():
            print(f"[Trainer] No trainer_state.pt in {path} — starting fresh.")
            return
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.global_step = state["global_step"]
        self.epoch = state.get("epoch", 0)
        self._micro_count = state.get("micro_count", 0)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self._scaler and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])
        # Restore RNG state for exact resume of DataLoader shuffling
        if "rng_state" in state:
            torch.set_rng_state(torch.from_numpy(state["rng_state"]))
        print(f"[Trainer] Resumed from step {self.global_step}.")
