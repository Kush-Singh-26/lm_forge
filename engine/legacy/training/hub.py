"""
engine/training/hub.py

HubSync — bidirectional checkpoint sync with HuggingFace Hub.

Implements the "Colab Nomad" pattern:
  1. pull() at session start — resume from the last pushed checkpoint
  2. push() periodically during training — so if Colab dies, no work is lost
  3. The Hub repo is the single source of truth for all training state

The Hub repo layout::

    {repo_id}/
    ├── checkpoints/
    │   ├── step_000500/
    │   │   ├── model_config.json
    │   │   ├── model.safetensors    (or pytorch_model.bin)
    │   │   └── trainer_state.pt
    │   └── step_001000/
    │       └── ...
    ├── final/
    │   ├── model_config.json
    │   └── model.safetensors
    └── README.md                   ← auto-generated model card

Authentication:
  Provide a HF write token in the environment variable named by
  HubConfig.token_env (default "HF_TOKEN").  In Colab, add it to
  Secrets or set it manually:

      import os; os.environ["HF_TOKEN"] = "hf_..."

  If the token is missing, all Hub operations are silently skipped
  so you can develop locally without a token.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from engine.config.schema import HubConfig, ExperimentConfig


class HubSync:
    """
    Wraps HuggingFace huggingface_hub to push/pull checkpoints.

    All methods are safe to call even if huggingface_hub is not installed
    or the token is missing — they simply log a warning and return None.
    """

    def __init__(self, hub_cfg: HubConfig, exp_cfg: ExperimentConfig) -> None:
        self.cfg      = hub_cfg
        self.exp_cfg  = exp_cfg
        self._api     = None
        self._enabled = False

        if not hub_cfg.repo_id:
            print("[HubSync] No repo_id set — Hub sync disabled.")
            return

        token = os.environ.get(hub_cfg.token_env, "")
        if not token:
            print(f"[HubSync] Env var '{hub_cfg.token_env}' not set — Hub sync disabled.")
            return

        try:
            from huggingface_hub import HfApi
            self._api = HfApi(token=token)
            self._ensure_repo_exists()
            self._enabled = True
            print(f"[HubSync] Enabled → {hub_cfg.repo_id}")
        except ImportError:
            print("[HubSync] huggingface_hub not installed — pip install huggingface_hub")

    # ── internal ─────────────────────────────────────────────────────────

    def _ensure_repo_exists(self) -> None:
        """Create the Hub repo if it does not already exist."""
        from huggingface_hub import create_repo
        try:
            create_repo(
                self.cfg.repo_id,
                private=self.cfg.private,
                exist_ok=True,
                repo_type="model",
            )
        except Exception as e:
            print(f"[HubSync] Could not ensure repo exists: {e}")

    def _ckpt_folder(self, step: int) -> str:
        """Hub path prefix for a checkpoint at a given step."""
        return f"checkpoints/step_{step:07d}"

    # ── push ─────────────────────────────────────────────────────────────

    def push_checkpoint(self, local_dir: str | Path, step: int) -> None:
        """
        Upload all files in local_dir to checkpoints/step_{step:07d}/ on Hub.

        Called by the Trainer every save_every steps.
        """
        if not self._enabled:
            return
        local_dir = Path(local_dir)
        hub_folder = self._ckpt_folder(step)
        print(f"[HubSync] Pushing step {step} → {self.cfg.repo_id}/{hub_folder} ...")
        try:
            self._api.upload_folder(
                folder_path=str(local_dir),
                repo_id=self.cfg.repo_id,
                path_in_repo=hub_folder,
                repo_type="model",
                commit_message=f"Checkpoint step {step} — {self.exp_cfg.name}",
            )
            # Update a pointer file so pull_latest() knows where to resume
            pointer = f"checkpoints/latest_step.txt"
            self._api.upload_file(
                path_or_fileobj=str(step).encode(),
                path_in_repo=pointer,
                repo_id=self.cfg.repo_id,
                repo_type="model",
                commit_message=f"Update latest pointer → {step}",
            )
            print(f"[HubSync] Push complete.")
        except Exception as e:
            print(f"[HubSync] Push failed: {e}")

    def push_final(self, local_dir: str | Path) -> None:
        """Upload the final model to the 'final/' folder."""
        if not self._enabled:
            return
        try:
            self._api.upload_folder(
                folder_path=str(local_dir),
                repo_id=self.cfg.repo_id,
                path_in_repo="final",
                repo_type="model",
                commit_message=f"Final model — {self.exp_cfg.name}",
            )
            print(f"[HubSync] Final model pushed → {self.cfg.repo_id}/final")
        except Exception as e:
            print(f"[HubSync] Final push failed: {e}")

    # ── pull ─────────────────────────────────────────────────────────────

    def pull_latest(self, local_dir: str | Path) -> Optional[Path]:
        """
        Download the latest checkpoint from Hub into local_dir.

        Returns the local path to the checkpoint directory, or None if
        no checkpoint exists yet (fresh training run).

        Call this at the very start of a Colab session before constructing
        the Trainer — the Trainer's resume_from should point to the result.
        """
        if not self._enabled:
            return None

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            # Check for the latest pointer
            all_files = list(list_repo_files(self.cfg.repo_id, repo_type="model"))
            if "checkpoints/latest_step.txt" not in all_files:
                print("[HubSync] No checkpoint found on Hub — starting fresh.")
                return None

            # Read the step number
            pointer_path = hf_hub_download(
                self.cfg.repo_id,
                filename="checkpoints/latest_step.txt",
                repo_type="model",
            )
            step = int(Path(pointer_path).read_text().strip())
            hub_folder = self._ckpt_folder(step)
            print(f"[HubSync] Pulling checkpoint step {step} from Hub ...")

            ckpt_files = [f for f in all_files if f.startswith(hub_folder + "/")]
            if not ckpt_files:
                print(f"[HubSync] No files found at {hub_folder}")
                return None

            ckpt_local = local_dir / f"step_{step:07d}"
            ckpt_local.mkdir(parents=True, exist_ok=True)

            for hub_path in ckpt_files:
                filename = hub_path[len(hub_folder) + 1:]   # strip prefix
                local_file = hf_hub_download(
                    self.cfg.repo_id,
                    filename=hub_path,
                    repo_type="model",
                    local_dir=str(ckpt_local),
                )
                # hf_hub_download may nest; normalise to flat structure
                downloaded = Path(local_file)
                target = ckpt_local / filename
                if downloaded != target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(downloaded, target)

            print(f"[HubSync] Checkpoint ready at {ckpt_local}")
            return ckpt_local

        except Exception as e:
            print(f"[HubSync] Pull failed: {e}")
            return None
