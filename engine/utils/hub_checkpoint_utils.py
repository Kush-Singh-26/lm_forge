"""
engine/utils/hub_checkpoint_utils.py

HubCheckpointManager — manage checkpoints directly on Hugging Face Hub.
Uses HfApi to upload/delete folders to avoid Git history bloat while
providing remote storage for the "Colab Nomad" pattern.
Checkpoints are stored on a dedicated 'checkpoints' branch to keep 'main' clean.
"""

from __future__ import annotations
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFolder
from engine.config.schema import HubConfig, ExperimentConfig


class HubCheckpointManager:
    """
    Manages checkpoints on the Hugging Face Hub.
    Provides methods to upload, prune, and download checkpoints.
    Uses a dedicated branch to prevent Git history bloat on 'main'.
    """

    def __init__(self, hub_cfg: HubConfig, exp_cfg: ExperimentConfig) -> None:
        self.cfg = hub_cfg
        self.exp_cfg = exp_cfg
        self._enabled = False
        self.branch = "checkpoints"

        if not hub_cfg.repo_id:
            return

        self.token = os.environ.get(hub_cfg.token_env, "")
        if not self.token:
            print(
                f"[HubCheckpoint] WARNING: Env var '{hub_cfg.token_env}' not set. "
                "Remote operations will likely fail for private repos."
            )

        self.api = HfApi(token=self.token)
        self._enabled = True
        self.remote_prefix = f"checkpoints/{self.exp_cfg.name}"

        # 1. Verify permissions and repo access
        try:
            user_info = self.api.whoami()
            print(
                f"[HubCheckpoint] Authenticated as: {user_info.get('name', 'unknown')}"
            )

            # Ensure the checkpoints branch exists
            self.api.create_branch(
                repo_id=self.cfg.repo_id, branch=self.branch, exist_ok=True
            )
        except Exception as e:
            print(
                f"[HubCheckpoint] WARNING: Permission check or branch creation failed: {e}"
            )
            pass

        print(
            f"[HubCheckpoint] Initialized for repo: {hub_cfg.repo_id} (branch: {self.branch})"
        )

    def upload_checkpoint(
        self, local_dir: str | Path, step: int, retries: int = 3
    ) -> None:
        """Upload a local checkpoint folder to the Hub with retries."""
        if not self._enabled:
            return

        local_dir = Path(local_dir)
        path_in_repo = f"{self.remote_prefix}/checkpoint-{step}"

        for attempt in range(retries):
            try:
                print(
                    f"[HubCheckpoint] Uploading checkpoint-{step} to branch '{self.branch}' "
                    f"(Attempt {attempt + 1}/{retries})..."
                )
                self.api.upload_folder(
                    repo_id=self.cfg.repo_id,
                    folder_path=str(local_dir),
                    path_in_repo=path_in_repo,
                    commit_message=f"Upload checkpoint-{step}",
                    allow_patterns=["*"],
                    revision=self.branch,
                )
                print(f"[HubCheckpoint] Upload complete: {path_in_repo}")
                return
            except Exception as e:
                print(
                    f"[HubCheckpoint] ERROR: Upload attempt {attempt + 1} failed: {e}"
                )
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    print("[HubCheckpoint] Maximum retries reached. Upload failed.")

    def prune_checkpoints(self, keep: int = 2) -> None:
        """Keep only the N most recent checkpoints on the remote branch."""
        if not self._enabled:
            return

        steps = self.list_remote_checkpoints()
        if len(steps) <= keep:
            return

        to_delete = steps[:-keep]
        print(
            f"[HubCheckpoint] Pruning {len(to_delete)} old checkpoints from '{self.branch}': {to_delete}"
        )

        for step in to_delete:
            path_in_repo = f"{self.remote_prefix}/checkpoint-{step}"
            try:
                self.api.delete_folder(
                    repo_id=self.cfg.repo_id,
                    path_in_repo=path_in_repo,
                    commit_message=f"Prune old checkpoint-{step}",
                    revision=self.branch,
                )
                print(f"[HubCheckpoint] Deleted from Hub: {path_in_repo}")
            except Exception as e:
                print(f"[HubCheckpoint] ERROR: Deletion failed for {path_in_repo}: {e}")

    def list_remote_checkpoints(self) -> List[int]:
        """List checkpoint steps currently on the Hub branch for this experiment."""
        if not self._enabled:
            return []

        try:
            repo_tree = self.api.list_repo_tree(
                repo_id=self.cfg.repo_id,
                path_in_repo=self.remote_prefix,
                recursive=False,
                revision=self.branch,
            )

            steps = []
            for item in repo_tree:
                if isinstance(item, RepoFolder) and "checkpoint-" in item.path:
                    folder_name = Path(item.path).name
                    step_str = folder_name.split("checkpoint-")[-1]
                    if step_str.isdigit():
                        steps.append(int(step_str))

            return sorted(steps)
        except Exception:
            return []

    def push_final(self, local_dir: str | Path, retries: int = 3) -> None:
        """Upload the final model to the 'final/' folder in the Hub repo (main branch)."""
        if not self._enabled:
            return

        local_dir = Path(local_dir)
        path_in_repo = f"final/{self.exp_cfg.name}"

        for attempt in range(retries):
            try:
                print(
                    f"[HubCheckpoint] Uploading final model to Hub (main) at {path_in_repo} "
                    f"(Attempt {attempt + 1}/{retries})..."
                )
                self.api.upload_folder(
                    repo_id=self.cfg.repo_id,
                    folder_path=str(local_dir),
                    path_in_repo=path_in_repo,
                    commit_message=f"Upload final model - {self.exp_cfg.name}",
                    allow_patterns=["*"],
                )
                print(f"[HubCheckpoint] Final model upload complete.")
                return
            except Exception as e:
                print(
                    f"[HubCheckpoint] ERROR: Final upload attempt {attempt + 1} failed: {e}"
                )
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))

    def pull_latest(self, local_root: str | Path) -> Optional[Path]:
        """
        Download the latest checkpoint from the Hub branch.
        Uses shadow rotation for atomic-like local updates.
        """
        if not self._enabled:
            return None

        steps = self.list_remote_checkpoints()
        if not steps:
            print(
                f"[HubCheckpoint] No remote checkpoints found on branch '{self.branch}'."
            )
            return None

        latest_step = steps[-1]
        path_in_repo = f"{self.remote_prefix}/checkpoint-{latest_step}"
        target_path = Path(local_root).resolve() / f"checkpoint-{latest_step}"
        backup_path = Path(str(target_path) + ".old")

        print(
            f"[HubCheckpoint] Pulling latest checkpoint-{latest_step} from branch '{self.branch}'..."
        )
        try:
            # 1. Download to project root (mirrors repo structure)
            snapshot_download(
                repo_id=self.cfg.repo_id,
                allow_patterns=[f"{path_in_repo}/*"],
                local_dir=".",
                token=self.token,
                revision=self.branch,
            )

            downloaded_path = Path(".").resolve() / path_in_repo
            if not downloaded_path.exists():
                print(
                    f"[HubCheckpoint] ERROR: Downloaded path not found: {downloaded_path}"
                )
                return None

            # 2. Check if we need to move it to a specific local_root
            if downloaded_path.resolve() != target_path.resolve():
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Shadow rotation: preserve old if exists
                if target_path.exists():
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    target_path.rename(backup_path)

                try:
                    shutil.move(str(downloaded_path), str(target_path))
                    # Success: clean up backup
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                except Exception as e:
                    # Failure: restore backup
                    print(f"[HubCheckpoint] ERROR during move: {e}. Restoring backup.")
                    if backup_path.exists():
                        if target_path.exists():
                            shutil.rmtree(target_path)
                        backup_path.rename(target_path)
                    raise e

            print(f"[HubCheckpoint] Pull complete: {target_path}")
            return target_path

        except Exception as e:
            print(f"[HubCheckpoint] ERROR: Pull failed: {e}")

        return None
