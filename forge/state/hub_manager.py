"""
forge/state/hub_manager.py

HubManager — manages checkpoints directly on Hugging Face Hub.
Provides remote storage for the "Nomad Training" pattern.
Checkpoints are stored on a dedicated branch to keep 'main' clean.
"""

from __future__ import annotations
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List
import json
import tempfile

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFolder
from forge.config import StateConfig

class HubManager:
    """
    Manages checkpoints on the Hugging Face Hub.
    Provides methods to upload, prune, and download checkpoints.
    """

    def __init__(self, cfg: StateConfig, project_name: str) -> None:
        self.cfg = cfg
        self.project_name = project_name
        self.branch = cfg.branch
        self.repo_id = cfg.repo_id

        self.token = os.environ.get(cfg.token_env, "")
        if not self.token:
            print(
                f"[Forge.Hub] WARNING: Env var '{cfg.token_env}' not set. "
                "Remote operations will likely fail for private repos."
            )

        self.api = HfApi(token=self.token)
        self.remote_prefix = f"checkpoints/{self.project_name}"

        # Ensure the branch exists
        try:
            self.api.create_branch(
                repo_id=self.repo_id, branch=self.branch, exist_ok=True
            )
        except Exception as e:
            print(f"[Forge.Hub] WARNING: Branch creation failed: {e}")

    def upload_checkpoint(
        self, local_dir: str | Path, step: int, retries: int = 3
    ) -> None:
        """Upload a local checkpoint folder to the Hub with retries."""
        local_dir = Path(local_dir)
        path_in_repo = f"{self.remote_prefix}/checkpoint-{step}"

        for attempt in range(retries):
            try:
                print(
                    f"[Forge.Hub] Uploading checkpoint-{step} to '{self.repo_id}' [{self.branch}] "
                    f"(Attempt {attempt + 1}/{retries})..."
                )
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(local_dir),
                    path_in_repo=path_in_repo,
                    commit_message=f"Upload checkpoint-{step}",
                    allow_patterns=["*"],
                    revision=self.branch,
                )
                
                # ATOMIC POINTER: Update the 'latest' pointer only after successful upload
                self._update_latest_pointer(step)
                
                print(f"[Forge.Hub] Upload complete & verified: {path_in_repo}")
                return
            except Exception as e:
                print(f"[Forge.Hub] ERROR: Upload attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))

    def _update_latest_pointer(self, step: int):
        """Updates a tiny JSON file on the Hub to mark this step as the latest verified state."""
        
        pointer_path = f"{self.remote_prefix}/latest.json"
        data = {"latest_step": step, "timestamp": time.time()}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name
            
        try:
            self.api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=pointer_path,
                repo_id=self.repo_id,
                revision=self.branch,
                commit_message=f"Update latest pointer to step {step}"
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_latest_verified_step(self) -> Optional[int]:
        """Reads the latest.json pointer from the Hub."""
        pointer_path = f"{self.remote_prefix}/latest.json"
        try:
            # We use hf_hub_download to get the tiny pointer file
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=pointer_path,
                revision=self.branch,
                token=self.token
            )
            with open(local_path, "r") as f:
                return json.load(f).get("latest_step")
        except Exception:
            return None

    def prune_checkpoints(self, keep: Optional[int] = None) -> None:
        """Keep only the N most recent checkpoints on the remote branch."""
        keep = keep if keep is not None else self.cfg.checkpoint_limit
        steps = self.list_remote_checkpoints()
        if len(steps) <= keep:
            return

        to_delete = steps[:-keep]
        for step in to_delete:
            path_in_repo = f"{self.remote_prefix}/checkpoint-{step}"
            try:
                self.api.delete_folder(
                    repo_id=self.repo_id,
                    path_in_repo=path_in_repo,
                    commit_message=f"Prune old checkpoint-{step}",
                    revision=self.branch,
                )
                print(f"[Forge.Hub] Pruned: {path_in_repo}")
            except Exception as e:
                print(f"[Forge.Hub] ERROR: Deletion failed for {path_in_repo}: {e}")

    def list_remote_checkpoints(self) -> List[int]:
        """List checkpoint steps currently on the Hub branch."""
        try:
            repo_tree = self.api.list_repo_tree(
                repo_id=self.repo_id,
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

    def is_valid_checkpoint(self, local_dir: str | Path) -> bool:
        """Check if a checkpoint folder contains essential files for resuming."""
        local_dir = Path(local_dir)
        # We look for common markers. Since we are engine-agnostic, 
        # we check for either model.safetensors or pytorch_model.bin 
        has_model = (local_dir / "model.safetensors").exists() or (local_dir / "pytorch_model.bin").exists() or (local_dir / "model.bin").exists()
        return has_model

    def ship_model(self, local_root: str | Path) -> Optional[str]:
        """
        Takes the latest checkpoint, prunes it, and pushes to 'main' branch.
        """
        latest_verified = self.get_latest_verified_step()
        if latest_verified is None:
            # Fallback to listing if pointer is missing
            remote_steps = self.list_remote_checkpoints()
            if not remote_steps:
                print("[Forge.Hub] Error: No checkpoints found to ship.")
                return None
            latest_verified = remote_steps[-1]

        # 1. Download to temporary location
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"[Forge.Hub] Downloading checkpoint-{latest_verified} for shipping...")
            checkpoint_path = self.pull_latest(tmp_dir, force=True)
            if not checkpoint_path:
                return None

            # 2. Prune optimizer and scheduler states
            print("[Forge.Hub] Pruning optimizer and scheduler states...")
            patterns_to_delete = [
                "optimizer.pt", "optimizer.bin", "pytorch_optimizer.bin",
                "scheduler.pt", "rng_state.pth", "trainer_state.json",
                "data_state.json" # Not needed for final weights
            ]
            for pattern in patterns_to_delete:
                for p in checkpoint_path.glob(pattern):
                    if p.is_file():
                        p.unlink()

            # 3. Generate README.md (Model Card)
            metadata_file = checkpoint_path / "forge_metadata.json"
            readme_content = self._generate_readme(metadata_file)
            with open(checkpoint_path / "README.md", "w") as f:
                f.write(readme_content)

            # 4. Upload to main branch
            print(f"[Forge.Hub] Uploading pruned model to '{self.repo_id}' [main]...")
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(checkpoint_path),
                path_in_repo=".",
                commit_message=f"Ship model from checkpoint-{latest_verified} via Forge",
                revision="main"
            )
            return f"https://huggingface.co/{self.repo_id}"

    def _generate_readme(self, metadata_file: Path) -> str:
        """Generates a basic Model Card from Forge metadata."""
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        name = metadata.get("project_name", self.project_name)
        step = metadata.get("global_step", "unknown")
        loss = metadata.get("loss", "unknown")
        hw = metadata.get("hardware", "unknown")

        return f"""---
license: other
library_name: transformers
tags:
- forge
- nomad-training
---

# {name}

This model was trained using [Forge](https://github.com/your-repo/forge).

## Training Details
- **Final Step**: {step}
- **Final Loss**: {loss}
- **Hardware**: {hw}
- **Orchestration**: Nomad Training via Forge

## Checkpoint Source
Exported from atomic checkpoint `{step}` on the `checkpoints` branch.
"""

    def get_local_checkpoints(self, local_root: str | Path) -> List[int]:
        """List checkpoint steps currently in the local root."""
        local_root = Path(local_root)
        if not local_root.exists():
            return []
        
        steps = []
        for item in local_root.iterdir():
            if item.is_dir() and "checkpoint-" in item.name:
                step_str = item.name.split("checkpoint-")[-1]
                if step_str.isdigit():
                    steps.append(int(step_str))
        return sorted(steps)

    def pull_latest(self, local_root: str | Path, force: bool = False) -> Optional[Path]:
        """
        Download the latest valid checkpoint from the Hub.
        If force=True, clears the local_root first.
        """
        local_root = Path(local_root)
        
        # Try to get the latest verified step via pointer first
        latest_verified = self.get_latest_verified_step()
        remote_steps = self.list_remote_checkpoints()
        
        if not remote_steps:
            print("[Forge.Hub] No remote checkpoints found.")
            return None

        # Fallback to highest folder if pointer is missing
        latest_remote = latest_verified if latest_verified is not None else remote_steps[-1]
        
        local_steps = self.get_local_checkpoints(local_root)
        latest_local = local_steps[-1] if local_steps else -1

        if latest_remote <= latest_local and not force:
            print(f"[Forge.Hub] Local state (step {latest_local}) is up-to-date with Hub (step {latest_remote}).")
            return local_root / f"checkpoint-{latest_local}"

        if force and local_root.exists():
            print(f"[Forge.Hub] Force pull requested. Clearing {local_root}...")
            shutil.rmtree(local_root)
            local_root.mkdir(parents=True, exist_ok=True)

        # Download specifically the verified step
        target_steps = [latest_remote] if latest_remote in remote_steps else reversed(remote_steps)
        
        for latest_step in target_steps:
            path_in_repo = f"{self.remote_prefix}/checkpoint-{latest_step}"
            target_path = local_root.resolve() / f"checkpoint-{latest_step}"

            print(f"[Forge.Hub] Pulling checkpoint-{latest_step}...")
            try:
                # Optimized download: target the specific checkpoint folder directly
                snapshot_download(
                    repo_id=self.repo_id,
                    allow_patterns=[f"{path_in_repo}/*"],
                    local_dir=str(local_root),
                    token=self.token,
                    revision=self.branch,
                    local_dir_use_symlinks=False # Better for some environments
                )
                
                # After snapshot_download with local_dir, the files are at local_root/remote_prefix/checkpoint-N
                downloaded_path = local_root / path_in_repo
                if downloaded_path.exists():
                    if target_path.exists() and target_path != downloaded_path:
                        shutil.rmtree(target_path)
                    
                    if target_path != downloaded_path:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(downloaded_path), str(target_path))
                        
                        # Cleanup the empty remote_prefix folders
                        current = downloaded_path.parent
                        while current != local_root:
                            try:
                                current.rmdir()
                                current = current.parent
                            except OSError:
                                break

                if self.is_valid_checkpoint(target_path):
                    print(f"[Forge.Hub] Pull complete: {target_path}")
                    return target_path
            except Exception as e:
                print(f"[Forge.Hub] ERROR: Pull failed: {e}")

        return None
