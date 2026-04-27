"""
forge/providers/modal_provider.py

Orchestrates training jobs on Modal.
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import modal
from forge.config import ModalConfig, ForgeConfig

class ModalProvider:
    """
    Handles remote execution on Modal.
    """

    def __init__(self, forge_cfg: ForgeConfig) -> None:
        self.cfg = forge_cfg
        self.modal_cfg = forge_cfg.providers.modal
        
        # Build the image dynamically
        requirements = []
        if Path("requirements.txt").exists():
            with open("requirements.txt", "r") as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        base_requirements = [
            "torch", "transformers", "datasets", "accelerate",
            "huggingface_hub[cli]", "safetensors", "pyyaml",
            "pydantic", "typer", "wandb", "rich", "watchfiles"
        ]
        # De-duplicate
        install_list = list(set(base_requirements + requirements))

        self.image = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install(*install_list)
        )
        
        self.app = modal.App(f"forge-{self.cfg.name}", image=self.image)

    def run(self, script_path: str, args: List[str]):
        """
        Launches the remote training job.
        """
        project_name = self.cfg.name
        modal_cfg = self.modal_cfg
        
        # We need to define the function inside run or as a member to capture app
        @self.app.function(
            gpu=modal_cfg.gpu,
            cpu=modal_cfg.cpu,
            memory=modal_cfg.memory,
            timeout=modal_cfg.timeout,
            secrets=[modal.Secret.from_dotenv()],
            mounts=[
                modal.Mount.from_local_dir(
                    ".",
                    remote_path="/root/project",
                    condition=lambda p: not any(x in p for x in [".venv", "__pycache__", ".git", "checkpoints", "outputs"])
                )
            ],
        )
        def remote_train():
            os.chdir("/root/project")
            
            # 0. Zero-Install: Ensure local forge and requirements are available
            import subprocess
            print("[Forge.Modal] Installing dependencies from mounted directory...")
            subprocess.run(["pip", "install", "-e", "."], check=True)
            
            # 1. Automatic Resumption Pull
            print(f"[Forge.Modal] Checking for latest checkpoint on Hub...")
            from forge.state.hub_manager import HubManager
            from forge.config import ForgeConfig
            
            cfg = ForgeConfig.load("forge.yaml")
            hub = HubManager(cfg.state, cfg.name)
            
            # Pull to local 'checkpoints' dir in container
            hub.pull_latest("checkpoints")
            
            # 2. Run the user script
            cmd = ["python", script_path] + args
            print(f"[Forge.Modal] Executing: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line, end="")
                sys.stdout.flush()
            
            process.wait()
            if process.returncode != 0:
                sys.exit(process.returncode)

        with self.app.run():
            remote_train.remote()
