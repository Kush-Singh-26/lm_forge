"""
forge/providers/modal_provider.py

Orchestrates training jobs on Modal.
Conforms to Modal 1.0+ global-scope function definition and dynamic options standards.
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import modal
from forge.config import ModalConfig, ForgeConfig

# 1. Define the Modal App globally to comply with Modal's global-scope requirement
app = modal.App("forge-remote")

# Build the image dynamically at module load time
requirements = []
if Path("requirements.txt").exists():
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

base_requirements = [
    "torch", "transformers", "datasets", "accelerate",
    "huggingface_hub[cli]", "safetensors", "pyyaml",
    "pydantic", "typer", "wandb", "rich", "watchfiles",
    "python-dotenv"
]
# De-duplicate
install_list = list(set(base_requirements + requirements))

# Locate the local 'forge' library dynamically on the host machine
import forge
forge_dir = Path(forge.__file__).parent.parent.resolve()

remote_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(*install_list)
    # Add active project directory dynamically via modern add_local_dir
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[".venv", "__pycache__", ".git", "checkpoints", "outputs"]
    )
    # Add sibling lm_forge orchestrator directory dynamically
    .add_local_dir(
        str(forge_dir),
        remote_path="/root/lm_forge",
        ignore=[".venv", "__pycache__", ".git", "outputs"]
    )
)

# Dynamically build secrets list
remote_secrets = []
if Path(".env").exists():
    remote_secrets.append(modal.Secret.from_dotenv())
elif os.environ.get("HF_TOKEN"):
    remote_secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))
else:
    # Fallback to a named secret if it exists on the dashboard
    remote_secrets.append(modal.Secret.from_name("huggingface", required_empty=False))

hf_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# 2. Define the remote function globally to satisfy Modal's serialization scope rules
@app.function(image=remote_image, secrets=remote_secrets, volumes={"/root/.cache/huggingface": hf_volume})
def remote_train(script_path: str, args: List[str]):
    # 0. Zero-Install: Ensure sibling orchestrator package is installed in container
    import subprocess
    import os
    import sys
    
    print("[Forge.Modal] Installing sibling lm_forge package inside container...")
    try:
        subprocess.run(["pip", "install", "-e", "/root/lm_forge"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[Forge.Modal] PIP INSTALL FAILED!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        raise RuntimeError(f"Pip install failed: {e.stderr}")
    
    os.chdir("/root/project")
    # Note: We do NOT run `pip install -e .` here because the project may not have a build-backend configured.
    # Dependencies are already installed via requirements.txt in the Image build step.
    
    # 1. Automatic Resumption Pull — pull into ./outputs so ForgeTrainer.train() can find it
    print(f"[Forge.Modal] Checking for latest checkpoint on Hub...")
    from forge.state.hub_manager import HubManager
    from forge.config import ForgeConfig
    
    cfg = ForgeConfig.load("forge.yaml")
    hub = HubManager(cfg.state, cfg.name)
    
    # CRITICAL: pull into the same directory that TrainingArguments.output_dir points to
    # so that ForgeTrainer.train() auto-detection finds the checkpoint correctly.
    hub.pull_latest("outputs")

    
    # 2. Run the user script
    cmd = ["python", "-u", script_path] + args
    print(f"[Forge.Modal] Executing: {' '.join(cmd)}")
    sys.stdout.flush()
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    
    output_log = []
    for line in iter(process.stdout.readline, ''):
        print(line, end="")
        sys.stdout.flush()
        output_log.append(line)
        if len(output_log) > 100:
            output_log.pop(0)
            
    process.wait()
    if process.returncode != 0:
        error_tail = "".join(output_log)
        raise RuntimeError(f"Training script failed with exit code {process.returncode}.\nOutput Tail:\n{error_tail}")

class ModalProvider:
    """
    Handles remote execution on Modal.
    """

    def __init__(self, forge_cfg: ForgeConfig) -> None:
        self.cfg = forge_cfg
        self.modal_cfg = forge_cfg.providers.modal
        self.app = app

    def run(self, script_path: str, args: List[str]):
        """
        Launches the remote training job.
        """
        modal_cfg = self.modal_cfg
        
        # Override options dynamically at runtime using .with_options() 
        # (image and secrets are not supported here, only hardware/scheduling configs)
        configured_train = remote_train.with_options(
            gpu=modal_cfg.gpu,
            cpu=modal_cfg.cpu,
            memory=modal_cfg.memory,
            timeout=modal_cfg.timeout,
        )
        
        with self.app.run():
            configured_train.remote(script_path, args)
