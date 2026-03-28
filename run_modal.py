import modal
import subprocess
import sys
import os

# 1. Define the standard Debian container and install the latest packages
env_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=5.0.0",
        "datasets>=2.18.0",
        "huggingface_hub[cli]>=0.20.0",
        "safetensors>=0.4.0",
        "wandb>=0.15.0",
        "accelerate>=0.20.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(
        ".",
        remote_path="/root/lm_forge",
        ignore=[".venv", "**/__pycache__", "**/*.pyc", ".git"],
    )
)

app = modal.App("lm-forge-pretrain", image=env_image)


# 2. Configure the Hardware and Secrets
@app.function(
    gpu="A100",
    cpu=16,
    memory=32768,
    timeout=86400,
    secrets=[modal.Secret.from_dotenv()],
)
def run_training():
    print("🚀 Booting A100-40GB for Phase 1 Pretraining...")

    # Navigate to the mounted directory inside the container
    os.chdir("/root/lm_forge")

    # Define the command from your README
    cmd = ["python", "experiments/pretrain_base/train.py", "--phase", "1"]

    # Execute the command and stream the logs directly back to your local terminal
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()

    process.wait()

    if process.returncode != 0:
        print(f"❌ Training failed with return code {process.returncode}")
        sys.exit(process.returncode)
    else:
        print("✅ Phase 1 Training Complete!")


# 3. Local Entrypoint
@app.local_entrypoint()
def main():
    run_training.remote()
