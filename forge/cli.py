"""
forge/cli.py

CLI for the Forge Training Assist tool.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from forge.config import ForgeConfig, StateConfig

app = typer.Typer(help="Forge: Training Assist for Nomad Training")

@app.command()
def init(
    name: str = typer.Option(..., help="Project name"),
    repo_id: str = typer.Option(..., help="HF Hub Repo ID"),
):
    """Initialize a new Forge project."""
    if Path("forge.yaml").exists():
        typer.confirm("forge.yaml already exists. Overwrite?", abort=True)
    
    cfg = ForgeConfig(
        name=name,
        state=StateConfig(repo_id=repo_id),
        profiles={
            "default": {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 4},
            "colab": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 16},
            "kaggle": {
                "per_device_train_batch_size": 4, 
                "gradient_accumulation_steps": 8,
                "fp16": True,
                "ddp_find_unused_parameters": False
            },
            "sagemaker": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 16},
        }
    )
    cfg.save("forge.yaml")
    typer.echo("✅ Initialized forge.yaml")
    
    if not Path(".env").exists():
        typer.echo("📝 Creating .env template...")
        with open(".env", "w") as f:
            f.write("HF_TOKEN=your_token_here\n")

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider: modal, local"),
    script: str = typer.Option("train.py", help="Training script to run"),
):
    """Run training on a specific provider."""
    if not Path("forge.yaml").exists():
        typer.echo("❌ Error: forge.yaml not found. Run 'forge init' first.")
        raise typer.Exit(1)
    
    cfg = ForgeConfig.load("forge.yaml")
    
    if provider == "modal":
        from forge.providers.modal_provider import ModalProvider
        typer.echo(f"🚀 Launching {script} on Modal...")
        p = ModalProvider(cfg)
        p.run(script, ctx.args)
    elif provider == "local":
        typer.echo(f"🏠 Running {script} locally...")
        import subprocess
        import sys
        
        # Check if accelerate is available and if there's a config
        use_accelerate = False
        try:
            subprocess.run(["accelerate", "--version"], capture_output=True, check=True)
            use_accelerate = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        if use_accelerate:
            cmd = ["accelerate", "launch", script] + ctx.args
            typer.echo("🚀 Using accelerate launch for local execution.")
        else:
            cmd = ["python", script] + ctx.args
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Error: Local run failed with exit code {e.returncode}.")
            raise typer.Exit(e.returncode)
    else:
        typer.echo(f"❌ Unknown provider: {provider}")

@app.command()
def pull(
    config: str = typer.Option("forge.yaml", help="Path to forge.yaml"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite local state"),
    output_dir: str = typer.Option("outputs", help="Directory to pull into"),
):
    """Pull the latest checkpoint from the Hub manually."""
    if not Path(config).exists():
        typer.echo(f"❌ Error: {config} not found.")
        raise typer.Exit(1)
        
    cfg = ForgeConfig.load(config)
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(config).parent / ".env")
    from forge.state.hub_manager import HubManager
    hub = HubManager(cfg.state, cfg.name)
    
    typer.echo(f"Searching for checkpoints for {cfg.name}...")
    path = hub.pull_latest(output_dir, force=force)
    if path:
        typer.echo(f"✅ State ready at: {path}")
    else:
        typer.echo("No remote checkpoints to pull.")

@app.command()
def ship(
    config: str = typer.Option("forge.yaml", help="Path to forge.yaml"),
    target_repo: Optional[str] = typer.Option(None, "--target-repo", help="HF Hub Repo ID to ship to (default: same repo main branch)"),
):
    """Prunes and exports the latest verified checkpoint to the Hub 'main' branch."""
    if not Path(config).exists():
        typer.echo(f"❌ Error: {config} not found.")
        raise typer.Exit(1)
        
    cfg = ForgeConfig.load(config)
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(config).parent / ".env")
    from forge.state.hub_manager import HubManager
    hub = HubManager(cfg.state, cfg.name)
    
    typer.echo(f"🚀 Shipping latest verified checkpoint for {cfg.name}...")
    url = hub.ship_model("temp_ship", target_repo_id=target_repo)
    if url:
        typer.echo(f"✅ Model shipped to main branch: {url}")
    else:
        typer.echo("❌ Shipping failed.")

@app.command()
def notebook(platform: str = typer.Argument("colab", help="Platform: colab, kaggle, sagemaker")):
    """Generates a setup snippet for various notebook platforms."""
    install_cmd = "!pip install -q git+https://github.com/your-repo/forge.git"
    
    auth_block = ""
    if platform == "colab":
        auth_block = """
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")"""
    elif platform == "kaggle":
        auth_block = """
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")"""
        
        # Multi-GPU Launch Snippet
        launch_block = """
# Detect GPUs and generate launch config
!accelerate config default
!accelerate launch train.py"""
    elif platform == "sagemaker":
        auth_block = """
# Ensure you have set HF_TOKEN in your SageMaker environment variables or .env
# os.environ["HF_TOKEN"] = "your_token" """

    snippet = f"""
# --- Forge {platform.capitalize()} Setup ---
{install_cmd}
import os

{auth_block.strip()}

# Pull latest state and run
!forge pull
!python train.py
# -------------------------
    """
    typer.echo(f"\nCopy this into a {platform.capitalize()} cell:")
    typer.echo("-" * 40)
    typer.echo(snippet)
    typer.echo("-" * 40)

@app.command()
def status(config: str = "forge.yaml"):
    """Shows the status of the project, including local vs remote state."""
    if not Path(config).exists():
        typer.echo(f"❌ Error: {config} not found.")
        raise typer.Exit(1)
        
    cfg = ForgeConfig.load(config)
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(config).parent / ".env")
    from forge.state.hub_manager import HubManager
    hub = HubManager(cfg.state, cfg.name)
    
    typer.echo(f"\n[bold blue]Forge Project: {cfg.name}[/bold blue]")
    typer.echo("-" * 40)
    
    # 1. Environment
    typer.echo(f"Environment: [green]{cfg.detect_env()}[/green]")
    
    # 2. Remote State
    remote_steps = hub.list_remote_checkpoints()
    verified_step = hub.get_latest_verified_step()
    typer.echo(f"Hub Repo: [cyan]{cfg.state.repo_id}[/cyan] ([dim]{cfg.state.branch}[/dim])")
    typer.echo(f"Hub Verified Step: [bold green]{verified_step if verified_step is not None else 'None'}[/bold green]")
    typer.echo(f"Remote Checkpoints: {remote_steps}")
    
    # 3. Local State
    local_steps = hub.get_local_checkpoints("outputs")
    latest_local = local_steps[-1] if local_steps else "None"
    typer.echo(f"Local Latest Step: [bold yellow]{latest_local}[/bold yellow]")
    
    # 4. Sync Status
    if verified_step is not None and latest_local != "None":
        if verified_step > latest_local:
            typer.echo("\n⚠️ [yellow]Local state is behind Hub. Run 'forge pull'.[/yellow]")
        elif verified_step < latest_local:
            typer.echo("\n⬆️ [blue]Local state is ahead of Hub. Waiting for next sync...[/blue]")
        else:
            typer.echo("\n✅ [green]Local and Hub are in sync.[/green]")
    elif verified_step is not None:
        typer.echo("\n⬇️ [yellow]No local checkpoints found. Run 'forge pull'.[/yellow]")
    
    typer.echo("-" * 40 + "\n")

@app.command()
def dash(output_dir: str = typer.Option("outputs", help="Directory containing forge_status.json")):
    """Launch the TUI dashboard to monitor training."""
    from forge.integration.dash import show_dash
    show_dash(output_dir)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def watch(
    ctx: typer.Context,
    command: str = typer.Argument(..., help="Command to watch (e.g. 'forge run modal')"),
    max_retries: int = typer.Option(10, help="Max retries"),
):
    """Resiliency daemon to restart failed or preempted jobs."""
    from forge.integration.watchdog import run_watchdog
    # Combine the command and any extra args
    full_cmd = [command] + ctx.args
    run_watchdog(full_cmd, max_retries=max_retries)

@app.command()
def profile(
    model_id: str = typer.Argument(..., help="HF Model ID or local path"),
    batch_size: int = typer.Option(8, help="Batch size"),
    seq_len: int = typer.Option(512, help="Sequence length"),
    steps: int = typer.Option(20, help="Number of steps to profile"),
):
    """Profile a model's throughput and MFU on the current hardware."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    from forge.integration.profiler import Profiler
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Loading model {model_id} on {device}...")
    
    # Load config first to avoid downloading full model if only counting params is possible
    # but Profiler needs the actual model for some counts and to run steps.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    p = Profiler(model, seq_len=seq_len, batch_size=batch_size, device=device)
    
    typer.echo(f"Profiling {steps} steps...")
    with torch.no_grad():
        for i in range(steps + p.warmup_steps):
            # Dummy input
            input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)
            with p.step():
                model(input_ids)
                
            if i >= p.warmup_steps:
                typer.echo(f"Step {i - p.warmup_steps + 1}/{steps}: {p.report()}")

    typer.echo("\n[bold green]Final Profile Results:[/bold green]")
    typer.echo(f"Model Parameters: {p.stats.model_params / 1e6:.1f}M")
    typer.echo(f"Non-Embedding Params: {p.stats.non_embed_params / 1e6:.1f}M")
    typer.echo(f"Peak Memory: {p.stats.peak_memory_gb:.2f} GB")
    typer.echo(f"Throughput: {p.stats.tokens_per_sec:,.0f} tokens/s")
    mfu = p.mfu()
    if mfu:
        typer.echo(f"MFU: {mfu * 100:.1f}%")

if __name__ == "__main__":
    app()
