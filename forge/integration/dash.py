"""
forge/integration/dash.py

TUI Dashboard for Forge using Rich.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import box

class ForgeDash:
    def __init__(self, output_dir: str = "outputs", refresh_rate: float = 1.0):
        self.output_dir = Path(output_dir)
        self.status_file = self.output_dir / "forge_status.json"
        self.refresh_rate = refresh_rate
        self.console = Console()

    def load_status(self) -> Optional[Dict[str, Any]]:
        if not self.status_file.exists():
            return None
        try:
            with open(self.status_file, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="progress", ratio=2),
        )
        return layout

    def generate_dashboard(self, status: Optional[Dict[str, Any]]) -> Panel:
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        if status:
            step = status.get("step", 0)
            max_steps = status.get("max_steps", 1)
            loss = status.get("loss")
            lr = status.get("learning_rate")
            throughput = status.get("throughput")
            mfu = status.get("mfu")
            
            table.add_row("Step", f"{step:,} / {max_steps:,}")
            table.add_row("Epoch", f"{status.get('epoch', 0):.2f}")
            table.add_row("Loss", f"{loss:.4f}" if loss else "N/A")
            table.add_row("Learning Rate", f"{lr:.2e}" if lr else "N/A")
            table.add_row("Throughput", f"{throughput:.2f} samples/s" if throughput else "N/A")
            if mfu:
                table.add_row("MFU", f"{mfu*100:.1f}%")
        else:
            table.add_row("Status", "Waiting for trainer...")

        return Panel(table, title="[bold]Live Metrics[/bold]", border_style="blue")

    def generate_progress(self, status: Optional[Dict[str, Any]]) -> Panel:
        if not status:
            return Panel("No progress data available", title="Progress")
        
        step = status.get("step", 0)
        max_steps = status.get("max_steps", 1)
        
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("[green]Training", total=max_steps, completed=step)
        
        return Panel(progress, title="[bold]Training Progress[/bold]", border_style="green")

    def run(self):
        layout = self.make_layout()
        
        # Simple header
        layout["header"].update(Panel("[bold yellow]Forge Nomad Training Dashboard[/bold yellow]", text_align="center"))
        layout["footer"].update(Panel(f"Monitoring: {self.status_file}", style="dim"))

        with Live(layout, refresh_per_second=4, screen=True) as live:
            while True:
                status = self.load_status()
                layout["main"]["stats"].update(self.generate_dashboard(status))
                layout["main"]["progress"].update(self.generate_progress(status))
                time.sleep(self.refresh_rate)

def show_dash(output_dir: str = "outputs"):
    dash = ForgeDash(output_dir=output_dir)
    try:
        dash.run()
    except KeyboardInterrupt:
        pass
