"""
experiments/ablation_001/run_ablation.py

Runs the full ablation sweep defined in ablation.yaml and prints a ranked
results table.  Each variant trains for max_steps steps on synthetic data
(swap for your real dataset).

Usage:
    python experiments/ablation_001/run_ablation.py
    python experiments/ablation_001/run_ablation.py --cpu --steps 50

Results are written to experiments/ablation_001/ablation_results.json.
Already-completed variants are skipped automatically, so you can interrupt
and resume this script at any point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader, Dataset

from engine import CausalLM, load_experiment_config
from engine.config.schema import ExperimentConfig
from engine.legacy.training import DeviceManager, Trainer, build_scheduler
from engine.legacy.training.hub import HubSync
from engine.utils import AblationRunner


class SyntheticDataset(Dataset):
    def __init__(self, vocab_size, seq_len, n=500):
        gen = torch.Generator().manual_seed(42)
        self.data = torch.randint(0, vocab_size, (n, seq_len), generator=gen)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"input_ids": self.data[i], "labels": self.data[i]}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# CLI args are set globally so the train_fn can see them
_ARGS = None


def train_fn(exp: ExperimentConfig) -> float:
    """Train one variant, return eval loss."""
    if _ARGS and _ARGS.cpu:
        exp.training.backend = "cpu"
        exp.training.dtype = "float32"
    if _ARGS and _ARGS.steps:
        exp.training.max_steps = _ARGS.steps

    dm = DeviceManager(exp.training)
    hub = HubSync(exp.hub, exp)  # Hub disabled if no repo_id / token

    model = dm.prepare(CausalLM(exp.model))

    train_ds = SyntheticDataset(exp.model.vocab_size, exp.training.seq_len, n=500)
    eval_ds = SyntheticDataset(exp.model.vocab_size, exp.training.seq_len, n=100)

    train_loader = DataLoader(
        train_ds,
        batch_size=exp.training.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=exp.training.batch_size, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp.training.lr,
        weight_decay=exp.training.weight_decay,
        betas=tuple(exp.training.betas),
    )
    scheduler = build_scheduler(optimizer, exp.training)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dm=dm,
        exp_cfg=exp,
        hub=hub,
        scheduler=scheduler,
    )
    trainer.train(train_loader)
    return trainer.evaluate(eval_loader)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument(
        "--skip",
        action="store_true",
        help="Skip already-completed variants (default: True)",
    )
    return p.parse_args()


if __name__ == "__main__":
    _ARGS = parse_args()

    runner = AblationRunner(
        ablation_yaml=Path(__file__).parent / "ablation.yaml",
    )
    results = runner.run(train_fn, skip_existing=True)
