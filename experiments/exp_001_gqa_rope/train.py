"""
experiments/exp_001_gqa_rope/train.py

Two training paths — pick one via --trainer flag:

  --trainer engine   (default)  lm_forge Trainer + HubSync
  --trainer hf                  HuggingFace Trainer

Data workflow (run Step 0 once, everything else is automatic):

  Step 0 — tokenize & push to Hub (run once, anywhere):
    python -m engine.data.pretokenize \\
        --dataset roneneldan/TinyStories \\
        --tokenizer gpt2 \\
        --hub_repo YOUR_HF_USERNAME/tinystories-gpt2 \\
        --output_dir data/tinystories_gpt2

  Step 1 — set in config.yaml:
    training:
      data_hub_repo: "YOUR_HF_USERNAME/tinystories-gpt2"
      vocab_size: 50257   # must match the tokenizer

  Step 2 — train (Colab or local):
    python experiments/exp_001_gqa_rope/train.py

  Subsequent sessions: the MemmapDataset.from_hub() call skips
  already-downloaded files, so startup is instant after the first pull.

CPU smoke-test (no GPU, no HF token, no real data):
    python experiments/exp_001_gqa_rope/train.py --cpu --steps 30 --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import Dataset

from engine import CausalLM, load_experiment_config
from engine.config import LMForgeConfig
from engine.config.schema import ExperimentConfig
from engine.models.hf_model import HFCausalLM
from engine.legacy.training import DeviceManager, HubSync, Trainer, build_scheduler
from engine.legacy.data import MemmapDataset, PackedDataset
from engine.data import build_dataloader
from engine.data.hf_utils import prepare_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset resolution — real data first, synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────


def get_datasets(
    exp: ExperimentConfig, force_synthetic: bool = False, hf_native: bool = False
):
    """
    Resolve train + val datasets.

    hf_native: if True, returns raw HF datasets prepared via prepare_dataset()
    """
    t = exp.training

    if hf_native and not force_synthetic:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer

            # Default to TinyStories for this experiment
            print("[data] Loading roneneldan/TinyStories (HF native)...")
            train_raw = load_dataset("roneneldan/TinyStories", split="train")
            val_raw = load_dataset("roneneldan/TinyStories", split="validation")

            # Cap for quick demo if needed, otherwise use full
            # train_raw = train_raw.select(range(50000))
            # val_raw = val_raw.select(range(2000))

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            train_ds = prepare_dataset(train_raw, tokenizer, seq_len=t.seq_len)
            val_ds = prepare_dataset(val_raw, tokenizer, seq_len=t.seq_len)

            return train_ds, val_ds, tokenizer.vocab_size
        except Exception as e:
            print(f"[data] HF native load failed ({e}) — falling back")

    # ── Path 1: pre-tokenized memmap ─────────────────────────────────────────
    if not force_synthetic and (t.data_hub_repo or t.data_dir):
        try:
            if t.data_dir and Path(t.data_dir).exists():
                print(f"[data] Loading pre-tokenized data from: {t.data_dir}")
                train_ds = MemmapDataset.from_dir(
                    t.data_dir, split="train", seq_len=t.seq_len
                )
                val_ds = MemmapDataset.from_dir(
                    t.data_dir, split="val", seq_len=t.seq_len
                )
            elif t.data_hub_repo:
                print(f"[data] Pulling pre-tokenized data from Hub: {t.data_hub_repo}")
                train_ds = MemmapDataset.from_hub(
                    t.data_hub_repo, split="train", seq_len=t.seq_len
                )
                val_ds = MemmapDataset.from_hub(
                    t.data_hub_repo, split="val", seq_len=t.seq_len
                )
            else:
                raise FileNotFoundError("data_dir not found and data_hub_repo not set")

            print(f"[data] {train_ds}")
            print(f"[data] {val_ds}")
            return train_ds, val_ds, None

        except Exception as e:
            print(f"[data] Memmap load failed ({e}) — trying HF dataset fallback")

    # ── Path 2: HF dataset on-the-fly ────────────────────────────────────────
    if not force_synthetic:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer

            print("[data] Loading roneneldan/TinyStories via HF datasets...")
            hf_ds = load_dataset("roneneldan/TinyStories", split="train")
            hf_ds_val = load_dataset("roneneldan/TinyStories", split="validation")
            tok = AutoTokenizer.from_pretrained("gpt2")

            train_ds = PackedDataset.from_hf(
                hf_ds,
                tok,
                seq_len=t.seq_len,
                eos_id=tok.eos_token_id,
                max_samples=50_000,
            )
            val_ds = PackedDataset.from_hf(
                hf_ds_val,
                tok,
                seq_len=t.seq_len,
                eos_id=tok.eos_token_id,
                max_samples=2_000,
            )
            print(f"[data] HF dataset loaded: {len(train_ds):,} train chunks")
            return train_ds, val_ds, tok.vocab_size

        except Exception as e:
            print(f"[data] HF dataset failed ({e}) — falling back to synthetic data")

    # ── Path 3: synthetic ─────────────────────────────────────────────────────
    print("[data] Using synthetic random tokens (smoke test only)")

    class _SyntheticDS(Dataset):
        def __init__(self, n):
            self.data = torch.randint(0, exp.model.vocab_size, (n, t.seq_len))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return {"input_ids": self.data[i], "labels": self.data[i]}

    return _SyntheticDS(2000), _SyntheticDS(400), None


# ─────────────────────────────────────────────────────────────────────────────
# Path A — engine Trainer
# ─────────────────────────────────────────────────────────────────────────────


def train_with_engine(exp: ExperimentConfig, args):
    dm = DeviceManager(exp.training)
    hub = HubSync(exp.hub, exp)

    # Pull latest checkpoint (no-op if Hub disabled or fresh run)
    ckpt_root = Path("checkpoints") / exp.name
    resume = hub.pull_latest(ckpt_root)

    # Build or reload model
    if resume and (resume / "model_config.json").exists():
        model = CausalLM.from_pretrained(resume, map_location=str(dm.device))
    else:
        model = CausalLM(exp.model)

    # Gradient checkpointing — essential for seq_len >= 512 on T4
    model.model.enable_gradient_checkpointing()
    model = dm.prepare(model)

    n = model.num_parameters()
    print(f"\n[engine Trainer]  {n:,} params  ({n / 1e6:.1f}M)")

    # Data
    train_ds, val_ds, vocab_override = get_datasets(exp, force_synthetic=args.synthetic)
    if vocab_override and vocab_override != exp.model.vocab_size:
        print(
            f"[data] Note: tokenizer vocab_size={vocab_override} differs from "
            f"model vocab_size={exp.model.vocab_size}. Update model.vocab_size in config.yaml."
        )

    # DataLoader — num_workers and pin_memory handled by build_dataloader auto-detect
    num_workers = None if exp.training.num_workers == -1 else exp.training.num_workers
    pin_memory = exp.training.pin_memory and dm.device.type == "cuda"

    train_loader = build_dataloader(
        train_ds,
        batch_size=exp.training.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=exp.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Optimizer — fused AdamW on CUDA, standard elsewhere
    optimizer = dm.build_optimizer(
        model,
        lr=exp.training.lr,
        weight_decay=exp.training.weight_decay,
        betas=tuple(exp.training.betas),
        fused=exp.training.fused_adamw,
    )
    scheduler = build_scheduler(optimizer, exp.training)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dm=dm,
        exp_cfg=exp,
        hub=hub,
        scheduler=scheduler,
        resume_from=str(resume) if resume else None,
    )
    trainer.train(train_loader, val_loader)

    # Optional: convert to HF model and push to Hub with model card
    # from engine.utils import generate_model_card
    # from engine.models import HFCausalLM
    # hf_model = HFCausalLM.from_engine_model(model)
    # hf_model.push_to_hub(exp.hub.repo_id)
    # generate_model_card(exp, "checkpoints/README.md", author="your-name")


# ─────────────────────────────────────────────────────────────────────────────
# Path B — HF Trainer
# ─────────────────────────────────────────────────────────────────────────────


def train_with_hf(exp: ExperimentConfig, args):
    try:
        from transformers import Trainer as HFTrainer, TrainingArguments
    except ImportError:
        print("HF Trainer requires: pip install transformers[torch]")
        raise

    HFCausalLM.register()

    hf_cfg = LMForgeConfig.from_model_config(exp.model)
    hf_cfg.use_cache = False
    model = HFCausalLM(hf_cfg)
    model.lm.model.enable_gradient_checkpointing()

    n = model.num_parameters()
    print(f"\n[HF Trainer]  {n:,} params  ({n / 1e6:.1f}M)")

    train_ds, val_ds, _ = get_datasets(
        exp, force_synthetic=args.synthetic, hf_native=True
    )

    # 1. Base arguments from exp_cfg
    training_args_dict = dict(
        output_dir=f"checkpoints/{exp.name}_hf",
        max_steps=exp.training.max_steps,
        per_device_train_batch_size=exp.training.batch_size,
        gradient_accumulation_steps=exp.training.grad_accum,
        learning_rate=exp.training.lr,
        weight_decay=exp.training.weight_decay,
        warmup_ratio=exp.training.warmup_ratio,
        max_grad_norm=exp.training.max_grad_norm,
        bf16=(exp.training.dtype == "bfloat16"),
        fp16=(exp.training.dtype == "float16"),
        logging_steps=exp.training.log_every,
        eval_strategy="steps",
        eval_steps=exp.training.eval_every,
        save_steps=exp.training.save_every,
        save_total_limit=3,
        load_best_model_at_end=True,
        push_to_hub=bool(exp.hub.repo_id),
        hub_model_id=exp.hub.repo_id or None,
        hub_private_repo=exp.hub.private,
        hub_strategy="checkpoint",
        report_to=["none"],
        torch_compile=exp.training.compile,
        dataloader_drop_last=True,
        dataloader_num_workers=max(0, exp.training.num_workers),
        gradient_checkpointing=True,
    )

    # 2. Override with hf_args from YAML
    training_args_dict.update(exp.training.hf_args)

    training_args = TrainingArguments(**training_args_dict)

    hf_trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    ckpt_dir = Path(training_args.output_dir)
    last_ckpt = None
    if ckpt_dir.exists():
        ckpts = sorted(
            ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if ckpts:
            last_ckpt = str(ckpts[-1])
            print(f"[HF Trainer] Resuming from {last_ckpt}")

    hf_trainer.train(resume_from_checkpoint=last_ckpt)
    hf_trainer.save_model()

    if exp.hub.repo_id:
        hf_trainer.push_to_hub(commit_message="Training complete")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trainer", default="hf", choices=["engine", "hf"])
    p.add_argument("--cpu", action="store_true", help="Force CPU backend")

    p.add_argument("--steps", type=int, default=None, help="Override max_steps")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip real data, use random tokens (smoke test)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    exp = load_experiment_config(Path(__file__).parent / "config.yaml")

    if args.cpu:
        exp.training.backend = "cpu"
        exp.training.dtype = "float32"
        exp.training.num_workers = 0  # no forking on CPU smoke tests
        exp.training.pin_memory = False
    if args.steps:
        exp.training.max_steps = args.steps

    print(f"\nExperiment : {exp.name}")
    print(
        f"Attention  : {exp.model.attention.type}  "
        f"({exp.model.attention.num_heads}Q / {exp.model.attention.num_kv_heads}KV)"
    )
    print(f"Positional : {exp.model.positional.type}")
    print(f"FFN        : {exp.model.ffn.type}  |  Norm: {exp.model.norm.type}")
    print(
        f"Data       : {'synthetic' if args.synthetic else exp.training.data_hub_repo or 'auto-detect'}"
    )
    print(f"Trainer    : {args.trainer}\n")

    if args.trainer == "hf":
        train_with_hf(exp, args)
    else:
        train_with_engine(exp, args)


if __name__ == "__main__":
    main()
