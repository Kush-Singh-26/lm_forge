"""
lm_forge CLI entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from engine import load_experiment_config, HFCausalLM
from experiments.pretrain_base.train import main as pretrain_main


def train_cmd(args):
    """Bridge to the pretraining entry point."""
    # We modify sys.argv and call pretrain_main to reuse its complex logic
    # but with CLI overrides from main.py
    sys.argv = [sys.argv[0]] + args.extra
    pretrain_main()


def profile_cmd(args):
    """Benchmark a model configuration."""
    from engine.utils import Profiler
    from engine.config.hf_config import LMForgeConfig

    cfg = load_experiment_config(args.config)
    hf_cfg = LMForgeConfig.from_model_config(cfg.model)
    model = HFCausalLM(hf_cfg).to(args.device)

    print(f"Profiling model: {cfg.name}")
    print(f"Device: {args.device}")

    # Simple benchmark
    results = Profiler.benchmark(
        model,
        # Mocking DeviceManager behavior
        type(
            "DM",
            (),
            {
                "device": torch.device(args.device),
                "autocast": lambda: torch.amp.autocast(args.device),
            },
        )(),
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        n_steps=args.steps,
    )

    import json

    print(json.dumps(results, indent=2))


def eval_cmd(args):
    """Run evaluation metrics on a model."""
    from engine.eval.metrics import calculate_perplexity
    from engine.data import build_dataloader
    from transformers import AutoTokenizer
    import os

    device = torch.device(args.device)
    print(f"Loading model from {args.model}...")
    model = HFCausalLM.from_pretrained(args.model).to(device)

    # Check if we need to load a specific tokenizer
    tokenizer_name = (
        args.tokenizer or model.config.name_or_path or "HuggingFaceTB/SmolLM-135M"
    )
    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if args.dataset:
        from datasets import load_dataset

        print(f"Loading dataset {args.dataset}...")
        dataset = load_dataset(args.dataset, split=args.split or "validation")

        # Simple tokenization & mapping for evaluation
        # Note: In a production setting, we'd use our internal loaders
        # but for a quick CLI eval, this is enough.
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.seq_len)

        tokenized = dataset.map(
            tokenize, batched=True, remove_columns=dataset.column_names
        )

        from engine.data.collators import CLMCollator

        collator = CLMCollator(
            pad_id=tokenizer.pad_token_id or 0, max_seq_len=args.seq_len
        )

        dataloader = torch.utils.data.DataLoader(
            tokenized, batch_size=args.batch_size, collate_fn=collator
        )

        ppl = calculate_perplexity(model, dataloader, device)
        print(f"\nPerplexity: {ppl:.4f}")

    # Zero-shot benchmarks via lm-evaluation-harness if tasks are provided
    if args.tasks:
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            from lm_eval.evaluator import simple_evaluate

            print(f"Running zero-shot benchmarks: {args.tasks}")

            # Wrap our model for lm-eval
            lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer)

            results = simple_evaluate(
                model=lm_eval_model,
                tasks=args.tasks.split(","),
                device=args.device,
                batch_size=args.batch_size,
            )

            import json

            print(json.dumps(results["results"], indent=2))

        except ImportError:
            print(
                "Error: lm-evaluation-harness not installed. Run: pip install lm-eval"
            )


def main():
    p = argparse.ArgumentParser(description="lm_forge CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # Train
    train_p = sub.add_parser("train", help="Run a training experiment")
    train_p.add_argument("--config", type=str, required=True)
    train_p.add_argument(
        "extra", nargs=argparse.REMAINDER, help="Arguments passed to train.py"
    )
    train_p.set_defaults(func=train_cmd)

    # Profile
    prof_p = sub.add_parser("profile", help="Profile model throughput")
    prof_p.add_argument("--config", type=str, required=True)
    prof_p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    prof_p.add_argument("--steps", type=int, default=20)
    prof_p.set_defaults(func=profile_cmd)

    # Eval
    eval_p = sub.add_parser("eval", help="Evaluate a model")
    eval_p.add_argument(
        "--model", type=str, required=True, help="Path to HF model directory"
    )
    eval_p.add_argument("--dataset", type=str, help="HF dataset name for perplexity")
    eval_p.add_argument("--split", type=str, default="validation")
    eval_p.add_argument("--tokenizer", type=str, help="Tokenizer name (optional)")
    eval_p.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of zero-shot tasks (requires lm-eval)",
    )
    eval_p.add_argument("--batch_size", type=int, default=8)
    eval_p.add_argument("--seq_len", type=int, default=1024)
    eval_p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    eval_p.set_defaults(func=eval_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
