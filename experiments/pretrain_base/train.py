"""
experiments/pretrain_base/train.py

Pretraining entry point for lm_forge 85M base model.

Phases:
  --phase 1    Train on HuggingFaceFW/fineweb-edu (general knowledge)
  --phase 2    Continue on bigcode/the-stack-v2 (code) from Phase 1 checkpoint

Modes:
  (default)    GPU training (auto-detects CUDA)
  --t4         T4 overrides: batch=8, grad_accum=32, fp16
  --smoke      CPU smoke test: synthetic data, 50 steps

Examples:
  # Phase 1 on GPU (Modal / local A100)
  python experiments/pretrain_base/train.py --phase 1

  # Phase 1 on T4
  python experiments/pretrain_base/train.py --phase 1 --t4

  # Phase 2 resuming from Phase 1
  python experiments/pretrain_base/train.py --phase 2 --resume checkpoints/pretrain_phase1/checkpoint-38000

  # Phase 2 on T4
  python experiments/pretrain_base/train.py --phase 2 --resume <path> --t4

  # CPU smoke test
  python experiments/pretrain_base/train.py --phase 1 --smoke

  # Custom config
  python experiments/pretrain_base/train.py --phase 1 --config my_config.yaml

  # Override steps
  python experiments/pretrain_base/train.py --phase 1 --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer as HFTrainer,
)

from engine import load_experiment_config, HFCausalLM, prepare_dataset
from engine.config.hf_config import LMForgeConfig
from engine.utils import ProfilingCallback
from data_streaming import (
    build_fineweb_edu,
    build_stack_v2,
    SyntheticDataset,
)


# ─── CLI ────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="lm_forge pretraining")
    p.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=[1, 2],
        help="Training phase: 1=fineweb-edu, 2=the-stack-v2",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config_phase<phase>.yaml)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    p.add_argument(
        "--t4",
        action="store_true",
        help="Apply T4 GPU overrides (batch=8, grad_accum=32, fp16)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke test with synthetic data (50 steps)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Use local/downloaded datasets instead of streaming (for offline use)",
    )
    return p.parse_args()


# ─── Dataset Builders ───────────────────────────────────────────────────────


def get_datasets(
    phase: int,
    tokenizer,
    seq_len: int,
    smoke: bool = False,
    no_streaming: bool = False,
    vocab_size: int = 49152,
):
    """
    Build train and eval datasets for the given phase.

    Returns (train_ds, eval_ds).
    For streaming datasets, eval_ds is None (HF Trainer skips eval in that case).
    """
    if smoke:
        print("[data] Using synthetic data (smoke test)")
        return SyntheticDataset(2000, seq_len, vocab_size), SyntheticDataset(
            400, seq_len, vocab_size
        )

    if no_streaming:
        # Fallback: load a small slice from HF (downloading full datasets is impractical)
        from datasets import load_dataset as load_hf_ds

        if phase == 1:
            print("[data] Loading fineweb-edu (non-streaming, small slice)...")
            raw = load_hf_ds("HuggingFaceFW/fineweb-edu", split="train[:50000]")
            train_ds = prepare_dataset(raw, tokenizer, seq_len=seq_len)
            raw_eval = load_hf_ds(
                "HuggingFaceFW/fineweb-edu", split="train[50000:52000]"
            )
            eval_ds = prepare_dataset(raw_eval, tokenizer, seq_len=seq_len)
        else:
            print("[data] Loading the-stack-v2 Python (non-streaming, small slice)...")
            raw = load_hf_ds("bigcode/the-stack-v2", "Python", split="train[:50000]")
            raw = raw.rename_column("content", "text")
            train_ds = prepare_dataset(raw, tokenizer, seq_len=seq_len)
            raw_eval = load_hf_ds(
                "bigcode/the-stack-v2", "Python", split="train[50000:52000]"
            )
            raw_eval = raw_eval.rename_column("content", "text")
            eval_ds = prepare_dataset(raw_eval, tokenizer, seq_len=seq_len)

        return train_ds, eval_ds

    # Streaming mode (default for real training)
    if phase == 1:
        train_ds = build_fineweb_edu(tokenizer, seq_len=seq_len)
    else:
        # Top languages by data volume in the-stack-v2
        code_languages = [
            "Python",
            "JavaScript",
            "Java",
            "C__",
            "TypeScript",
            "C",
            "Go",
            "Rust",
        ]
        train_ds = build_stack_v2(tokenizer, seq_len=seq_len, languages=code_languages)

    # No separate eval for streaming (use a held-out subset or skip)
    eval_ds = None
    return train_ds, eval_ds


# ─── T4 Overrides ───────────────────────────────────────────────────────────


def apply_t4_overrides(exp):
    """Override config for T4 GPU (16GB VRAM)."""
    print("[config] Applying T4 overrides: batch=8, grad_accum=32, fp16")
    exp.training.batch_size = 8
    exp.training.grad_accum = 32
    exp.training.dtype = "float16"
    # Update HF args if present
    if hasattr(exp.training, "hf_args") and isinstance(exp.training.hf_args, dict):
        exp.training.hf_args.pop("optim", None)  # let HF choose safe default for fp16
    return exp


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # 1. Load config
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = Path(__file__).parent / f"config_phase{args.phase}.yaml"

    exp = load_experiment_config(cfg_path)

    if args.t4:
        exp = apply_t4_overrides(exp)
    if args.smoke:
        exp.training.backend = "cpu"
        exp.training.dtype = "float32"
        exp.training.num_workers = 0
        exp.training.pin_memory = False
        # Use tiny model for CPU smoke test (85M is too slow on CPU)
        exp.model.num_layers = 2
        exp.model.hidden_size = 64
        exp.model.max_seq_len = 128
        exp.model.attention.num_heads = 2
        exp.model.attention.num_kv_heads = 2
        exp.model.ffn.intermediate_size = 128
        exp.model.tie_word_embeddings = False
        exp.training.seq_len = 64
        exp.training.batch_size = 2
    if args.steps:
        exp.training.max_steps = args.steps

    # 2. Print config summary
    phase_name = "fineweb-edu" if args.phase == 1 else "the-stack-v2"
    print(f"\n{'=' * 60}")
    print(f"  Pretrain Phase {args.phase}: {phase_name}")
    print(f"  Model: ~85M params (16L, 512H, GQA-8Q/2KV, RoPE, SwiGLU)")
    print(f"  Steps: {exp.training.max_steps:,}")
    print(
        f"  Batch: {exp.training.batch_size} x {exp.training.grad_accum} = "
        f"{exp.training.effective_batch_size} effective"
    )
    print(f"  Seq len: {exp.training.seq_len}")
    print(f"  LR: {exp.training.lr}")
    print(f"  Dtype: {exp.training.dtype}")
    print(f"  Backend: {exp.training.backend}")
    if args.resume:
        print(f"  Resume: {args.resume}")
    print(f"{'=' * 60}\n")

    # 3. Tokenizer
    print("Loading SmolLM tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Model
    print("Building model...")
    HFCausalLM.register()

    if args.resume and not args.smoke:
        # Load from checkpoint (model + config come from checkpoint dir)
        print(f"Loading model from checkpoint: {args.resume}")
        model = HFCausalLM.from_pretrained(args.resume)
    else:
        # Fresh model from config
        hf_cfg = LMForgeConfig.from_model_config(exp.model)
        hf_cfg.use_cache = False
        model = HFCausalLM(hf_cfg)

    if not args.smoke:
        model.lm.model.enable_gradient_checkpointing()
    n_params = model.num_parameters()
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # 5. Data
    print("Preparing datasets...")
    train_ds, eval_ds = get_datasets(
        phase=args.phase,
        tokenizer=tokenizer,
        seq_len=exp.training.seq_len,
        smoke=args.smoke,
        no_streaming=args.no_streaming,
        vocab_size=exp.model.vocab_size,
    )

    # 6. Training Arguments
    output_dir = f"checkpoints/{exp.name}"

    training_args_dict = dict(
        output_dir=output_dir,
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
        save_steps=exp.training.save_every,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        push_to_hub=bool(exp.hub.repo_id),
        hub_model_id=exp.hub.repo_id or None,
        hub_private_repo=exp.hub.private,
        report_to=["none"],
        torch_compile=exp.training.compile,
        dataloader_num_workers=max(0, exp.training.num_workers),
    )

    # Add eval config if we have eval data
    if eval_ds is not None:
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["eval_steps"] = exp.training.eval_every
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "loss"
    else:
        # Streaming mode — no eval
        training_args_dict["eval_strategy"] = "no"

    # Merge YAML hf_args overrides
    if exp.training.hf_args:
        training_args_dict.update(exp.training.hf_args)

    training_args = TrainingArguments(**training_args_dict)

    # 7. Trainer
    callbacks = []
    if not args.smoke:
        callbacks.append(ProfilingCallback())

    hf_trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # 8. Resume from checkpoint
    resume_from = None
    if args.resume and Path(args.resume).exists():
        resume_from = args.resume
        print(f"\nResuming training from: {resume_from}")
    elif args.resume:
        print(f"\nWARNING: Checkpoint not found: {args.resume}")
        print("Starting from scratch.\n")
    else:
        # Auto-detect latest checkpoint in output_dir
        ckpt_dir = Path(output_dir)
        if ckpt_dir.exists():
            ckpts = sorted(
                ckpt_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if ckpts:
                resume_from = str(ckpts[-1])
                print(f"\nAuto-resuming from: {resume_from}")

    # 9. Train
    print(f"\nStarting Phase {args.phase} training...")
    hf_trainer.train(resume_from_checkpoint=resume_from)

    # 10. Save final model + tokenizer
    final_path = f"{output_dir}/final"
    print(f"\nSaving final model to: {final_path}")
    hf_trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # 11. Quick generation test
    if not args.smoke:
        print("\n--- Quick Generation Test ---")
        from transformers import pipeline as hf_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = hf_pipeline(
            "text-generation",
            model=final_path,
            tokenizer=tokenizer,
            device=device,
        )

        if args.phase == 1:
            prompts = [
                "The history of science begins with",
                "In mathematics, the concept of",
                "Water is composed of",
            ]
        else:
            prompts = [
                "def fibonacci(n):",
                "import os\n\ndef list_files(",
                "class DataProcessor:\n    def __init__(self):",
            ]

        for prompt in prompts:
            output = gen(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
            )
            print(f"\n  Prompt: {prompt}")
            print(f"  Output: {output[0]['generated_text'][:200]}")

    print(f"\nPhase {args.phase} complete!")
    print(f"Checkpoint: {final_path}")
    if args.phase == 1:
        print(f"\nNext: Run Phase 2 with:")
        print(
            f"  python experiments/pretrain_base/train.py --phase 2 --resume {final_path}"
        )


if __name__ == "__main__":
    main()
