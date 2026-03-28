"""
experiments/pretrain_base/train.py

Pretraining entry point for lm_forge 85M base model.
Phases:
  --phase 1    Train on HuggingFaceFW/fineweb-edu (general knowledge)
  --phase 2    Continue on bigcode/the-stack-v2 (code) from Phase 1 checkpoint
"""

from __future__ import annotations

import os

os.environ["TRANSFORMERS_SAFE_SERIALIZATION"] = "0"

import argparse
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer as HFTrainer,
    DataCollatorForLanguageModeling,
)

from engine import (
    load_experiment_config,
    HFCausalLM,
    prepare_dataset,
    build_fineweb_edu,
    build_stack_v2,
    SyntheticDataset,
)
from engine.config.hf_config import LMForgeConfig
from engine.utils import ProfilingCallback, HubCheckpointCallback, HubCheckpointManager


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
    shuffle_buffer: int = 10000,
):
    if smoke:
        print("[data] Using synthetic data (smoke test)")
        return SyntheticDataset(2000, seq_len, vocab_size), SyntheticDataset(
            400, seq_len, vocab_size
        )

    if no_streaming:
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

    if phase == 1:
        train_ds = build_fineweb_edu(
            tokenizer, seq_len=seq_len, shuffle_buffer=shuffle_buffer
        )
    else:
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
        train_ds = build_stack_v2(
            tokenizer,
            seq_len=seq_len,
            languages=code_languages,
            shuffle_buffer=shuffle_buffer,
        )

    eval_ds = None
    return train_ds, eval_ds


# ─── T4 Overrides ───────────────────────────────────────────────────────────


def apply_t4_overrides(exp):
    print("[config] Applying T4 overrides: batch=8, grad_accum=32, fp16")
    exp.training.batch_size = 8
    exp.training.grad_accum = 32
    exp.training.dtype = "float16"
    if hasattr(exp.training, "hf_args") and isinstance(exp.training.hf_args, dict):
        exp.training.hf_args["optim"] = "adamw_torch"
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
        exp.model.num_layers = 2
        exp.model.hidden_size = 64
        exp.model.max_seq_len = 128
        exp.model.attention.num_heads = 2
        exp.model.attention.num_kv_heads = 2
        exp.model.ffn.intermediate_size = 128
        exp.training.seq_len = 64
        exp.training.batch_size = 2
    if args.steps:
        exp.training.max_steps = args.steps

    exp.model.__post_init__()

    # 2. Print config summary
    phase_name = "fineweb-edu" if args.phase == 1 else "the-stack-v2"
    m = exp.model
    print(f"\n{'=' * 60}")
    print(f"  Pretrain Phase {args.phase}: {phase_name}")
    print(
        f"  Model Architecture: {m.num_layers}L, {m.hidden_size}H, "
        f"{m.attention.type}, {m.positional.type}, {m.ffn.type}"
    )
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
    print(f"Loading tokenizer: {exp.training.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(exp.training.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Model
    print("Building model...")
    torch.manual_seed(42)
    HFCausalLM.register()

    if args.resume and not args.smoke:
        print(f"Loading model from checkpoint: {args.resume}")
        model = HFCausalLM.from_pretrained(args.resume)
    else:
        hf_cfg = LMForgeConfig.from_model_config(exp.model)
        hf_cfg.use_cache = False
        model = HFCausalLM(hf_cfg)

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
        shuffle_buffer=exp.training.shuffle_buffer,
    )

    # 6. Training Arguments
    output_dir = f"checkpoints/{exp.name}"

    from engine.data import default_num_workers

    if eval_ds is None:
        workers = 0
    else:
        workers = (
            exp.training.num_workers
            if exp.training.num_workers >= 0
            else default_num_workers()
        )

    training_args_dict = dict(
        output_dir=output_dir,
        max_steps=exp.training.max_steps,
        per_device_train_batch_size=exp.training.batch_size,
        gradient_accumulation_steps=exp.training.grad_accum,
        learning_rate=exp.training.lr,
        weight_decay=exp.training.weight_decay,
        warmup_steps=exp.training.warmup_ratio,
        max_grad_norm=exp.training.max_grad_norm,
        bf16=(exp.training.dtype == "bfloat16"),
        fp16=(exp.training.dtype == "float16"),
        logging_steps=exp.training.log_every,
        save_steps=exp.training.save_every,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        push_to_hub=False,  # Use our manual callback for non-bloated checkpoints
        report_to=exp.logging.report_to,
        torch_compile=exp.training.compile,
        dataloader_num_workers=workers,
    )

    # Initialize W&B if requested
    if "wandb" in exp.logging.report_to:
        import wandb

        wandb.init(
            project=exp.logging.wandb_project,
            entity=exp.logging.wandb_entity,
            name=exp.name,
            config=exp.training.__dict__,
        )
        # Add model logging if requested
        if exp.logging.log_model != "none":
            os.environ["WANDB_LOG_MODEL"] = exp.logging.log_model

    if eval_ds is not None:
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["eval_steps"] = exp.training.eval_every
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "loss"
    else:
        training_args_dict["eval_strategy"] = "no"
        training_args_dict["load_best_model_at_end"] = False

    if exp.training.hf_args:
        for k, v in exp.training.hf_args.items():
            training_args_dict[k] = v

    if args.smoke:
        training_args_dict["report_to"] = ["none"]

    training_args = TrainingArguments(**training_args_dict)

    # 7. Trainer
    callbacks = [HubCheckpointCallback(exp)]
    if not args.smoke:
        callbacks.append(ProfilingCallback(seq_len=exp.training.seq_len))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    try:
        hf_trainer = HFTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        hf_trainer = HFTrainer(tokenizer=tokenizer, **trainer_kwargs)

    # 8. Resume Logic
    resume_from = None
    if args.resume and Path(args.resume).exists():
        if (Path(args.resume) / "trainer_state.json").exists():
            resume_from = args.resume
            print(f"\nResuming Trainer state from: {resume_from}")
        else:
            print(f"\nLoaded weights from {args.resume}, starting new Trainer state.")
    elif args.resume:
        print(f"\nWARNING: Checkpoint not found: {args.resume}")
    else:
        # 1. Try to pull from Hub if enabled
        if exp.hub.use_hub_checkpoints:
            print(f"\n[HubCheckpoint] Checking Hub for latest checkpoint...")
            manager = HubCheckpointManager(exp.hub, exp)
            downloaded_ckpt = manager.pull_latest(output_dir)
            if downloaded_ckpt:
                resume_from = str(downloaded_ckpt)
                print(
                    f"[HubCheckpoint] Found remote checkpoint, resuming from: {resume_from}"
                )

        # 2. Local fallback discovery
        if not resume_from:
            ckpt_dir = Path(output_dir)
            if ckpt_dir.exists():
                ckpts = sorted(
                    ckpt_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]),
                )
                if ckpts:
                    resume_from = str(ckpts[-1])
                    print(f"\nAuto-resuming from local: {resume_from}")

    # 9. Train
    print(f"\nStarting Phase {args.phase} training...")
    hf_trainer.train(resume_from_checkpoint=resume_from)

    # 10. Save final model
    final_path = f"{output_dir}/final"
    print(f"\nSaving final model to: {final_path}")
    hf_trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # 10.1 Manual push weights (clean)
    if exp.hub.repo_id and not args.smoke:
        print(f"\n[Hub] Pushing final model weights to {exp.hub.repo_id}...")
        try:
            model.push_to_hub(
                exp.hub.repo_id,
                private=exp.hub.private,
                commit_message=f"Final weights Phase {args.phase}",
            )
            tokenizer.push_to_hub(exp.hub.repo_id, private=exp.hub.private)
            print("[Hub] Push complete.")
        except Exception as e:
            print(f"[Hub] Push failed: {e}")

    print(f"\nPhase {args.phase} complete!")


if __name__ == "__main__":
    main()
