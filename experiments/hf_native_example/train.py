"""
experiments/hf_native_example/train.py

Simplified training entry point demonstrating HF-native usage.
Usage:
    python experiments/hf_native_example/train.py --steps 100 --cpu
"""

from __future__ import annotations
import argparse
import sys
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer as HFTrainer

from engine import load_experiment_config, HFCausalLM, prepare_dataset
from engine.config.hf_config import LMForgeConfig
from engine.utils import ProfilingCallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    # 1. Load Config
    cfg_path = Path(__file__).parent / "config.yaml"
    exp = load_experiment_config(cfg_path)

    if args.steps:
        exp.training.max_steps = args.steps
    if args.cpu:
        exp.training.backend = "cpu"
        exp.training.dtype = "float32"

    print(f"--- Experiment: {exp.name} ---")

    # 2. Prepare Model
    HFCausalLM.register()
    hf_cfg = LMForgeConfig.from_model_config(exp.model)
    hf_cfg.use_cache = False  # Disable cache during training
    model = HFCausalLM(hf_cfg)
    model.lm.model.enable_gradient_checkpointing()
    print(f"Model parameters: {model.num_parameters():,}")

    # 3. Prepare Data
    print("Loading TinyStories dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_train = load_dataset(
        "roneneldan/TinyStories", split="train[:1%]"
    )  # small slice for demo
    raw_eval = load_dataset("roneneldan/TinyStories", split="validation[:1%]")

    train_ds = prepare_dataset(raw_train, tokenizer, seq_len=exp.training.seq_len)
    eval_ds = prepare_dataset(raw_eval, tokenizer, seq_len=exp.training.seq_len)

    # 4. HF Training Arguments
    training_args_dict = dict(
        output_dir=f"checkpoints/{exp.name}",
        max_steps=exp.training.max_steps,
        per_device_train_batch_size=exp.training.batch_size,
        gradient_accumulation_steps=exp.training.grad_accum,
        learning_rate=exp.training.lr,
        warmup_ratio=exp.training.warmup_ratio,
        bf16=(exp.training.dtype == "bfloat16"),
        fp16=(exp.training.dtype == "float16"),
        push_to_hub=False,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
    )
    # Merge with YAML hf_args
    training_args_dict.update(exp.training.hf_args)

    training_args = TrainingArguments(**training_args_dict)

    # 5. Train
    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[ProfilingCallback()],
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    final_path = f"checkpoints/{exp.name}/final"
    trainer.save_model(final_path)

    # 6. Test Generation
    print("\n--- Testing Generation ---")
    from transformers import pipeline

    # Use the saved model and tokenizer
    gen = pipeline(
        "text-generation",
        model=final_path,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    prompt = "Once upon a time,"
    print(f"Prompt: {prompt}")
    output = gen(prompt, max_new_tokens=30, do_sample=True, temperature=0.7)
    print(f"Output: {output[0]['generated_text']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
