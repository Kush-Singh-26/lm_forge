"""
experiments/hf_native_example/train.py

Example of using the HF Trainer + ProfilingCallback + HubCheckpointCallback.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer as HFTrainer,
    DataCollatorForLanguageModeling,
)

from engine import load_experiment_config, HFCausalLM, prepare_dataset, SyntheticDataset
from engine.config.hf_config import LMForgeConfig
from engine.utils import ProfilingCallback, HubCheckpointCallback, HubCheckpointManager


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg_path = Path(__file__).parent / "config.yaml"
    exp = load_experiment_config(cfg_path)

    if args.steps:
        exp.training.max_steps = args.steps
    if args.cpu:
        exp.training.dtype = "float32"

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model
    HFCausalLM.register()
    hf_cfg = LMForgeConfig.from_model_config(exp.model)
    model = HFCausalLM(hf_cfg)

    # 3. Data
    train_ds = SyntheticDataset(1000, exp.training.seq_len, exp.model.vocab_size)

    # 4. Training Arguments
    output_dir = f"checkpoints/{exp.name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=exp.training.max_steps,
        per_device_train_batch_size=exp.training.batch_size,
        gradient_accumulation_steps=exp.training.grad_accum,
        learning_rate=exp.training.lr,
        bf16=(exp.training.dtype == "bfloat16"),
        logging_steps=exp.training.log_every,
        save_steps=exp.training.save_every,
        report_to=["none"],
        push_to_hub=False,
    )

    # 5. Trainer
    callbacks = [ProfilingCallback(), HubCheckpointCallback(exp)]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    hf_trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=tokenizer,
    )

    # 6. Resume
    resume_from = None
    if exp.hub.use_hub_checkpoints:
        manager = HubCheckpointManager(exp.hub, exp)
        downloaded = manager.pull_latest(output_dir)
        if downloaded:
            resume_from = str(downloaded)

    # 7. Train
    hf_trainer.train(resume_from_checkpoint=resume_from)


if __name__ == "__main__":
    main()
