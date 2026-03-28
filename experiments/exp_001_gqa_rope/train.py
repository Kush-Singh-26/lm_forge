"""
experiments/exp_001_gqa_rope/train.py

Example using custom config + HF Trainer.
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

from engine import load_experiment_config, HFCausalLM, prepare_dataset
from engine.config.hf_config import LMForgeConfig
from engine.utils import ProfilingCallback, HubCheckpointCallback, HubCheckpointManager
from data_streaming import SyntheticDataset


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    exp = load_experiment_config(cfg_path)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_cfg = LMForgeConfig.from_model_config(exp.model)
    model = HFCausalLM(hf_cfg)

    train_ds = SyntheticDataset(2000, exp.training.seq_len, exp.model.vocab_size)

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

    resume_from = None
    if exp.hub.use_hub_checkpoints:
        manager = HubCheckpointManager(exp.hub, exp)
        downloaded = manager.pull_latest(output_dir)
        if downloaded:
            resume_from = str(downloaded)

    hf_trainer.train(resume_from_checkpoint=resume_from)


if __name__ == "__main__":
    main()
