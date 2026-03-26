"""
scripts/verify_hf_native.py

Quick smoke test for HF-native features:
1. Load CausalLM from HF config
2. Tokenize and pack a small dataset
3. Run a forward pass and check for loss + KV cache
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer as HFTrainer

from engine.models.hf_model import HFCausalLM
from engine.config.hf_config import LMForgeConfig
from engine.data.hf_utils import prepare_dataset


def main():
    print("--- 1. Initializing HF-native Model ---")
    # Minimal config for fast test
    hf_cfg = LMForgeConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        max_seq_len=64,
        attention__num_heads=4,
        attention__num_kv_heads=2,
    )
    HFCausalLM.register()
    model = HFCausalLM(hf_cfg)
    print(f"Model initialized: {model.num_parameters():,} parameters.")

    print("\n--- 2. Preparing HF-native Dataset ---")
    # Dummy data
    raw_data = {
        "text": ["Hello world! This is a test.", "Another sequence for training."]
    }
    ds = Dataset.from_dict(raw_data)

    # Use standard GPT-2 tokenizer for this test
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Update model vocab_size to match tokenizer
    hf_cfg.vocab_size = tokenizer.vocab_size
    model = HFCausalLM(hf_cfg)
    print(f"Model initialized: {model.num_parameters():,} parameters.")

    packed_ds = prepare_dataset(ds, tokenizer, seq_len=32)

    print(f"Dataset prepared: {len(packed_ds)} chunks.")
    print(f"First chunk keys: {packed_ds[0].keys()}")

    print("\n--- 3. Forward Pass & KV Cache Check ---")
    batch = torch.tensor([packed_ds[0]["input_ids"]])
    labels = torch.tensor([packed_ds[0]["labels"]])

    outputs = model(input_ids=batch, labels=labels, use_cache=True)

    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(
        f"KV Cache present (layers): {len(outputs.past_key_values) if outputs.past_key_values else 'None'}"
    )

    assert outputs.loss is not None
    assert outputs.past_key_values is not None
    assert len(outputs.past_key_values) == hf_cfg.num_layers

    print("\n--- HF-native Smoke Test PASSED! ---")


if __name__ == "__main__":
    main()
