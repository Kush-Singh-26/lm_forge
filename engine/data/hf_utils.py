"""
engine/data/hf_utils.py

Utilities for preparing Hugging Face datasets for training.
Includes tokenization and native packing (concatenate and chunk).
"""

from __future__ import annotations
from typing import Optional
import itertools
import inspect

def prepare_dataset(
    dataset,
    tokenizer,
    seq_len: int,
    text_column: str = "text",
    num_proc: int = 4,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Tokenize and pack a HF dataset into fixed-length chunks.

    This uses the 'concatenate and chunk' strategy which is the standard
    way to train causal language models efficiently.
    """
    # Check if tokenizer accepts add_special_tokens
    sig = inspect.signature(tokenizer)
    supports_special_tokens = "add_special_tokens" in sig.parameters

    def tokenize_function(examples):
        texts = examples[text_column]
        if supports_special_tokens:
            return tokenizer(texts, add_special_tokens=False)
        return tokenizer(texts)

    # 1. Tokenize
    tokenized_ds = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # 2. Pack (group texts into seq_len chunks)
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it
        # but packing is usually done with dropping the last bit.
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len

        # Split into chunks of max_len.
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        # For CausalLM, labels are typically input_ids shifted.
        # HF Trainer handles shifting internally if 'labels' are provided.
        result["labels"] = result["input_ids"].copy()
        return result

    packed_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping texts in chunks of {seq_len}",
    )

    if shuffle:
        packed_ds = packed_ds.shuffle(seed=seed)

    return packed_ds
