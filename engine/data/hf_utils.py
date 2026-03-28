"""
engine/data/hf_utils.py

Utilities for preparing Hugging Face datasets for training.
Includes tokenization and vectorized packing (concatenate and chunk) using NumPy.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def prepare_dataset(
    dataset,
    tokenizer,
    seq_len: int,
    text_column: str = "text",
    num_proc: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Tokenize and pack a HF dataset into fixed-length chunks.
    Uses NumPy for high-performance vectorized packing.
    """
    if num_proc is None:
        import os

        num_proc = os.cpu_count() or 1

    def tokenize_function(examples):
        texts = examples[text_column]
        try:
            return tokenizer(texts, add_special_tokens=False)
        except TypeError:
            return tokenizer(texts)

    # 1. Tokenize
    tokenized_ds = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # 2. Vectorized Packing
    def group_texts(examples):
        # Concatenate tokens across the batch using NumPy (much faster for large integer arrays)
        concatenated = {k: np.concatenate(examples[k]) for k in examples.keys()}

        # Calculate how many full sequences we can make
        first_key = list(examples.keys())[0]
        total_length = len(concatenated[first_key])

        if total_length < seq_len:
            return {k: [] for k in examples.keys()}

        num_full = (total_length // seq_len) * seq_len

        # Slice and reshape for O(1) view-based chunking
        result = {
            k: concatenated[k][:num_full].reshape(-1, seq_len).tolist()
            for k in concatenated.keys()
        }

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
