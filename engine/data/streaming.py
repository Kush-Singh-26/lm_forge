"""
engine/data/streaming.py

Streaming data pipeline for pretraining on massive HF datasets.
Handles on-the-fly tokenization and vectorized packing (concatenate-and-chunk).
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from datasets import load_dataset


# ─── Streaming Dataset Builders ─────────────────────────────────────────────


def build_streaming_dataset(
    dataset_name: str,
    tokenizer,
    seq_len: int = 1024,
    text_column: str = "text",
    config_name: Optional[str] = None,
    shuffle_buffer: int = 10000,
    split: str = "train",
    seed: int = 42,
):
    """
    Build a tokenized, packed streaming dataset from a HF dataset.
    Uses numpy for vectorized packing (much faster than Python loops).
    """
    print(f"[data] Streaming {dataset_name} (config={config_name}, split={split})...")

    ds = load_dataset(
        dataset_name,
        name=config_name,
        split=split,
        streaming=True,
    )

    if text_column != "text" and text_column in ds.column_names:
        ds = ds.rename_column(text_column, "text")

    if "text" not in ds.column_names:
        available = ds.column_names
        raise ValueError(f"No 'text' column found. Available: {available}.")

    cols_to_remove = [c for c in ds.column_names if c != "text"]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    def tokenize_fn(examples):
        try:
            return tokenizer(examples["text"], add_special_tokens=False)
        except TypeError:
            return tokenizer(examples["text"])

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    # Step 2: Vectorized Packing
    def group_fn(examples):
        # Concatenate tokens across the batch using itertools (fast)
        concatenated = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}

        # Get total length from first key
        first_key = list(examples.keys())[0]
        total_len = len(concatenated[first_key])

        if total_len < seq_len:
            return {k: [] for k in examples.keys()}

        num_full = (total_len // seq_len) * seq_len

        # Slice and reshape using NumPy for efficiency
        result = {}
        for k in concatenated.keys():
            arr = np.array(concatenated[k])[:num_full].reshape(-1, seq_len)
            result[k] = arr.tolist()

        return result

    packed = tokenized.map(
        group_fn,
        batched=True,
        batch_size=1000,
    )

    if shuffle_buffer > 0:
        packed = packed.shuffle(seed=seed, buffer_size=shuffle_buffer)

    print(
        f"[data] Streaming dataset ready (seq_len={seq_len}, shuffle={shuffle_buffer})"
    )
    return packed


def build_fineweb_edu(tokenizer, seq_len: int = 1024, shuffle_buffer: int = 10000):
    """Phase 1: fineweb-edu (general knowledge)."""
    return build_streaming_dataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column="text",
        config_name="default",
        shuffle_buffer=shuffle_buffer,
    )


def build_stack_v2(
    tokenizer,
    seq_len: int = 1024,
    languages: Optional[list[str]] = None,
    shuffle_buffer: int = 10000,
):
    """Phase 2: the-stack-v2 (code)."""
    if languages is None:
        return build_streaming_dataset(
            dataset_name="bigcode/the-stack-v2",
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_column="content",
            config_name="default",
            shuffle_buffer=shuffle_buffer,
        )

    from datasets import interleave_datasets

    streams = []
    for lang in languages:
        lang_config = "C++" if lang == "C__" else lang
        try:
            ds = build_streaming_dataset(
                dataset_name="bigcode/the-stack-v2",
                tokenizer=tokenizer,
                seq_len=seq_len,
                text_column="content",
                config_name=lang_config,
                shuffle_buffer=shuffle_buffer // len(languages),
            )
            streams.append(ds)
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to load language config '{lang_config}': {e}. Skipping."
            )

    combined = interleave_datasets(streams, seed=42, stopping_strategy="all_exhausted")
    return combined


# ─── Synthetic Fallback (smoke tests) ──────────────────────────────────────


class SyntheticDataset:
    def __init__(self, n_samples: int, seq_len: int, vocab_size: int = 50257):
        import torch

        gen = torch.Generator().manual_seed(42)
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len), generator=gen)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch

        ids = self.data[idx].clone()
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }
