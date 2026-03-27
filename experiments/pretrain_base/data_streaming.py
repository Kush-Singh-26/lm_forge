"""
experiments/pretrain_base/data_streaming.py

Streaming data pipeline for pretraining on massive HF datasets.
Handles on-the-fly tokenization and packing (concatenate-and-chunk).

Supports:
  - HuggingFaceFW/fineweb-edu  (text column: "text")
  - bigcode/the-stack-v2        (text column: "content")
  - Any HF IterableDataset with a text column

Usage:
    from data_streaming import build_streaming_dataset

    train_ds = build_streaming_dataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer=tokenizer,
        seq_len=1024,
    )
"""

from __future__ import annotations

import itertools
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer


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

    Returns an IterableDataset that yields {"input_ids": list, "labels": list}
    where each list has length seq_len.
    """
    print(f"[data] Streaming {dataset_name} (config={config_name}, split={split})...")

    ds = load_dataset(
        dataset_name,
        name=config_name,
        split=split,
        streaming=True,
    )

    # Rename content → text if needed (the-stack-v2 uses "content")
    if text_column != "text" and text_column in ds.column_names:
        ds = ds.rename_column(text_column, "text")

    # Ensure "text" column exists
    if "text" not in ds.column_names:
        available = ds.column_names
        raise ValueError(
            f"No 'text' column found. Available: {available}. "
            f"Set text_column= explicitly."
        )

    # Remove all non-text columns to keep things lean
    cols_to_remove = [c for c in ds.column_names if c != "text"]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    # Step 1: Tokenize (batched for speed)
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

    # Step 2: Pack into seq_len chunks (concatenate-and-chunk)
    def group_fn(examples):
        concatenated = list(itertools.chain(*examples["input_ids"]))
        total = (len(concatenated) // seq_len) * seq_len
        chunks = [concatenated[i : i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    packed = tokenized.map(
        group_fn,
        batched=True,
        batch_size=1000,
    )

    # Step 3: Shuffle (buffer-based for streaming)
    if shuffle_buffer > 0:
        packed = packed.shuffle(seed=seed, buffer_size=shuffle_buffer)

    print(f"[data] Streaming dataset ready (seq_len={seq_len})")
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
    """
    Phase 2: the-stack-v2 (code).

    If languages is None, streams the full default config (all 600+ languages).
    If languages is provided (e.g. ["Python", "JavaScript"]), concatenates
    streams from those specific language configs.
    """
    if languages is None:
        # Full dataset — all languages
        return build_streaming_dataset(
            dataset_name="bigcode/the-stack-v2",
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_column="content",
            config_name="default",
            shuffle_buffer=shuffle_buffer,
        )

    # Multiple language configs — interleave them
    from datasets import interleave_datasets

    streams = []
    for lang in languages:
        # Language config name "C__" should be "C++" for the-stack-v2
        lang_config = "C++" if lang == "C__" else lang
        print(f"[data]   Adding language: {lang_config}")
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

    # Interleave with uniform sampling, exhausting all streams
    combined = interleave_datasets(streams, seed=42, stopping_strategy="all_exhausted")
    print(f"[data] Combined {len(languages)} language streams")
    return combined


# ─── Synthetic Fallback (smoke tests) ──────────────────────────────────────


class SyntheticDataset:
    """Random-token dataset for CPU smoke testing. NOT IterableDataset."""

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
