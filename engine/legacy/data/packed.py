"""
engine/data/packed.py

PackedDataset — the correct way to train a causal LM.

Problem with naive batching:
  Most sequences are shorter than seq_len, so you pad them.  Padding tokens
  contribute zero signal but still cost compute in attention.  At seq_len=1024
  with typical web text, naive batching can be 40-60% wasted compute.

Solution — sequence packing (a.k.a. "concatenate and chunk"):
  Concatenate all token sequences end-to-end with an EOS separator, then
  slice into fixed-length chunks.  Every single position in every batch
  is a real token.  Zero padding, 100% utilisation.

  [seq1 tokens... EOS][seq2 tokens... EOS][seq3 tokens...] → chunks of seq_len

  Each chunk is treated as an independent training example.  The model
  never sees across chunk boundaries (the causal mask handles this naturally
  since it only looks backward within the chunk).

Usage::

    from engine.data import PackedDataset, build_dataloader

    # From a list of pre-tokenised sequences (List[List[int]])
    ds = PackedDataset(token_sequences, seq_len=1024, eos_id=2)
    loader = build_dataloader(ds, batch_size=8)

    # From a HuggingFace dataset (streaming supported)
    from datasets import load_dataset
    hf_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ds = PackedDataset.from_hf(hf_ds, tokenizer, seq_len=1024)

    # The dataset also records which chunk belongs to which source document
    # via doc_ids — useful for per-document evaluation.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class PackedDataset(Dataset):
    """
    Offline packing — loads all token sequences into RAM, packs, and indexes.

    Use for datasets that fit in memory (≲ a few GB of tokens).
    For larger datasets use StreamingPackedDataset.

    Args:
        sequences : List of token-id lists.  Each list is one document.
        seq_len   : Chunk length (= model max_seq_len during training).
        eos_id    : Token inserted between documents.  -1 = no separator.
        stride    : Overlap between consecutive chunks.  0 = no overlap
                    (recommended for training).  > 0 gives more samples
                    at the cost of redundancy.
    """

    def __init__(
        self,
        sequences: List[List[int]],
        seq_len: int,
        eos_id: int = -1,
        stride: int = 0,
    ) -> None:
        self.seq_len = seq_len
        self._chunks: List[torch.Tensor] = []

        # Concatenate all sequences with EOS separators
        flat: List[int] = []
        for seq in sequences:
            flat.extend(seq)
            if eos_id >= 0:
                flat.append(eos_id)

        # Slice into non-overlapping (or strided) chunks of seq_len+1
        # The +1 gives us the labels (shifted by 1) for free
        step = seq_len - stride if stride > 0 else seq_len
        full_tensor = torch.tensor(flat, dtype=torch.long)

        for start in range(0, len(full_tensor) - seq_len, step):
            chunk = full_tensor[start : start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self._chunks.append(chunk.clone())

        if not self._chunks:
            raise ValueError(
                f"No complete chunks of length {seq_len} could be formed from "
                f"{len(flat)} total tokens.  Either the dataset is too small or "
                f"seq_len is too large."
            )

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self._chunks[idx]
        return {
            "input_ids": chunk[:-1].clone(),   # tokens 0..N-1
            "labels":    chunk[1:].clone(),    # tokens 1..N (next-token targets)
        }

    @property
    def num_tokens(self) -> int:
        """Total training tokens across all chunks."""
        return len(self._chunks) * self.seq_len

    # ── Factory methods ───────────────────────────────────────────────────

    @classmethod
    def from_hf(
        cls,
        hf_dataset,
        tokenizer,
        seq_len: int,
        text_column: str = "text",
        eos_id: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> "PackedDataset":
        """
        Build a PackedDataset from a HuggingFace dataset.

        Args:
            hf_dataset   : A HF Dataset or IterableDataset.
            tokenizer    : Any tokenizer with __call__(text) → input_ids.
            seq_len      : Chunk length.
            text_column  : Column name containing raw text.
            eos_id       : EOS token id.  If None, uses tokenizer.eos_token_id.
            max_samples  : Cap number of documents (useful for quick tests).

        Example::

            from datasets import load_dataset
            from transformers import AutoTokenizer

            ds  = load_dataset("roneneldan/TinyStories", split="train")
            tok = AutoTokenizer.from_pretrained("gpt2")
            packed = PackedDataset.from_hf(ds, tok, seq_len=1024)
        """
        _eos_id = eos_id if eos_id is not None else getattr(tokenizer, "eos_token_id", -1)
        sequences = []
        for i, row in enumerate(hf_dataset):
            if max_samples is not None and i >= max_samples:
                break
            text = row[text_column]
            ids  = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                sequences.append(ids)

        return cls(sequences, seq_len=seq_len, eos_id=_eos_id)

    @classmethod
    def from_text_files(
        cls,
        paths: List[str],
        tokenizer,
        seq_len: int,
        eos_id: Optional[int] = None,
    ) -> "PackedDataset":
        """
        Build from a list of plain text file paths.

        Each file becomes one document (EOS inserted between files).
        """
        import pathlib
        sequences = []
        _eos_id = eos_id if eos_id is not None else getattr(tokenizer, "eos_token_id", -1)
        for path in paths:
            text = pathlib.Path(path).read_text(encoding="utf-8")
            ids  = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                sequences.append(ids)
        return cls(sequences, seq_len=seq_len, eos_id=_eos_id)


class StreamingPackedDataset(IterableDataset):
    """
    Online packing — streams documents and packs on-the-fly.

    Suitable for datasets that don't fit in RAM (Common Crawl, The Pile, etc.).
    Compatible with HF streaming datasets.

    Args:
        hf_iterable  : HF IterableDataset or any iterable of {"text": str} dicts.
        tokenizer    : Tokenizer with __call__(text) → input_ids.
        seq_len      : Output chunk length.
        text_column  : Column name for raw text.
        eos_id       : Token inserted between documents.
        buffer_size  : How many packed chunks to buffer before yielding.
                       Larger buffer = better shuffling at the cost of RAM.
        shuffle      : Shuffle the buffer before yielding.
    """

    def __init__(
        self,
        hf_iterable,
        tokenizer,
        seq_len: int,
        text_column: str = "text",
        eos_id: Optional[int] = None,
        buffer_size: int = 1000,
        shuffle: bool = True,
    ) -> None:
        self.hf_iterable   = hf_iterable
        self.tokenizer     = tokenizer
        self.seq_len       = seq_len
        self.text_column   = text_column
        self.eos_id        = eos_id if eos_id is not None else getattr(tokenizer, "eos_token_id", -1)
        self.buffer_size   = buffer_size
        self.shuffle       = shuffle

    def __iter__(self) -> Iterator[dict]:
        buffer: List[dict] = []
        carry: List[int] = []   # leftover tokens from the previous document

        for row in self.hf_iterable:
            text = row.get(self.text_column, "")
            if not text:
                continue
            ids  = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            carry.extend(ids)
            if self.eos_id >= 0:
                carry.append(self.eos_id)

            # Slice complete chunks from carry
            while len(carry) >= self.seq_len + 1:
                chunk = torch.tensor(carry[: self.seq_len + 1], dtype=torch.long)
                carry = carry[self.seq_len:]
                buffer.append({
                    "input_ids": chunk[:-1].clone(),
                    "labels":    chunk[1:].clone(),
                })
                if len(buffer) >= self.buffer_size:
                    if self.shuffle:
                        random.shuffle(buffer)
                    yield from buffer
                    buffer = []

        # Yield remaining buffer
        if self.shuffle:
            random.shuffle(buffer)
        yield from buffer
