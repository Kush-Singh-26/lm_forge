"""
engine/data/collators.py

Collators for Causal LM and Masked LM training.

CLMCollator:
    For causal language modelling.  Pads input_ids and labels to the longest
    sequence in the batch, sets pad positions in labels to -100 (ignored by
    cross-entropy), and builds the attention_mask.

    When used with PackedDataset (no padding) it is a near-zero-cost passthrough.

MLMCollator:
    For masked language modelling (BERT-style encoder training).
    Randomly masks 15% of input tokens:
      80% → [MASK]
      10% → random token
      10% → unchanged
    Labels are -100 everywhere except masked positions.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch


class CLMCollator:
    """
    Collate a batch of CLM samples into padded tensors.

    Args:
        pad_id      : Token id to use for padding input_ids.
        max_seq_len : Truncate sequences longer than this.  None = no truncation.

    Returns batches with keys: input_ids, labels, attention_mask.
    """

    def __init__(self, pad_id: int = 0, max_seq_len: Optional[int] = None) -> None:
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [b["input_ids"] for b in batch]
        labels_list = [b.get("labels", b["input_ids"]) for b in batch]
        attn_mask_list = [b.get("attention_mask") for b in batch]

        # Truncate and convert to tensors if they aren't already
        def to_tensor(x, dtype=torch.long):
            if isinstance(x, torch.Tensor):
                return x[: self.max_seq_len] if self.max_seq_len else x
            t = torch.tensor(x, dtype=dtype)
            return t[: self.max_seq_len] if self.max_seq_len else t

        input_ids_tensors = [to_tensor(x) for x in input_ids_list]
        labels_tensors = [to_tensor(x) for x in labels_list]

        # Use pad_sequence for efficient padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_tensors, batch_first=True, padding_value=self.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_tensors, batch_first=True, padding_value=-100
        )

        B, max_len = input_ids.shape
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)

        for i, mask in enumerate(attn_mask_list):
            n = len(input_ids_tensors[i])
            if mask is not None:
                m = to_tensor(mask)
                attn_mask[i, : len(m)] = m
            else:
                attn_mask[i, :n] = 1

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


class MLMCollator:
    """
    Collate and mask tokens for masked language modelling (BERT-style).

    Args:
        vocab_size    : Total vocabulary size (for random token replacement).
        mask_token_id : The [MASK] token id.
        pad_id        : Padding token id.
        mask_prob     : Fraction of tokens to mask (default 0.15).
        max_seq_len   : Optional truncation length.

    Special tokens (ids in special_token_ids) are never masked.
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        pad_id: int = 0,
        mask_prob: float = 0.15,
        special_token_ids: Optional[List[int]] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_id = pad_id
        self.mask_prob = mask_prob
        self.special_ids = set(special_token_ids or [pad_id, mask_token_id])
        self.max_seq_len = max_seq_len
        self._min_random_id = max(self.special_ids) + 1 if self.special_ids else 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [b["input_ids"] for b in batch]

        def to_tensor(x):
            t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
            return t[: self.max_seq_len] if self.max_seq_len else t

        input_ids_tensors = [to_tensor(x) for x in input_ids_list]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_tensors, batch_first=True, padding_value=self.pad_id
        )

        B, max_len = input_ids.shape
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, t in enumerate(input_ids_tensors):
            attn_mask[i, : len(t)] = 1

        # ── Masking ───────────────────────────────────────────────────────
        labels = input_ids.clone()

        # Use random matrix instead of multiple bernoulli calls
        rand = torch.rand(input_ids.shape)
        masked_indices = (rand < self.mask_prob) & (attn_mask == 1)

        # Never mask special tokens
        special_token_mask = torch.isin(
            input_ids, torch.tensor(list(self.special_ids), device=input_ids.device)
        )
        masked_indices &= ~special_token_mask

        # Labels: -100 everywhere except masked positions
        labels[~masked_indices] = -100

        # Sub-masking: Of masked tokens, 80% [MASK], 10% random, 10% unchanged
        # Generate another random matrix for those masked positions
        rand_mask = torch.rand(input_ids.shape)

        # 80% → [MASK]
        replace_mask = masked_indices & (rand_mask < 0.8)
        input_ids[replace_mask] = self.mask_token_id

        # 10% → random token (0.8 <= rand_mask < 0.9)
        random_indices = masked_indices & (rand_mask >= 0.8) & (rand_mask < 0.9)
        num_random = random_indices.sum().item()
        if num_random > 0:
            random_tokens = torch.randint(
                self._min_random_id,
                self.vocab_size,
                (num_random,),
                dtype=torch.long,
                device=input_ids.device,
            )
            input_ids[random_indices] = random_tokens

        # 10% → unchanged (already is)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}
