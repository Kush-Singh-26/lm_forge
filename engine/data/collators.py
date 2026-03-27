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
        self.pad_id     = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [b["input_ids"] for b in batch]
        labels_list    = [b.get("labels", b["input_ids"]) for b in batch]

        # Truncate if needed
        if self.max_seq_len is not None:
            input_ids_list = [x[: self.max_seq_len] for x in input_ids_list]
            labels_list    = [x[: self.max_seq_len] for x in labels_list]

        max_len = max(len(x) for x in input_ids_list)

        input_ids    = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        labels       = torch.full((len(batch), max_len), -100,         dtype=torch.long)
        attn_mask    = torch.zeros((len(batch), max_len),               dtype=torch.long)

        for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
            n = len(ids)
            input_ids[i, :n] = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids)
            labels[i, :n]    = lbls if isinstance(lbls, torch.Tensor) else torch.tensor(lbls)
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
        self.vocab_size        = vocab_size
        self.mask_token_id     = mask_token_id
        self.pad_id            = pad_id
        self.mask_prob         = mask_prob
        self.special_ids       = set(special_token_ids or [pad_id, mask_token_id])
        self.max_seq_len       = max_seq_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [b["input_ids"] for b in batch]
        if self.max_seq_len is not None:
            input_ids_list = [x[: self.max_seq_len] for x in input_ids_list]

        max_len   = max(len(x) for x in input_ids_list)
        n_samples = len(batch)

        input_ids = torch.full((n_samples, max_len), self.pad_id, dtype=torch.long)
        attn_mask = torch.zeros((n_samples, max_len),              dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            n = len(ids)
            input_ids[i, :n] = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids)
            attn_mask[i, :n] = 1

        # ── Masking ───────────────────────────────────────────────────────
        labels     = input_ids.clone()
        prob_mat   = torch.full(input_ids.shape, self.mask_prob)

        # Never mask padding or special tokens
        for sid in self.special_ids:
            prob_mat[input_ids == sid] = 0.0
        prob_mat[attn_mask == 0] = 0.0

        masked_indices = torch.bernoulli(prob_mat).bool()

        # Labels: -100 everywhere except masked positions
        labels[~masked_indices] = -100

        # 80% → [MASK]
        replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[replace_mask] = self.mask_token_id

        # 10% → random token  (of the remaining 20%)
        random_indices = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~replace_mask
        )
        random_tokens = torch.randint(len(self.special_ids), self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[random_indices] = random_tokens[random_indices]

        # 10% → unchanged (already is — nothing to do)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}
