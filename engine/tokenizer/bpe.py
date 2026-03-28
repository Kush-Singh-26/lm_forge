"""
engine/tokenizer/bpe.py

High-performance Byte-level BPE Tokenizer — backed by the Rust 'tokenizers' library.
Replaces the legacy O(N^2) Python implementation with production-grade speed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


class BPETokenizer:
    """
    Rust-backed Byte-level BPE tokenizer.
    Compatible with GPT-2, LLaMA, and other modern LLM tokenization schemes.
    """

    DEFAULT_SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.special_tokens_list = (
            special_tokens if special_tokens is not None else self.DEFAULT_SPECIAL
        )

        # Initialize the Rust tokenizer with a BPE model
        self._tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

        # Byte-level pre-tokenization (maps bytes to unique unicode chars)
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> None:
        """
        Train the tokenizer on a corpus using the Rust trainer.
        """
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens_list,
            show_progress=verbose,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # tokenizers.train expects an iterator of strings
        self._tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to a list of token ids.
        """
        if not text:
            return []

        output = self._tokenizer.encode(text)
        ids = output.ids

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Encode a list of strings."""
        outputs = self._tokenizer.encode_batch(texts)
        return [output.ids for output in outputs]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token ids back to a string.
        """
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def _get_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token) or 0

    @property
    def pad_token_id(self) -> int:
        return self._get_id("<PAD>")

    @property
    def bos_token_id(self) -> int:
        return self._get_id("<BOS>")

    @property
    def eos_token_id(self) -> int:
        return self._get_id("<EOS>")

    @property
    def mask_token_id(self) -> int:
        return self._get_id("<MASK>")

    @property
    def unk_token_id(self) -> int:
        return self._get_id("<UNK>")

    def __call__(self, text: str | List[str], **kwargs) -> dict:
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": self.encode_batch(text)}

    def save(self, path: str | Path) -> None:
        """Save tokenizer to a JSON file."""
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a JSON file."""
        instance = cls()
        instance._tokenizer = Tokenizer.from_file(str(path))
        return instance
