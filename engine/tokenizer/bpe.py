"""
engine/tokenizer/bpe.py

Byte-level BPE Tokenizer — implemented from scratch.

Byte-level BPE (used by GPT-2, RoBERTa, LLaMA) works in two stages:

  Training (build_vocab):
    1. Represent every character as its UTF-8 byte value (0-255).
    2. Count all adjacent byte pairs across the corpus.
    3. Repeatedly merge the most frequent pair into a new token.
    4. Repeat until vocab_size is reached.

  Encoding (encode):
    1. Convert text to bytes.
    2. Greedily apply learned merges in the order they were discovered.

  Decoding (decode):
    1. Map token ids back to byte sequences.
    2. Decode bytes as UTF-8 (with error replacement).

Key advantages over character-level or word-level:
  • Handles any Unicode — no <UNK> tokens for rare characters.
  • Vocabulary size is a tunable hyperparameter.
  • Subword units capture morphology better than characters.

This is a clean from-scratch implementation suitable for small-to-medium
corpora (up to ~1B characters).  For production use on large corpora,
consider using the Rust-backed tokenizers library with a custom trainer.

Usage::

    from engine.tokenizer import BPETokenizer

    # Train from scratch
    tok = BPETokenizer()
    tok.train(["Hello world!", "Hello there."], vocab_size=300)
    tok.save("tokenizer.json")

    # Encode / decode
    ids = tok.encode("Hello world!")
    txt = tok.decode(ids)

    # Load saved tokenizer
    tok2 = BPETokenizer.load("tokenizer.json")
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Special tokens are inserted at positions 0..len(special_tokens)-1.
    Byte tokens occupy positions len(special_tokens)..len(special_tokens)+255.
    Merge tokens follow in the order merges were applied.

    Attributes:
        vocab           : {token_string: token_id}
        vocab_inv       : {token_id: token_string}
        merges          : Ordered list of (a, b) merge pairs.
        special_tokens  : {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, ...}
    """

    # Default special tokens — indices 0-3
    DEFAULT_SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.special_tokens: Dict[str, int] = {}
        self.vocab: Dict[str, int] = {}
        self.vocab_inv: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._merge_rank: Dict[Tuple[str, str], int] = {}

        # Add special tokens
        specials = (
            special_tokens if special_tokens is not None else self.DEFAULT_SPECIAL
        )
        for i, tok in enumerate(specials):
            self.special_tokens[tok] = i
            self.vocab[tok] = i
            self.vocab_inv[i] = tok

        # Base byte vocab (256 tokens, one per byte value)
        offset = len(self.special_tokens)
        for b in range(256):
            tok = self._byte_to_str(b)
            idx = offset + b
            self.vocab[tok] = idx
            self.vocab_inv[idx] = tok

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _byte_to_str(b: int) -> str:
        """Represent a byte as a printable string token."""
        # Use the GPT-2 byte→unicode mapping for clean display
        if (b >= ord("!") and b <= ord("~")) or (b >= 161 and b <= 172) or (b >= 174):
            return chr(b)
        return chr(b + 256)

    def _text_to_bytes(self, text: str) -> List[str]:
        """Convert text to a list of byte-token strings."""
        return [self._byte_to_str(b) for b in text.encode("utf-8")]

    def _get_stats(self, corpus: List[List[str]]) -> Counter:
        """Count adjacent pair frequencies across all tokenised sequences."""
        stats: Counter = Counter()
        for seq in corpus:
            for a, b in zip(seq[:-1], seq[1:]):
                stats[(a, b)] += 1
        return stats

    def _apply_merge(
        self,
        corpus: List[List[str]],
        pair: Tuple[str, str],
        merged: str,
    ) -> List[List[str]]:
        """Replace all occurrences of pair (a, b) with merged token."""
        a, b = pair
        new_corpus = []
        for seq in corpus:
            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_corpus.append(new_seq)
        return new_corpus

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> None:
        """
        Train the tokenizer on a corpus.

        Args:
            texts         : Iterable of raw text strings (documents).
            vocab_size    : Target vocabulary size (including special + byte tokens).
            min_frequency : Pairs seen fewer times than this are not merged.
            verbose       : Print progress every 100 merges.

        The number of merges performed = vocab_size - len(special) - 256.
        """
        base_size = len(self.special_tokens) + 256
        n_merges = max(0, vocab_size - base_size)

        if n_merges == 0:
            if verbose:
                print(
                    f"[BPETokenizer] vocab_size={vocab_size} ≤ base size {base_size}. "
                    "No merges needed."
                )
            return

        if verbose:
            print(
                f"[BPETokenizer] Training {n_merges} merges "
                f"(target vocab_size={vocab_size})..."
            )

        # Build initial corpus
        corpus = [self._text_to_bytes(t) for t in texts if t.strip()]

        for merge_idx in range(n_merges):
            stats = self._get_stats(corpus)
            if not stats:
                break

            # Filter low-frequency pairs
            best_pair, freq = stats.most_common(1)[0]
            if freq < min_frequency:
                if verbose:
                    print(
                        f"  Stopping at merge {merge_idx}: "
                        f"best pair freq={freq} < min_frequency={min_frequency}"
                    )
                break

            # Register the new token
            merged = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[merged] = new_id
            self.vocab_inv[new_id] = merged
            self.merges.append(best_pair)
            self._merge_rank[best_pair] = merge_idx

            corpus = self._apply_merge(corpus, best_pair, merged)

            if verbose and (merge_idx + 1) % 100 == 0:
                print(
                    f"  Merge {merge_idx + 1}/{n_merges}  "
                    f"pair={best_pair!r}  freq={freq}  "
                    f"vocab={len(self.vocab)}"
                )

        if verbose:
            print(f"[BPETokenizer] Done. Vocab size: {len(self.vocab)}")

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to a list of token ids.

        Args:
            text    : Input string.
            add_bos : Prepend BOS token (if <BOS> is a special token).
            add_eos : Append EOS token (if <EOS> is a special token).

        Note:
            This implementation has O(n^2) complexity per sequence. It is
            suitable for educational use and small-to-medium corpora. For
            production use at scale, use HFBPETokenizer or the Rust-backed
            tokenizers library instead.
        """
        tokens = self._text_to_bytes(text)

        # Apply merges in training order (greedy BPE)
        while True:
            # Find the highest-priority (lowest rank) merge applicable
            best_rank = len(self.merges)
            best_idx = -1
            for i, (a, b) in enumerate(zip(tokens[:-1], tokens[1:])):
                rank = self._merge_rank.get((a, b), len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            a, b = tokens[best_idx], tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [a + b] + tokens[best_idx + 2 :]

        ids = [self.vocab.get(t, self.special_tokens.get("<UNK>", 1)) for t in tokens]

        if add_bos and "<BOS>" in self.special_tokens:
            ids = [self.special_tokens["<BOS>"]] + ids
        if add_eos and "<EOS>" in self.special_tokens:
            ids = ids + [self.special_tokens["<EOS>"]]

        return ids

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Encode a list of strings."""
        return [self.encode(t, **kwargs) for t in texts]

    # ── Decoding ──────────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token ids back to a string.

        Args:
            ids           : List of token ids.
            skip_special  : If True, special tokens are omitted from output.
        """
        special_ids = set(self.special_tokens.values())
        byte_list: List[int] = []

        for tok_id in ids:
            if tok_id in special_ids:
                if not skip_special:
                    # Append the special token string as UTF-8 bytes
                    tok_str = self.vocab_inv.get(tok_id, "")
                    byte_list.extend(tok_str.encode("utf-8"))
                continue
            tok_str = self.vocab_inv.get(tok_id, "")
            for ch in tok_str:
                code = ord(ch)
                byte_list.append(code if code < 256 else code - 256)

        return bytes(byte_list).decode("utf-8", errors="replace")

    # ── Vocabulary properties ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens.get("<PAD>", 0)

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens.get("<BOS>", 2)

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens.get("<EOS>", 3)

    @property
    def mask_token_id(self) -> int:
        return self.special_tokens.get("<MASK>", 4)

    @property
    def unk_token_id(self) -> int:
        return self.special_tokens.get("<UNK>", 1)

    # HF-compatible interface (for use with DataCollators, etc.)
    def __call__(self, text: str | List[str], **kwargs) -> dict:
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": self.encode_batch(text)}

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save tokenizer to a JSON file."""
        data = {
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
        }
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a JSON file saved by .save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(special_tokens=list(data["special_tokens"].keys()))
        # Override special token ids in case they differ from defaults
        for name, idx in data["special_tokens"].items():
            tok.special_tokens[name] = idx
            tok.vocab[name] = idx
            tok.vocab_inv[idx] = name

        # Replay merges
        for merge_idx, (a, b) in enumerate(data["merges"]):
            merged = a + b
            if merged not in tok.vocab:
                new_id = len(tok.vocab)
                tok.vocab[merged] = new_id
                tok.vocab_inv[new_id] = merged
            tok.merges.append((a, b))
            tok._merge_rank[(a, b)] = merge_idx

        return tok
