"""
engine/data/memmap.py

MemmapDataset — reads pre-tokenized uint16 binary files via numpy memmap.

Why memmap?
  • The entire dataset stays on disk. The OS pages in only the chunks
    the DataLoader actually reads — a 10GB .bin file uses ~zero RAM.
  • Random access is O(1): seeking to chunk i is a pointer arithmetic op.
  • No Python tokenization overhead in the training loop.
  • Consistent with nanoGPT's approach, which achieves ~57% MFU on A100.

Typical workflow:

    1. Run pretokenize.py once (minutes)
       → train.bin, val.bin, meta.json written to disk + pushed to Hub

    2. In each Colab session (seconds):
       data_dir = pull_tokenized("username/tinystories-gpt2-tokenized", "data/")

    3. In the training loop (zero overhead):
       train_ds = MemmapDataset(data_dir / "train.bin", seq_len=1024)
       loader   = build_dataloader(train_ds, batch_size=8)

The dataset yields {input_ids, labels} dicts where labels = input_ids shifted
by 1 (the standard CLM objective). No collation overhead since every item is
already the same length.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """
    Dataset that reads tokens from a pre-tokenized uint16 numpy memmap file.

    Every item is a (seq_len,) slice of the flat token array.
    Labels are the same slice shifted by 1 (next-token prediction).

    Args:
        path    : Path to a .bin file produced by pretokenize.py.
        seq_len : Sequence length for each training sample.
        stride  : Step between consecutive chunks.  Default = seq_len
                  (no overlap).  Set < seq_len for sliding window sampling.

    Example::

        ds = MemmapDataset("data/train.bin", seq_len=1024)
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4,
                            pin_memory=True)
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        stride: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        if not self.path.exists():
            raise FileNotFoundError(
                f"Token file not found: {self.path}\n"
                f"Run pretokenize.py first, or call pull_tokenized() to download."
            )

        # Memory-map the file. mode="r" = read-only, safe for multiprocessing.
        self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self._n_tokens = len(self._data)

        # Number of complete (seq_len + 1) chunks we can extract
        # +1 because we need one extra token for the label shift
        self._n_chunks = (self._n_tokens - seq_len) // self.stride

        if self._n_chunks == 0:
            raise ValueError(
                f"{self.path} has {self._n_tokens:,} tokens, "
                f"which is too few for seq_len={seq_len}. "
                f"Needs at least {seq_len + 1} tokens."
            )

    def __len__(self) -> int:
        return self._n_chunks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.stride
        # Slice seq_len+1 tokens, convert uint16→int64 for embedding lookup
        chunk = torch.from_numpy(
            self._data[start : start + self.seq_len + 1].astype(np.int64)
        )
        # Returns shifted input/labels (standard convention)
        return {
            "input_ids": chunk[:-1].clone(),
            "labels": chunk[1:].clone(),
        }

    @property
    def num_tokens(self) -> int:
        return self._n_chunks * self.seq_len

    @property
    def vocab_size_hint(self) -> int:
        """Maximum token id seen in this shard + 1. Useful for sanity-checking."""
        # Sample 100k tokens to estimate; don't scan the whole file
        sample = self._data[:: max(1, len(self._data) // 100_000)]
        return int(sample.max()) + 1

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_dir(
        cls,
        data_dir: str | Path,
        split: str = "train",
        seq_len: Optional[int] = None,
        **kwargs,
    ) -> "MemmapDataset":
        """
        Load from a directory containing train.bin / val.bin / meta.json.

        If seq_len is not specified, uses the recommended seq_len from meta.json.

        Args:
            data_dir : Directory produced by pretokenize.py.
            split    : "train" or "val".
            seq_len  : Sequence length.  Reads from meta.json if not given.
        """
        data_dir = Path(data_dir)
        bin_path = data_dir / f"{split}.bin"

        if seq_len is None:
            meta_path = data_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                seq_len = meta.get("seq_recommended", 1024)
                print(
                    f"[MemmapDataset] Using recommended seq_len={seq_len} from meta.json"
                )
            else:
                seq_len = 1024
                print(f"[MemmapDataset] No meta.json found, defaulting to seq_len=1024")

        return cls(bin_path, seq_len=seq_len, **kwargs)

    @classmethod
    def from_hub(
        cls,
        hub_repo: str,
        split: str = "train",
        seq_len: Optional[int] = None,
        local_dir: Optional[str | Path] = None,
        token_env: str = "HF_TOKEN",
        **kwargs,
    ) -> "MemmapDataset":
        """
        Pull a pre-tokenized dataset from HF Hub and return a MemmapDataset.

        Downloads train.bin / val.bin / meta.json once; skips files that
        already exist on subsequent calls (idempotent).

        Args:
            hub_repo  : HF Hub dataset repo, e.g. "username/tinystories-gpt2-tokenized".
            split     : "train" or "val".
            seq_len   : Sequence length. Reads from meta.json if not given.
            local_dir : Where to store downloaded files. Defaults to
                        "data/{repo_name}".

        Usage in Colab::

            train_ds = MemmapDataset.from_hub(
                "username/tinystories-gpt2-tokenized",
                split="train", seq_len=1024,
            )
        """
        from engine.legacy.data.pretokenize import pull_tokenized

        if local_dir is None:
            local_dir = Path("data") / hub_repo.split("/")[-1]

        data_dir = pull_tokenized(hub_repo, local_dir, token_env=token_env)
        return cls.from_dir(data_dir, split=split, seq_len=seq_len, **kwargs)

    def __repr__(self) -> str:
        return (
            f"MemmapDataset(path={self.path.name!r}, "
            f"seq_len={self.seq_len}, "
            f"chunks={len(self):,}, "
            f"tokens={self.num_tokens:,})"
        )
