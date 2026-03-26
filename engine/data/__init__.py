from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from engine.data.collators import CLMCollator, MLMCollator
from engine.data.hf_utils import prepare_dataset


def _default_num_workers() -> int:
    """
    Sensible num_workers default:
      - 0 on Windows (fork not supported)
      - 2 in Colab/Jupyter
      - 4 elsewhere
    """
    import platform

    if platform.system() == "Windows":
        return 0
    # Check if we're in a Colab/Jupyter environment
    try:
        import IPython

        if IPython.get_ipython() is not None:
            return 2
    except ImportError:
        pass
    return 4


def build_dataloader(
    dataset: Dataset | IterableDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    collator=None,
    pad_id: int = 0,
    max_seq_len: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build a DataLoader with performance-tuned defaults.
    """
    is_iterable = isinstance(dataset, IterableDataset)

    # ── Workers ──────────────────────────────────────────────────────────────
    if num_workers is None:
        num_workers = _default_num_workers()

    # ── Pin memory ───────────────────────────────────────────────────────────
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # ── Collator ──────────────────────────────────────────────────────────────
    if collator is None:
        collator = CLMCollator(pad_id=pad_id, max_seq_len=max_seq_len)

    # prefetch_factor is only valid when num_workers > 0
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle and not is_iterable),
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True  # avoid re-forking each epoch

    return DataLoader(dataset, **loader_kwargs)


__all__ = [
    "CLMCollator",
    "MLMCollator",
    "build_dataloader",
    "prepare_dataset",
]
