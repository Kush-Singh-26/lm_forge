"""
engine/data/pretokenize.py

Pre-tokenize a dataset once, save as a uint16 numpy memmap, push to HF Hub.

This is the correct way to handle data for LM training:

    ┌─────────────────────────────────────────────────────────┐
    │  Raw text corpus (HF dataset / local files)             │
    │          ↓  run once, takes minutes                     │
    │  Tokenized flat array (uint16 .bin files)               │
    │          ↓  push once to HF Hub                         │
    │  HF Hub dataset repo  (train.bin + val.bin + meta.json) │
    │          ↓  pull at session start, ~seconds             │
    │  Training loop reads via np.memmap (zero RAM overhead)  │
    └─────────────────────────────────────────────────────────┘

Why uint16?
  • Saves 2× space vs int32 and 4× vs int64
  • Supports vocab_size up to 65535 — covers GPT-2 (50257), LLaMA (32000),
    and all common tokenizers
  • numpy memmap over uint16 means the OS pages in only what the training
    loop needs — effectively zero RAM cost for a 10GB dataset

Why HF Hub dataset repo?
  • Colab pulls it in one line at session start
  • Version-controlled (each push is a commit)
  • Free, private repos supported
  • Compatible with datasets.load_dataset() for inspection

CLI usage:
    # Tokenize roneneldan/TinyStories, push to your Hub
    python -m engine.data.pretokenize \\
        --dataset roneneldan/TinyStories \\
        --tokenizer gpt2 \\
        --hub_repo YOUR_USERNAME/tinystories-gpt2-tokenized \\
        --output_dir data/tinystories_gpt2 \\
        --val_split 0.005

    # Local files only, no Hub push
    python -m engine.data.pretokenize \\
        --dataset roneneldan/TinyStories \\
        --tokenizer gpt2 \\
        --output_dir data/tinystories_gpt2 \\
        --no_push

Python API:
    from engine.data.pretokenize import pretokenize, TokenizedDataInfo

    info = pretokenize(
        dataset_name="roneneldan/TinyStories",
        tokenizer_name="gpt2",
        output_dir="data/tinystories_gpt2",
        hub_repo="username/tinystories-gpt2-tokenized",
    )
    print(info)
    # TokenizedDataInfo(train_tokens=474_069_024, val_tokens=2_369_024,
    #                   vocab_size=50257, seq_recommended=1024)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# ── Output info dataclass ─────────────────────────────────────────────────────


@dataclass
class TokenizedDataInfo:
    """Metadata written to meta.json alongside the .bin files."""

    dataset_name: str
    tokenizer_name: str
    vocab_size: int
    train_tokens: int
    val_tokens: int
    train_file: str = "train.bin"
    val_file: str = "val.bin"
    meta_file: str = "meta.json"
    eos_token_id: int = -1
    bos_token_id: int = -1
    seq_recommended: int = 1024  # sensible seq_len for this dataset

    def __str__(self) -> str:
        return (
            f"TokenizedDataInfo\n"
            f"  dataset    : {self.dataset_name}\n"
            f"  tokenizer  : {self.tokenizer_name} (vocab={self.vocab_size})\n"
            f"  train      : {self.train_tokens:,} tokens  ({self.train_tokens / 1e9:.2f}B)\n"
            f"  val        : {self.val_tokens:,} tokens\n"
            f"  dtype      : uint16 (2 bytes/token)\n"
            f"  disk (train): {self.train_tokens * 2 / 1024**3:.2f} GB\n"
            f"  recommended seq_len: {self.seq_recommended}"
        )


# ── Core tokenization function ────────────────────────────────────────────────


def pretokenize(
    dataset_name: str,
    tokenizer_name: str,
    output_dir: str | Path,
    hub_repo: Optional[str] = None,
    text_column: str = "text",
    train_split: str = "train",
    val_split: str = "validation",
    val_fraction: float = 0.005,
    max_train_docs: Optional[int] = None,
    num_proc: int = 4,
    hub_private: bool = True,
    hub_token_env: str = "HF_TOKEN",
    verbose: bool = True,
    add_bos: bool = False,
) -> TokenizedDataInfo:
    """
    Tokenize a HuggingFace dataset and save as uint16 memmap files.

    Args:
        dataset_name   : HF dataset identifier, e.g. "roneneldan/TinyStories".
        tokenizer_name : HF tokenizer identifier, e.g. "gpt2", "meta-llama/Llama-2-7b-hf".
        output_dir     : Local directory for .bin and meta.json files.
        hub_repo       : HF Hub dataset repo, e.g. "username/dataset-tokenized".
                         If None or empty, no Hub push is performed.
        text_column    : Column containing raw text in the dataset.
        train_split    : Dataset split to use for training.
        val_split      : Dataset split for validation. If it doesn't exist,
                         val_fraction of train is used instead.
        val_fraction   : Fraction of train to use as val when no val split exists.
        max_train_docs : Cap on training documents (useful for quick tests).
        num_proc       : Parallel workers for tokenization.
        hub_private    : Make Hub repo private.
        hub_token_env  : Env var name holding the HF write token.
        verbose        : Print progress.
        add_bos        : Prepend BOS token to each document (required for some models like LLaMA).

    Returns:
        TokenizedDataInfo with token counts and file paths.
    """
    try:
        from datasets import load_dataset, DatasetDict
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "datasets and transformers are required for pre-tokenization.\n"
            "Run: pip install datasets transformers"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load tokenizer ─────────────────────────────────────────────────────
    if verbose:
        print(f"\n[pretokenize] Loading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tok.vocab_size
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else -1

    if vocab_size > 65535:
        raise ValueError(
            f"Tokenizer vocab_size={vocab_size} exceeds uint16 max (65535). "
            "Use a tokenizer with a smaller vocabulary, or change the dtype to uint32."
        )

    if verbose:
        print(
            f"[pretokenize] vocab_size={vocab_size}, bos_id={bos_id}, eos_id={eos_id}"
        )

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    if verbose:
        print(f"[pretokenize] Loading dataset: {dataset_name}")

    try:
        ds = load_dataset(dataset_name, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    # Get train split
    train_ds = ds[train_split]
    if max_train_docs is not None:
        train_ds = train_ds.select(range(min(max_train_docs, len(train_ds))))

    # Get or create val split
    if val_split in ds:
        val_ds = ds[val_split]
    else:
        if verbose:
            print(
                f"[pretokenize] No '{val_split}' split — using {val_fraction:.1%} of train"
            )
        n_val = max(1, int(len(train_ds) * val_fraction))
        split = train_ds.train_test_split(test_size=n_val, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]

    if verbose:
        print(f"[pretokenize] Train docs: {len(train_ds):,}  Val docs: {len(val_ds):,}")

    # ── 3. Tokenize ───────────────────────────────────────────────────────────

    def _tokenize_batch(batch):
        """Tokenize a batch and append EOS/BOS to each document."""
        all_ids = []
        for text in batch[text_column]:
            ids = tok.encode(text, add_special_tokens=False)
            if add_bos and bos_id >= 0:
                ids.insert(0, bos_id)
            if eos_id >= 0:
                ids.append(eos_id)
            all_ids.extend(ids)
        return {"ids": [all_ids]}

    if verbose:
        print("[pretokenize] Tokenizing train split...")

    t0 = time.perf_counter()
    # Batch tokenization: map() processes in parallel, then flatten
    train_tok = train_ds.map(
        _tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    val_tok = val_ds.map(
        _tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    if verbose:
        print(f"[pretokenize] Tokenization done in {time.perf_counter() - t0:.0f}s")

    # ── 4. Write memmap .bin files ─────────────────────────────────────────────

    def _write_bin(tokenized_ds, path: Path, split_name: str) -> int:
        """Flatten all token lists and write as uint16 binary file."""
        # Count total tokens first
        total = sum(len(row["ids"]) for row in tokenized_ds)
        if verbose:
            print(f"[pretokenize] Writing {split_name}: {total:,} tokens → {path}")

        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        cursor = 0
        for row in tokenized_ds:
            ids = row["ids"]
            n = len(ids)
            arr[cursor : cursor + n] = ids
            cursor += n
        arr.flush()
        del arr
        return total

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.json"

    train_tokens = _write_bin(train_tok, train_path, "train")
    val_tokens = _write_bin(val_tok, val_path, "val")

    # ── 5. Write meta.json ────────────────────────────────────────────────────

    # Recommend a seq_len based on dataset size
    # Rule of thumb: ~500 steps worth of tokens at batch=8, plus some headroom
    recommended_seq = min(2048, max(256, 2 ** int(np.log2(train_tokens / 500 / 8))))

    info = TokenizedDataInfo(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        train_tokens=int(train_tokens),
        val_tokens=int(val_tokens),
        eos_token_id=int(eos_id),
        bos_token_id=int(bos_id),
        seq_recommended=int(recommended_seq),
    )
    meta_path.write_text(json.dumps(asdict(info), indent=2))

    if verbose:
        print(f"\n{info}")

    # ── 6. Push to HF Hub ─────────────────────────────────────────────────────

    if hub_repo:
        _push_to_hub(
            output_dir=output_dir,
            hub_repo=hub_repo,
            hub_private=hub_private,
            token_env=hub_token_env,
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            info=info,
            verbose=verbose,
        )

    return info


def _push_to_hub(
    output_dir: Path,
    hub_repo: str,
    hub_private: bool,
    token_env: str,
    dataset_name: str,
    tokenizer_name: str,
    info: TokenizedDataInfo,
    verbose: bool,
) -> None:
    """Push .bin files and meta.json to a HF Hub dataset repo."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[pretokenize] huggingface_hub not installed — skipping Hub push.")
        return

    token = os.environ.get(token_env, "")
    if not token:
        print(f"[pretokenize] {token_env} not set — skipping Hub push.")
        return

    api = HfApi(token=token)

    # Create the dataset repo
    try:
        create_repo(hub_repo, repo_type="dataset", private=hub_private, exist_ok=True)
    except Exception as e:
        print(f"[pretokenize] Could not create Hub repo: {e}")
        return

    if verbose:
        print(f"\n[pretokenize] Pushing to Hub: {hub_repo}")

    # Upload all files in output_dir
    failed_uploads = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for fname in ["train.bin", "val.bin", "meta.json"]:
            fpath = output_dir / fname
            if fpath.exists():
                future = executor.submit(
                    api.upload_file,
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=hub_repo,
                    repo_type="dataset",
                    commit_message=f"Tokenized {dataset_name} with {tokenizer_name}",
                )
                futures[future] = fname

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Uploading to Hub"
        ):
            fname = futures[future]
            try:
                future.result()
            except Exception as e:
                failed_uploads.append(fname)
                print(f"Failed to upload {fname}: {e}")

    # Refuse to leave the Hub in a corrupt partial state
    critical = {"train.bin", "meta.json"}
    if critical & set(failed_uploads):
        raise RuntimeError(
            f"Critical file(s) failed to upload: {failed_uploads}. "
            f"Hub repo {hub_repo} may be in an inconsistent state."
        )

    # Auto-generate a dataset card
    card = f"""---
license: other
tags:
  - tokenized
  - lm-forge
---

# {hub_repo.split("/")[-1]}

Pre-tokenized dataset produced by [lm_forge](https://github.com).

| Field | Value |
|---|---|
| Source dataset | `{dataset_name}` |
| Tokenizer | `{tokenizer_name}` |
| Vocab size | {info.vocab_size:,} |
| Train tokens | {info.train_tokens:,} ({info.train_tokens / 1e9:.2f}B) |
| Val tokens | {info.val_tokens:,} |
| Dtype | uint16 (2 bytes/token) |
| Recommended seq_len | {info.seq_recommended} |

## Usage with lm_forge

```python
from engine.data import MemmapDataset, build_dataloader

train_ds = MemmapDataset.from_hub("{hub_repo}", split="train", seq_len=1024)
loader   = build_dataloader(train_ds, batch_size=8)
```
"""
    try:
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=hub_repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    except Exception as e:
        print(f"  README.md upload failed: {e}")

    if verbose:
        print(f"[pretokenize] Done → https://huggingface.co/datasets/{hub_repo}")


# ── Pull helper ───────────────────────────────────────────────────────────────


def pull_tokenized(
    hub_repo: str,
    local_dir: str | Path,
    token_env: str = "HF_TOKEN",
    verbose: bool = True,
) -> Path:
    """
    Download a pre-tokenized dataset from HF Hub to a local directory.

    Returns the local path containing train.bin, val.bin, meta.json.

    Usage in Colab::

        from engine.data.pretokenize import pull_tokenized

        data_dir = pull_tokenized(
            "username/tinystories-gpt2-tokenized",
            "data/tinystories_gpt2",
        )
        train_ds = MemmapDataset(data_dir / "train.bin", seq_len=1024)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(token_env, None)

    for fname in ["train.bin", "val.bin", "meta.json"]:
        target = local_dir / fname
        if target.exists():
            if verbose:
                print(f"[pull_tokenized] {fname} already exists — skipping")
            continue
        if verbose:
            print(f"[pull_tokenized] Downloading {fname} from {hub_repo}...")
        try:
            downloaded = hf_hub_download(
                hub_repo,
                filename=fname,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=token,
            )
        except Exception as e:
            if verbose:
                print(f"  {fname} not found or download failed: {e}")

    return local_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a HF dataset and optionally push to Hub."
    )
    parser.add_argument("--dataset", required=True, help="HF dataset name")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer name")
    parser.add_argument("--output_dir", required=True, help="Local output directory")
    parser.add_argument("--hub_repo", default="", help="HF Hub dataset repo (optional)")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--val_fraction", type=float, default=0.005)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--no_push", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument(
        "--add_bos", action="store_true", help="Prepend BOS token to each document"
    )
    args = parser.parse_args()

    info = pretokenize(
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        hub_repo="" if args.no_push else args.hub_repo,
        text_column=args.text_column,
        val_split=args.val_split,
        val_fraction=args.val_fraction,
        max_train_docs=args.max_docs,
        num_proc=args.num_proc,
        hub_private=not args.public,
        add_bos=args.add_bos,
    )
    print(
        f"\nDone. To use in training:\n"
        f"  train_ds = MemmapDataset('{args.output_dir}/train.bin', seq_len=1024)"
    )
