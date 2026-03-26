"""
engine/data/hub_prepare.py

Modern HF-native data preparation script.
Tokenizes and packs a dataset, then saves it as a standard HF Dataset (Arrow format).
This dataset can then be used with load_dataset() or pushed to the Hub.

Usage:
    python -m engine.data.hub_prepare \\
        --dataset roneneldan/TinyStories \\
        --tokenizer gpt2 \\
        --seq_len 512 \\
        --output_dir data/tinystories_packed \\
        --push_to_hub YOUR_USERNAME/tinystories-packed
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from engine.data.hf_utils import prepare_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="HF dataset name")
    p.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path")
    p.add_argument("--seq_len", type=int, default=512, help="Target sequence length")
    p.add_argument("--split", type=str, default="train", help="Which split to process")
    p.add_argument(
        "--output_dir", type=str, default=None, help="Local directory to save"
    )
    p.add_argument(
        "--push_to_hub", type=str, default=None, help="HF Hub repo to push to"
    )
    p.add_argument(
        "--num_proc", type=int, default=4, help="Number of processes for mapping"
    )
    args = p.parse_args()

    print(f"--- Preparing {args.dataset} [{args.split}] ---")

    # 1. Load
    print(f"Loading dataset...")
    ds = load_dataset(args.dataset, split=args.split)

    # 2. Tokenize & Pack
    print(f"Loading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Processing (seq_len={args.seq_len})...")
    packed_ds = prepare_dataset(
        ds, tokenizer, seq_len=args.seq_len, num_proc=args.num_proc, shuffle=True
    )

    # 3. Save / Push
    if args.output_dir:
        print(f"Saving locally to {args.output_dir}...")
        packed_ds.save_to_disk(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing to HF Hub: {args.push_to_hub}...")
        packed_ds.push_to_hub(args.push_to_hub)

    print("Done!")


if __name__ == "__main__":
    main()
