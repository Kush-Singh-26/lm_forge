import sys
from pathlib import Path
from transformers import AutoTokenizer
import torch
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.models.hf_model import HFCausalLM
from engine.config.hf_config import LMForgeConfig


def test_save_pretrained():
    save_dir = "test_hf_save"
    if Path(save_dir).exists():
        shutil.rmtree(save_dir)

    # Create a small model
    config = LMForgeConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )
    model = HFCausalLM(config)

    # Get a dummy tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Saving to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Files in save_dir:")
    for f in Path(save_dir).iterdir():
        print(f" - {f.name}")

    # Check for expected files
    expected = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    for e in expected:
        if not (Path(save_dir) / e).exists():
            print(f"Missing expected file: {e}")
        else:
            print(f"Found: {e}")


if __name__ == "__main__":
    test_save_pretrained()
