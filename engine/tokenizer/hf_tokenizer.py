"""
engine/tokenizer/hf_tokenizer.py

HFBPETokenizer — A PreTrainedTokenizer wrapper around the engine's BPETokenizer.
This allows our custom BPE to be used with the HF ecosystem.
"""

from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

try:
    from transformers import PreTrainedTokenizer
    from engine.tokenizer.bpe import BPETokenizer

    class HFBPETokenizer(PreTrainedTokenizer):
        """
        HF-compatible wrapper for BPETokenizer.
        """

        model_input_names = ["input_ids", "attention_mask"]

        def __init__(
            self,
            tokenizer_file: Optional[str] = None,
            bos_token="<BOS>",
            eos_token="<EOS>",
            unk_token="<UNK>",
            pad_token="<PAD>",
            mask_token="<MASK>",
            **kwargs,
        ):
            self.bpe = (
                BPETokenizer.load(tokenizer_file) if tokenizer_file else BPETokenizer()
            )
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token,
                **kwargs,
            )

        @property
        def vocab_size(self) -> int:
            return self.bpe.vocab_size

        def get_vocab(self) -> Dict[str, int]:
            return self.bpe.vocab.copy()

        def _tokenize(self, text: str) -> List[str]:
            # BPETokenizer.encode returns IDs, but _tokenize should return tokens
            # This is a bit inefficient for our BPE which merges strings directly
            # We'll re-implement string-based tokenization if needed, but for now:
            ids = self.bpe.encode(text)
            return [self.bpe.vocab_inv[i] for i in ids]

        def _convert_token_to_id(self, token: str) -> int:
            return self.bpe.vocab.get(token, self.bpe.special_tokens.get("<UNK>", 1))

        def _convert_id_to_token(self, index: int) -> str:
            return self.bpe.vocab_inv.get(index, "<UNK>")

        def convert_tokens_to_string(self, tokens: List[str]) -> str:
            # BPETokenizer.decode works on IDs
            ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
            return self.bpe.decode(ids)

        def save_vocabulary(
            self, save_directory: str, filename_prefix: Optional[str] = None
        ) -> tuple[str]:
            path = Path(save_directory) / (
                (filename_prefix + "-" if filename_prefix else "") + "tokenizer.json"
            )
            self.bpe.save(path)
            return (str(path),)

except ImportError:

    class HFBPETokenizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("transformers is required for HFBPETokenizer")
