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
            # Load BPE if a file is provided
            if tokenizer_file and Path(tokenizer_file).exists():
                self.bpe = BPETokenizer.load(tokenizer_file)
            else:
                self.bpe = BPETokenizer(
                    special_tokens=[
                        pad_token,
                        unk_token,
                        bos_token,
                        eos_token,
                        mask_token,
                    ]
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
            # Use string-based tokenization by encoding and then decoding IDs to tokens
            ids = self.bpe.encode(text)
            return [self.bpe.vocab_inv[i] for i in ids]

        def _convert_token_to_id(self, token: str) -> int:
            return self.bpe.vocab.get(token, self.unk_token_id)

        def _convert_id_to_token(self, index: int) -> str:
            return self.bpe.vocab_inv.get(index, self.unk_token)

        def convert_tokens_to_string(self, tokens: List[str]) -> str:
            # Reconstruct the string from tokens
            byte_list: List[int] = []
            special_tokens_set = set(self.all_special_tokens)

            for tok in tokens:
                if tok in special_tokens_set:
                    continue
                for ch in tok:
                    byte_list.append(self.bpe.byte_decoder[ch])

            return bytes(byte_list).decode("utf-8", errors="replace")

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
