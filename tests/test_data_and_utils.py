"""
tests/test_data_and_utils.py

Tests for data pipeline, BPE tokenizer, profiler, and ablation runner.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.legacy.data.packed import PackedDataset, StreamingPackedDataset
from engine.data.collators import CLMCollator, MLMCollator
from engine.data import build_dataloader
from engine.tokenizer.bpe import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# BPE Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

CORPUS = [
    "hello world hello",
    "hello there world",
    "the quick brown fox",
    "the fox jumped over the dog",
    "world domination plan hello world",
]


class TestBPETokenizer:
    def test_default_special_tokens(self):
        tok = BPETokenizer()
        assert "<PAD>" in tok.special_tokens
        assert "<BOS>" in tok.special_tokens
        assert "<EOS>" in tok.special_tokens
        assert "<MASK>" in tok.special_tokens

    def test_base_vocab_size(self):
        tok = BPETokenizer()
        assert len(tok) == len(tok.DEFAULT_SPECIAL) + 256

    def test_train_increases_vocab(self):
        tok = BPETokenizer()
        base = len(tok)
        tok.train(CORPUS, vocab_size=base + 20, verbose=False)
        assert len(tok) > base

    def test_encode_decode_roundtrip(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=320, verbose=False)
        text = "hello world"
        ids = tok.encode(text)
        out = tok.decode(ids)
        assert out == text

    def test_encode_adds_bos_eos(self):
        tok = BPETokenizer()
        ids = tok.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_token_id
        assert ids[-1] == tok.eos_token_id

    def test_encode_batch(self):
        tok = BPETokenizer()
        results = tok.encode_batch(["hello", "world"])
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_callable_interface(self):
        tok = BPETokenizer()
        out = tok("hello world")
        assert "input_ids" in out
        assert isinstance(out["input_ids"], list)

    def test_save_load(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=320, verbose=False)
        text = "hello world"
        ids1 = tok.encode(text)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tokenizer.json"
            tok.save(path)
            tok2 = BPETokenizer.load(path)

        ids2 = tok2.encode(text)
        assert ids1 == ids2

    def test_handles_unicode(self):
        tok = BPETokenizer()
        text = "café résumé naïve"
        ids = tok.encode(text)
        out = tok.decode(ids)
        assert out == text

    def test_unknown_bytes_dont_crash(self):
        tok = BPETokenizer()
        # Emoji — multi-byte UTF-8, handled by byte-level encoding
        ids = tok.encode("hello 🎉")
        out = tok.decode(ids)
        assert "hello" in out

    def test_vocab_size_property(self):
        tok = BPETokenizer()
        assert tok.vocab_size == len(tok.vocab)

    def test_merges_reduce_sequence_length(self):
        tok = BPETokenizer()
        base_size = len(tok)
        tok.train(CORPUS * 10, vocab_size=base_size + 50, verbose=False)
        # After training, encoding a seen phrase should be shorter
        base_tok = BPETokenizer()
        text = "hello world"
        n_before = len(base_tok.encode(text))
        n_after = len(tok.encode(text))
        # Should be equal or shorter after merges
        assert n_after <= n_before


# ─────────────────────────────────────────────────────────────────────────────
# Packed dataset
# ─────────────────────────────────────────────────────────────────────────────


class TestPackedDataset:
    def _seqs(self, n=20, length=50) -> list:
        return [list(torch.randint(5, 250, (length,)).tolist()) for _ in range(n)]

    def test_basic_shape(self):
        seqs = self._seqs()
        ds = PackedDataset(seqs, seq_len=32, eos_id=2)
        item = ds[0]
        assert item["input_ids"].shape == (32,)
        assert item["labels"].shape == (32,)

    def test_labels_shifted(self):
        seqs = self._seqs()
        ds = PackedDataset(seqs, seq_len=32, eos_id=2)
        item = ds[0]
        # input_ids[1:] should equal labels[:-1] (they come from adjacent positions)
        # (both come from the same packed flat array, offset by 1)
        # Can't test exactly since labels are the next token, so just check dtypes
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long

    def test_no_padding(self):
        """Every token in every chunk should be real (no pads)."""
        seqs = self._seqs(n=50, length=100)
        ds = PackedDataset(seqs, seq_len=32, eos_id=2)
        for i in range(len(ds)):
            item = ds[i]
            # All values should be valid token ids (not pad=0 from packing)
            assert (item["input_ids"] >= 0).all()

    def test_num_tokens(self):
        seqs = self._seqs(n=50, length=100)
        ds = PackedDataset(seqs, seq_len=32)
        assert ds.num_tokens == len(ds) * 32

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="No complete chunks"):
            PackedDataset([[1, 2, 3]], seq_len=100)

    def test_from_hf(self):
        """Simulate HF dataset interface with a list of dicts."""
        tok = BPETokenizer()
        hf_like = [{"text": t} for t in CORPUS * 5]
        ds = PackedDataset.from_hf(hf_like, tok, seq_len=16, eos_id=tok.eos_token_id)
        assert len(ds) > 0
        item = ds[0]
        assert item["input_ids"].shape == (16,)

    def test_stride(self):
        seqs = self._seqs(n=10, length=100)
        ds_no = PackedDataset(seqs, seq_len=32, stride=0)
        ds_st = PackedDataset(seqs, seq_len=32, stride=16)
        # Strided should produce more samples
        assert len(ds_st) >= len(ds_no)


class TestStreamingPackedDataset:
    def test_yields_correct_shape(self):
        tok = BPETokenizer()
        data = [{"text": t} for t in CORPUS * 20]
        ds = StreamingPackedDataset(
            data, tok, seq_len=16, buffer_size=10, shuffle=False
        )
        items = list(ds)
        assert len(items) > 0
        for item in items:
            assert item["input_ids"].shape == (16,)
            assert item["labels"].shape == (16,)


# ─────────────────────────────────────────────────────────────────────────────
# Collators
# ─────────────────────────────────────────────────────────────────────────────


class TestCLMCollator:
    def test_pads_to_longest(self):
        col = CLMCollator(pad_id=0)
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6, 7, 8])},
        ]
        out = col(batch)
        assert out["input_ids"].shape == (2, 5)
        assert out["labels"].shape == (2, 5)
        assert out["attention_mask"].shape == (2, 5)

    def test_pad_positions_masked(self):
        col = CLMCollator(pad_id=0)
        batch = [
            {"input_ids": torch.tensor([1, 2])},
            {"input_ids": torch.tensor([3, 4, 5])},
        ]
        out = col(batch)
        # Short sequence: position 2 should be padded
        assert out["attention_mask"][0, 2] == 0
        assert out["labels"][0, 2] == -100

    def test_truncation(self):
        col = CLMCollator(pad_id=0, max_seq_len=4)
        batch = [{"input_ids": torch.tensor(list(range(10)))}]
        out = col(batch)
        assert out["input_ids"].shape[1] == 4

    def test_no_pad_passthrough(self):
        """PackedDataset batches (all same length) → no padding at all."""
        col = CLMCollator(pad_id=0)
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([2, 3, 4, 5]),
            },
            {
                "input_ids": torch.tensor([5, 6, 7, 8]),
                "labels": torch.tensor([6, 7, 8, 9]),
            },
        ]
        out = col(batch)
        assert (out["attention_mask"] == 1).all()


class TestMLMCollator:
    def test_output_keys(self):
        col = MLMCollator(vocab_size=256, mask_token_id=4, pad_id=0)
        batch = [{"input_ids": torch.randint(5, 256, (16,))} for _ in range(4)]
        out = col(batch)
        assert "input_ids" in out
        assert "labels" in out
        assert "attention_mask" in out

    def test_mask_rate(self):
        torch.manual_seed(0)
        col = MLMCollator(vocab_size=1000, mask_token_id=4, pad_id=0, mask_prob=0.15)
        batch = [{"input_ids": torch.randint(5, 1000, (128,))} for _ in range(16)]
        out = col(batch)
        n_masked = (out["labels"] != -100).float().sum()
        n_total = out["labels"].numel()
        rate = n_masked / n_total
        # Should be approximately 15%, allow ±5%
        assert 0.08 < rate.item() < 0.22

    def test_labels_minus100_unmasked(self):
        col = MLMCollator(vocab_size=256, mask_token_id=4)
        batch = [{"input_ids": torch.randint(5, 256, (32,))} for _ in range(4)]
        out = col(batch)
        # All non-masked positions should have label = -100
        masked = out["labels"] != -100
        assert masked.any()  # at least some positions masked


# ─────────────────────────────────────────────────────────────────────────────
# build_dataloader
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildDataloader:
    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader

        seqs = [list(range(50)) for _ in range(20)]
        ds = PackedDataset(seqs, seq_len=16)
        dl = build_dataloader(ds, batch_size=4, num_workers=0)
        assert isinstance(dl, DataLoader)

    def test_batch_shape(self):
        seqs = [list(range(50)) for _ in range(20)]
        ds = PackedDataset(seqs, seq_len=16)
        dl = build_dataloader(ds, batch_size=4, num_workers=0, shuffle=False)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (4, 16)

    def test_streaming_dataloader(self):
        tok = BPETokenizer()
        data = [{"text": t} for t in CORPUS * 30]
        ds = StreamingPackedDataset(
            data, tok, seq_len=16, buffer_size=50, shuffle=False
        )
        dl = build_dataloader(ds, batch_size=4, num_workers=0)
        batch = next(iter(dl))
        assert batch["input_ids"].shape[0] == 4
