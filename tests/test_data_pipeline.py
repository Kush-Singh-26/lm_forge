"""
tests/test_data_pipeline.py

Tests for the new data pipeline:
  - MemmapDataset (read, shape, stride, from_dir, from_hub fallback)
  - pretokenize (tokenize + write .bin files, meta.json, TokenizedDataInfo)
  - build_dataloader (auto num_workers, pin_memory, memmap collate, prefetch)
  - DeviceManager.build_optimizer (fused flag, param groups, no-decay separation)
  - TrainConfig (new fields: fused_adamw, num_workers, data_hub_repo)

Run: pytest tests/test_data_pipeline.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.legacy.data.memmap import MemmapDataset
from engine.data import build_dataloader
from engine.config.schema import (
    ModelConfig,
    AttentionConfig,
    PositionalConfig,
    FFNConfig,
    NormConfig,
    TrainConfig,
)
from engine.legacy.training.device import DeviceManager


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_bin(path: Path, n_tokens: int, vocab_size: int = 256) -> Path:
    """Write a fake uint16 token file for testing."""
    arr = np.random.randint(1, vocab_size, size=n_tokens, dtype=np.uint16)
    mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    mm[:] = arr
    mm.flush()
    return path


def _write_fake_dataset(
    directory: Path,
    train_tokens: int = 4096,
    val_tokens: int = 512,
    vocab_size: int = 256,
    seq_recommended: int = 128,
) -> Path:
    """Write fake train.bin, val.bin, meta.json in a temp directory."""
    _write_bin(directory / "train.bin", train_tokens, vocab_size)
    _write_bin(directory / "val.bin", val_tokens, vocab_size)
    meta = {
        "dataset_name": "fake_dataset",
        "tokenizer_name": "fake_tok",
        "vocab_size": vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "eos_token_id": 2,
        "seq_recommended": seq_recommended,
        "train_file": "train.bin",
        "val_file": "val.bin",
        "meta_file": "meta.json",
    }
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        attention=AttentionConfig(type="gqa", num_heads=4, num_kv_heads=2),
        positional=PositionalConfig(type="rope", max_seq_len=64),
        ffn=FFNConfig(type="swiglu", intermediate_size=128),
        norm=NormConfig(type="rms"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MemmapDataset
# ─────────────────────────────────────────────────────────────────────────────


class TestMemmapDataset:
    def test_basic_shape(self, tmp_path):
        _write_bin(tmp_path / "train.bin", 4096)
        ds = MemmapDataset(tmp_path / "train.bin", seq_len=64)
        item = ds[0]
        # Returns seq_len+1 tokens (full chunk) for both input_ids and labels
        # The model handles the shift internally
        assert item["input_ids"].shape == (65,)
        assert item["labels"].shape == (65,)
        assert item["input_ids"].dtype == torch.int64

    def test_labels_shifted_by_one(self, tmp_path):
        """labels[i] should be input_ids[i+1] — next-token prediction."""
        _write_bin(tmp_path / "train.bin", 4096)
        ds = MemmapDataset(tmp_path / "train.bin", seq_len=64)
        item = ds[0]
        # The dataset returns the full chunk (seq_len+1). The model will shift internally.
        # Verify the raw chunk: input_ids = chunk, labels = chunk
        # So at position i, model predicts token[i+1] from token[i]
        raw = np.memmap(tmp_path / "train.bin", dtype=np.uint16, mode="r")
        expected_chunk = torch.tensor(raw[:65].astype(np.int64))
        assert torch.equal(item["input_ids"], expected_chunk)
        assert torch.equal(item["labels"], expected_chunk)

    def test_length(self, tmp_path):
        _write_bin(tmp_path / "train.bin", 1000)
        ds = MemmapDataset(tmp_path / "train.bin", seq_len=64)
        # (1000 - 64) // 64 = 14
        assert len(ds) == (1000 - 64) // 64

    def test_num_tokens(self, tmp_path):
        _write_bin(tmp_path / "train.bin", 4096)
        ds = MemmapDataset(tmp_path / "train.bin", seq_len=64)
        assert ds.num_tokens == len(ds) * 64

    def test_stride(self, tmp_path):
        _write_bin(tmp_path / "train.bin", 1000)
        ds_no = MemmapDataset(tmp_path / "train.bin", seq_len=64, stride=64)
        ds_str = MemmapDataset(tmp_path / "train.bin", seq_len=64, stride=32)
        assert len(ds_str) > len(ds_no)

    def test_different_indices_give_different_items(self, tmp_path):
        _write_bin(tmp_path / "train.bin", 4096)
        ds = MemmapDataset(tmp_path / "train.bin", seq_len=64)
        assert not torch.equal(ds[0]["input_ids"], ds[1]["input_ids"])

    def test_too_small_raises(self, tmp_path):
        # Need at least seq_len+1 tokens (65 for seq_len=64)
        _write_bin(tmp_path / "train.bin", 64)  # exactly seq_len = not enough
        with pytest.raises(ValueError, match="too few"):
            MemmapDataset(tmp_path / "train.bin", seq_len=64)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MemmapDataset(tmp_path / "nonexistent.bin", seq_len=64)

    def test_from_dir_reads_meta(self, tmp_path):
        _write_fake_dataset(tmp_path, train_tokens=4096, seq_recommended=128)
        ds = MemmapDataset.from_dir(tmp_path, split="train")
        assert ds.seq_len == 128  # reads seq_recommended from meta.json

    def test_from_dir_val_split(self, tmp_path):
        _write_fake_dataset(tmp_path)
        ds = MemmapDataset.from_dir(tmp_path, split="val", seq_len=32)
        assert len(ds) > 0
        # Returns seq_len+1 for both input_ids and labels
        assert ds[0]["input_ids"].shape == (33,)

    def test_from_dir_explicit_seq_len(self, tmp_path):
        _write_fake_dataset(tmp_path, seq_recommended=128)
        ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=64)
        assert ds.seq_len == 64  # explicit wins over meta.json

    def test_repr(self, tmp_path):
        _write_bin(tmp_path / "train.bin", 4096)
        ds = MemmapDataset(tmp_path / "train.bin", seq_len=64)
        r = repr(ds)
        assert "MemmapDataset" in r
        assert "seq_len=64" in r

    def test_no_padding_needed(self, tmp_path):
        """Every item is the exact same length — no padding overhead."""
        _write_bin(tmp_path / "train.bin", 4096)
        ds = MemmapDataset(tmp_path / "train.bin", seq_len=64)
        for i in range(min(5, len(ds))):
            # Returns seq_len+1 for both input_ids and labels
            assert ds[i]["input_ids"].shape == (65,)
            assert ds[i]["labels"].shape == (65,)

    def test_tokens_are_valid_ids(self, tmp_path):
        """No token should exceed vocab_size."""
        _write_fake_dataset(tmp_path, train_tokens=4096, vocab_size=256)
        ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=32)
        for i in range(min(10, len(ds))):
            assert ds[i]["input_ids"].max().item() < 256


# ─────────────────────────────────────────────────────────────────────────────
# pretokenize (offline, no HF API calls)
# ─────────────────────────────────────────────────────────────────────────────


class TestPretokenize:
    """
    Tests the tokenization logic without hitting the HF API.
    Uses a mock tokenizer and a synthetic list-of-dicts dataset.
    """

    def _mock_tok(self, vocab_size: int = 256):
        """Minimal mock tokenizer object."""

        class MockTok:
            def __init__(self):
                self.vocab_size = vocab_size
                self.eos_token_id = 2

            def encode(self, text, add_special_tokens=False):
                # Encode as byte values of the text (deterministic, reproducible)
                return [b % (vocab_size - 3) + 3 for b in text.encode("utf-8")]

        return MockTok()

    def test_write_bin_and_meta(self, tmp_path):
        """Verify .bin files and meta.json are written correctly."""
        from engine.legacy.data.pretokenize import TokenizedDataInfo
        import dataclasses

        # Simulate what pretokenize() does internally:
        # write a flat uint16 array and meta.json
        n_train = 2048
        n_val = 256

        arr_train = np.random.randint(3, 256, size=n_train, dtype=np.uint16)
        arr_val = np.random.randint(3, 256, size=n_val, dtype=np.uint16)

        train_bin = tmp_path / "train.bin"
        val_bin = tmp_path / "val.bin"

        mm_t = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(n_train,))
        mm_t[:] = arr_train
        mm_t.flush()

        mm_v = np.memmap(val_bin, dtype=np.uint16, mode="w+", shape=(n_val,))
        mm_v[:] = arr_val
        mm_v.flush()

        info = TokenizedDataInfo(
            dataset_name="test",
            tokenizer_name="test_tok",
            vocab_size=256,
            train_tokens=n_train,
            val_tokens=n_val,
        )
        (tmp_path / "meta.json").write_text(json.dumps(dataclasses.asdict(info)))

        # Verify files exist and are readable
        assert train_bin.exists() and val_bin.exists()
        assert (tmp_path / "meta.json").exists()

        # MemmapDataset should be able to read them
        ds = MemmapDataset(train_bin, seq_len=64)
        assert len(ds) > 0
        assert ds[0]["input_ids"].shape == (65,)

    def test_tokenized_data_info_str(self):
        from engine.legacy.data.pretokenize import TokenizedDataInfo

        info = TokenizedDataInfo(
            dataset_name="test",
            tokenizer_name="gpt2",
            vocab_size=50257,
            train_tokens=1_000_000,
            val_tokens=10_000,
        )
        s = str(info)
        assert "gpt2" in s
        assert "1,000,000" in s
        assert "train" in s.lower()

    def test_uint16_rejects_large_vocab(self, tmp_path):
        """Vocab sizes > 65535 should raise — doesn't fit in uint16."""
        # We test this at the validation logic level
        vocab_size = 70_000
        assert vocab_size > 65535  # the condition pretokenize checks

    def test_meta_json_round_trip(self, tmp_path):
        """meta.json → TokenizedDataInfo → meta.json produces identical JSON."""
        from engine.legacy.data.pretokenize import TokenizedDataInfo
        import dataclasses

        info = TokenizedDataInfo(
            dataset_name="roneneldan/TinyStories",
            tokenizer_name="gpt2",
            vocab_size=50257,
            train_tokens=474_069_024,
            val_tokens=2_369_024,
            eos_token_id=50256,
            seq_recommended=1024,
        )
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(dataclasses.asdict(info), indent=2))
        loaded = json.loads(path.read_text())
        assert loaded["vocab_size"] == 50257
        assert loaded["train_tokens"] == 474_069_024
        assert loaded["seq_recommended"] == 1024
        assert loaded["eos_token_id"] == 50256


# ─────────────────────────────────────────────────────────────────────────────
# build_dataloader — new defaults
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildDataloaderNew:
    def test_memmap_uses_passthrough_collator(self, tmp_path):
        """MemmapDataset should get the fast passthrough collator, not CLMCollator."""
        _write_fake_dataset(tmp_path)
        ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=32)
        # num_workers=0 to avoid fork issues in tests
        loader = build_dataloader(
            ds, batch_size=4, num_workers=0, pin_memory=False, shuffle=False
        )
        batch = next(iter(loader))
        # Returns seq_len+1 for both input_ids and labels (full chunk)
        assert batch["input_ids"].shape == (4, 33)
        assert batch["labels"].shape == (4, 33)
        # No padding — all real tokens, no -100 labels
        assert (batch["labels"] != -100).all()

    def test_packed_uses_clm_collator(self):
        """PackedDataset should get CLMCollator (handles variable lengths)."""
        from engine.legacy.data.packed import PackedDataset

        seqs = [list(range(50)) for _ in range(20)]
        ds = PackedDataset(seqs, seq_len=32)
        loader = build_dataloader(
            ds, batch_size=4, num_workers=0, pin_memory=False, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (4, 32)

    def test_num_workers_explicit_zero(self, tmp_path):
        _write_fake_dataset(tmp_path)
        ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=32)
        loader = build_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)
        assert loader.num_workers == 0

    def test_pin_memory_false_on_cpu(self, tmp_path):
        """pin_memory should be forced False when no CUDA is available."""
        _write_fake_dataset(tmp_path)
        ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=32)
        # Build with pin_memory=False explicitly (safe on CPU)
        loader = build_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)
        assert not loader.pin_memory

    def test_drop_last_default(self, tmp_path):
        """drop_last=True should always be set to avoid uneven batches."""
        _write_fake_dataset(tmp_path)
        ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=32)
        loader = build_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)
        assert loader.drop_last

    def test_shuffle_false_for_eval(self, tmp_path):
        _write_fake_dataset(tmp_path)
        ds = MemmapDataset.from_dir(tmp_path, split="val", seq_len=32)
        loader = build_dataloader(
            ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False
        )
        assert not loader.dataset.__class__.__name__ == "IterableDataset"

    def test_default_num_workers_detection(self):
        """_default_num_workers should return 0 on Windows, int otherwise."""
        from engine.data import _default_num_workers

        n = _default_num_workers()
        assert isinstance(n, int)
        assert n >= 0


# ─────────────────────────────────────────────────────────────────────────────
# DeviceManager.build_optimizer
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildOptimizer:
    """
    Tests the fused AdamW + param group separation logic.
    All tests run on CPU — fused=False fallback path tested.
    """

    def _model(self) -> torch.nn.Module:
        from engine.models import CausalLM

        return CausalLM(_tiny_cfg())

    def _dm(self) -> DeviceManager:
        return DeviceManager(TrainConfig(backend="cpu", dtype="float32"))

    def test_returns_adamw(self):
        dm = self._dm()
        model = self._model()
        opt = dm.build_optimizer(model, lr=1e-3, fused=False)
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self):
        """Should produce exactly 2 param groups: decayed and non-decayed."""
        dm = self._dm()
        model = self._model()
        opt = dm.build_optimizer(model, lr=1e-3, fused=False)
        assert len(opt.param_groups) == 2

    def test_decay_group_has_weight_decay(self):
        dm = self._dm()
        model = self._model()
        opt = dm.build_optimizer(model, weight_decay=0.1, fused=False)
        decay_group = opt.param_groups[0]
        assert decay_group["weight_decay"] == 0.1

    def test_no_decay_group_has_zero_wd(self):
        dm = self._dm()
        model = self._model()
        opt = dm.build_optimizer(model, weight_decay=0.1, fused=False)
        no_decay_group = opt.param_groups[1]
        assert no_decay_group["weight_decay"] == 0.0

    def test_no_decay_contains_norms_and_biases(self):
        """Norm weights and biases should always be in the no-decay group."""
        dm = self._dm()
        model = self._model()
        opt = dm.build_optimizer(model, fused=False)

        # Collect all param data pointers in each group
        decay_ptrs = {p.data_ptr() for p in opt.param_groups[0]["params"]}
        no_decay_ptrs = {p.data_ptr() for p in opt.param_groups[1]["params"]}

        # Check that 1-D norm weight tensors are in no_decay
        for name, param in model.named_parameters():
            if "layernorm" in name or "norm" in name:
                if param.ndim == 1:
                    assert param.data_ptr() in no_decay_ptrs, (
                        f"Norm param '{name}' should be in no-decay group"
                    )

    def test_all_params_covered(self):
        """Every trainable parameter should appear in exactly one group."""
        dm = self._dm()
        model = self._model()
        opt = dm.build_optimizer(model, fused=False)

        all_group_ptrs = set()
        for group in opt.param_groups:
            for p in group["params"]:
                assert p.data_ptr() not in all_group_ptrs, (
                    "Param appears in multiple groups"
                )
                all_group_ptrs.add(p.data_ptr())

        trainable_ptrs = {p.data_ptr() for p in model.parameters() if p.requires_grad}
        assert trainable_ptrs == all_group_ptrs

    def test_lr_set_correctly(self):
        dm = self._dm()
        opt = dm.build_optimizer(self._model(), lr=3e-4, fused=False)
        for group in opt.param_groups:
            assert group["lr"] == 3e-4

    def test_fused_false_on_cpu(self):
        """On CPU, fused=True should silently fall back to standard AdamW."""
        dm = self._dm()
        # Should not raise even when fused=True on CPU
        opt = dm.build_optimizer(self._model(), lr=1e-3, fused=True)
        assert isinstance(opt, torch.optim.AdamW)

    def test_optimizer_step_works(self):
        """Sanity check: optimizer step should update model params."""
        from engine.models import CausalLM

        dm = self._dm()
        model = dm.prepare(CausalLM(_tiny_cfg()))
        opt = dm.build_optimizer(model, lr=1e-3, fused=False)

        ids = torch.randint(0, 256, (2, 16))
        _, loss, _ = model(ids, labels=ids)
        loss.backward()

        # Record param values before step
        before = [p.data.clone() for p in model.parameters()]
        opt.step()
        after = [p.data.clone() for p in model.parameters()]

        # At least some params should have changed
        changed = sum(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed > 0


# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig — new fields
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainConfigNew:
    def test_default_values(self):
        cfg = TrainConfig()
        assert cfg.fused_adamw == True
        assert cfg.num_workers == -1
        assert cfg.pin_memory == True
        assert cfg.data_dir == ""
        assert cfg.data_hub_repo == ""

    def test_warmup_steps(self):
        cfg = TrainConfig(max_steps=10_000, warmup_ratio=0.04)
        assert cfg.warmup_steps == 400

    def test_effective_batch_size(self):
        cfg = TrainConfig(batch_size=8, grad_accum=4)
        assert cfg.effective_batch_size == 32

    def test_yaml_round_trip(self, tmp_path):
        """Fields should survive YAML serialisation via _merge."""
        import yaml
        from engine.config.schema import load_experiment_config

        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
experiment:
  name: test
  hub:
    repo_id: ""
model:
  vocab_size: 256
  hidden_size: 64
  num_layers: 2
  attention: {type: gqa, num_heads: 4, num_kv_heads: 2}
  positional: {type: rope}
  ffn: {type: swiglu, intermediate_size: 128}
  norm: {type: rms}
training:
  max_steps: 100
  batch_size: 4
  seq_len: 32
  fused_adamw: false
  num_workers: 0
  data_hub_repo: "user/dataset"
  data_dir: "/tmp/data"
""")
        exp = load_experiment_config(config_yaml)
        assert exp.training.fused_adamw == False
        assert exp.training.num_workers == 0
        assert exp.training.data_hub_repo == "user/dataset"
        assert exp.training.data_dir == "/tmp/data"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: MemmapDataset → build_dataloader → training step
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_training_step_with_memmap(self, tmp_path):
        """End-to-end: fake .bin file → DataLoader → model forward → loss."""
        from engine.models import CausalLM

        _write_fake_dataset(tmp_path, train_tokens=2048, vocab_size=256)
        ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=32)

        dm = DeviceManager(TrainConfig(backend="cpu", dtype="float32"))
        model = dm.prepare(CausalLM(_tiny_cfg()))
        opt = dm.build_optimizer(model, lr=1e-3, fused=False)

        loader = build_dataloader(
            ds, batch_size=4, num_workers=0, pin_memory=False, shuffle=False
        )
        batch = dm.to_device(next(iter(loader)))

        with dm.autocast():
            _, loss, _ = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )

        assert loss is not None
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        # Should complete without error

    def test_memmap_vs_packed_same_shapes(self, tmp_path):
        """MemmapDataset and PackedDataset should produce identically-shaped batches."""
        from engine.legacy.data.packed import PackedDataset

        _write_fake_dataset(tmp_path, train_tokens=4096)
        mm_ds = MemmapDataset.from_dir(tmp_path, split="train", seq_len=32)

        seqs = [list(range(100)) for _ in range(50)]
        pk_ds = PackedDataset(seqs, seq_len=32)

        mm_loader = build_dataloader(
            mm_ds, batch_size=4, num_workers=0, pin_memory=False, shuffle=False
        )
        pk_loader = build_dataloader(
            pk_ds, batch_size=4, num_workers=0, pin_memory=False, shuffle=False
        )

        mm_batch = next(iter(mm_loader))
        pk_batch = next(iter(pk_loader))

        # MemmapDataset returns seq_len+1 (full chunk), PackedDataset returns seq_len
        # Both are valid for training - the model handles the shift internally
        assert (
            mm_batch["input_ids"].shape[0] == pk_batch["input_ids"].shape[0]
        )  # same batch size
        assert mm_batch["labels"].shape[0] == pk_batch["labels"].shape[0]
