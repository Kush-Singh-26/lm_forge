"""
tests/conftest.py — shared pytest fixtures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from engine.config.schema import (
    ModelConfig, AttentionConfig, PositionalConfig, FFNConfig, NormConfig,
)
from engine.models import CausalLM


@pytest.fixture(scope="session")
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256, hidden_size=64, num_layers=2,
        attention=AttentionConfig(type="gqa", num_heads=4, num_kv_heads=2),
        positional=PositionalConfig(type="rope", max_seq_len=64),
        ffn=FFNConfig(type="swiglu", intermediate_size=128),
        norm=NormConfig(type="rms"),
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg) -> CausalLM:
    torch.manual_seed(0)
    return CausalLM(tiny_cfg)
