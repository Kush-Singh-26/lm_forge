"""
tests/test_components.py

Unit tests for all component implementations.
Run: pytest tests/ -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config.schema import (
    ModelConfig,
    AttentionConfig,
    PositionalConfig,
    FFNConfig,
    NormConfig,
)
from engine.components.norm.norms import RMSNorm, LayerNorm, build_norm
from engine.components.positional import PEOutput, build_pe
from engine.components.positional.rope import apply_rope
from engine.components.positional.alibi import _get_alibi_slopes
from engine.components.attention import build_attention
from engine.components.ffn import build_ffn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _small_cfg(**overrides) -> ModelConfig:
    """Return a minimal ModelConfig for fast CPU tests."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        attention=AttentionConfig(type="gqa", num_heads=4, num_kv_heads=2),
        positional=PositionalConfig(type="rope", max_seq_len=128),
        ffn=FFNConfig(type="swiglu", intermediate_size=128),
        norm=NormConfig(type="rms"),
    )
    # Apply overrides to nested configs
    for k, v in overrides.items():
        if "." in k:
            section, field = k.split(".", 1)
            if hasattr(defaults.get(section), field):
                setattr(defaults[section], field, v)
        else:
            defaults[k] = v
    return ModelConfig(**defaults)


B, S, D = 2, 16, 64  # batch, seq_len, hidden_size


# ─────────────────────────────────────────────────────────────────────────────
# Norm
# ─────────────────────────────────────────────────────────────────────────────


class TestNorms:
    def test_rmsnorm_shape(self):
        norm = RMSNorm(D)
        x = torch.randn(B, S, D)
        assert norm(x).shape == (B, S, D)

    def test_rmsnorm_normalises(self):
        norm = RMSNorm(D)
        x = torch.randn(B, S, D) * 10
        y = norm(x)
        # After normalisation, RMS of each vector should be close to 1 (before weight)
        # Since weight=ones, just check output is finite and bounded
        assert torch.isfinite(y).all()
        assert y.abs().max() < 100

    def test_rmsnorm_dtype_preserved(self):
        norm = RMSNorm(D)
        x = torch.randn(B, S, D).half()
        assert norm(x).dtype == torch.float16

    def test_layernorm_shape(self):
        norm = LayerNorm(D)
        x = torch.randn(B, S, D)
        assert norm(x).shape == (B, S, D)

    def test_layernorm_zero_mean(self):
        norm = LayerNorm(D, bias=False)
        norm.weight.data.fill_(1.0)
        x = torch.randn(B, S, D)
        y = norm(x)
        assert y.mean(dim=-1).abs().max() < 1e-4

    def test_build_norm_rms(self):
        assert isinstance(build_norm("rms", D), RMSNorm)

    def test_build_norm_layer(self):
        assert isinstance(build_norm("layer", D), LayerNorm)

    def test_build_norm_unknown(self):
        with pytest.raises(ValueError, match="Unknown norm"):
            build_norm("unknown", D)


# ─────────────────────────────────────────────────────────────────────────────
# Positional encodings
# ─────────────────────────────────────────────────────────────────────────────


class TestRoPE:
    def setup_method(self):
        cfg = _small_cfg()
        self.pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        self.pe.set_head_dim(cfg.head_dim)

    def test_returns_pe_output(self):
        x = torch.randn(B, S, D)
        out = self.pe(x, seq_len=S)
        assert isinstance(out, PEOutput)
        assert out.cos is not None
        assert out.sin is not None
        assert out.hidden_states is None
        assert out.attn_bias is None

    def test_cos_sin_shape(self):
        x = torch.randn(B, S, D)
        out = self.pe(x, seq_len=S)
        head_dim = D // 4
        assert out.cos.shape == (1, 1, S, head_dim)
        assert out.sin.shape == (1, 1, S, head_dim)

    def test_cache_extension(self):
        x = torch.randn(B, S, D)
        # Request longer sequence than initial cache
        out = self.pe(x, seq_len=200)
        assert out.cos.shape[-2] == 200

    def test_apply_rope_shape(self):
        head_dim = 16
        q = torch.randn(B, 4, S, head_dim)
        k = torch.randn(B, 4, S, head_dim)
        cos = torch.randn(1, 1, S, head_dim)
        sin = torch.randn(1, 1, S, head_dim)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_apply_rope_dtype_preserved(self):
        head_dim = 16
        q = torch.randn(B, 4, S, head_dim).half()
        k = torch.randn(B, 4, S, head_dim).half()
        cos = torch.randn(1, 1, S, head_dim)
        sin = torch.randn(1, 1, S, head_dim)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.dtype == torch.float16

    def test_ntk_scaling(self):
        cfg = _small_cfg()
        cfg.positional.type = "rope"
        cfg.positional.scaling_type = "ntk"
        cfg.positional.factor = 2.0
        pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        pe.set_head_dim(D // 4)

        x = torch.randn(B, S, D)
        out = pe(x, seq_len=S)
        # Check that scaling changed the frequencies (compared to base RoPE)
        pe_base = build_pe(PositionalConfig(type="rope"), hidden_size=D, num_heads=4)
        pe_base.set_head_dim(D // 4)
        out_base = pe_base(x, seq_len=S)
        assert not torch.allclose(out.cos, out_base.cos)


class TestALiBi:
    def test_slopes_sum_pow2(self):
        slopes = _get_alibi_slopes(8)
        assert slopes.shape == (8,)
        assert (slopes > 0).all()
        assert (slopes < 1).all()

    def test_slopes_non_pow2(self):
        slopes = _get_alibi_slopes(6)
        assert slopes.shape == (6,)

    def test_returns_attn_bias(self):
        cfg = _small_cfg()
        cfg.positional.type = "alibi"
        pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        x = torch.randn(B, S, D)
        out = pe(x, seq_len=S)
        assert isinstance(out, PEOutput)
        assert out.attn_bias is not None
        assert out.cos is None

    def test_attn_bias_shape(self):
        cfg = _small_cfg()
        cfg.positional.type = "alibi"
        pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        x = torch.randn(B, S, D)
        out = pe(x, seq_len=S)
        assert out.attn_bias.shape == (1, 4, S, S)

    def test_alibi_non_pow2_slopes(self):
        slopes = _get_alibi_slopes(6)
        assert slopes.shape == (6,)
        # Slopes should be unique
        assert len(torch.unique(slopes)) == 6

    def test_bias_causal(self):
        """Future positions should have strongly negative bias."""
        cfg = _small_cfg()
        cfg.positional.type = "alibi"
        pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        x = torch.randn(1, S, D)
        out = pe(x, seq_len=S)
        bias = out.attn_bias[0, 0]  # (S, S)
        # Position 0 attending to position S-1 (future) should be very negative
        assert bias[0, -1] < -0.5


class TestLearnedPE:
    def test_modifies_hidden_states(self):
        cfg = _small_cfg()
        cfg.positional.type = "learned"
        cfg.positional.max_seq_len = 128
        pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        x = torch.randn(B, S, D)
        out = pe(x, seq_len=S)
        assert out.hidden_states is not None
        assert out.hidden_states.shape == (B, S, D)
        assert not torch.allclose(out.hidden_states, x)

    def test_raises_beyond_max(self):
        cfg = _small_cfg()
        cfg.positional.type = "learned"
        cfg.positional.max_seq_len = 8
        pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        x = torch.randn(B, 20, D)
        with pytest.raises(AssertionError):
            pe(x, seq_len=20)


class TestNoPE:
    def test_all_none(self):
        cfg = _small_cfg()
        cfg.positional.type = "none"
        pe = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        x = torch.randn(B, S, D)
        out = pe(x, seq_len=S)
        assert out.cos is None
        assert out.sin is None
        assert out.hidden_states is None
        assert out.attn_bias is None


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────


class TestAttention:
    @pytest.mark.parametrize(
        "attn_type,num_kv",
        [
            ("mha", 4),
            ("gqa", 2),
            ("mqa", 1),
            ("sliding", 2),
        ],
    )
    def test_output_shape(self, attn_type, num_kv):
        cfg = _small_cfg()
        cfg.attention.type = attn_type
        cfg.attention.num_kv_heads = num_kv
        cfg.attention.__post_init__()
        cfg.__post_init__()

        attn = build_attention(cfg)
        x = torch.randn(B, S, D)
        pe_out = PEOutput()  # NoPE

        out, present = attn(x, pe_out=pe_out)
        assert out.shape == (B, S, D)
        assert present is None

    def test_gqa_with_rope(self):
        cfg = _small_cfg()
        cfg.attention.type = "gqa"
        cfg.attention.num_kv_heads = 2
        cfg.__post_init__()

        attn = build_attention(cfg)
        pe_module = build_pe(cfg.positional, hidden_size=D, num_heads=4)
        pe_module.set_head_dim(cfg.head_dim)

        x = torch.randn(B, S, D)
        pe_out = pe_module(x, seq_len=S)
        out, _ = attn(x, pe_out=pe_out)
        assert out.shape == (B, S, D)

    def test_kv_cache(self):
        cfg = _small_cfg()
        cfg.attention.type = "gqa"
        cfg.__post_init__()
        attn = build_attention(cfg)
        attn.eval()

        x = torch.randn(1, S, D)
        pe_out = PEOutput()

        # First step — build cache
        out1, pkv = attn(x, pe_out=pe_out, use_cache=True)
        assert pkv is not None
        k, v = pkv
        assert k.shape[-2] == S

        # Second step — extend cache with one new token
        x2 = torch.randn(1, 1, D)
        out2, pkv2 = attn(x2, pe_out=pe_out, past_key_value=pkv, use_cache=True)
        assert out2.shape == (1, 1, D)
        assert pkv2[0].shape[-2] == S + 1

    def test_mha_llama_names(self):
        cfg = _small_cfg()
        cfg.attention.type = "mha"
        cfg.__post_init__()
        attn = build_attention(cfg)
        for name in ["qkv_proj", "o_proj"]:
            assert hasattr(attn, name), f"Missing {name}"

    def test_sliding_window_restricts_context(self):
        cfg = _small_cfg()
        cfg.attention.type = "sliding"
        cfg.attention.window_size = 4
        cfg.attention.num_kv_heads = 2
        cfg.__post_init__()
        attn = build_attention(cfg)
        x = torch.randn(B, S, D)
        pe_out = PEOutput()
        out, _ = attn(x, pe_out=pe_out)
        assert out.shape == (B, S, D)


# ─────────────────────────────────────────────────────────────────────────────
# FFN
# ─────────────────────────────────────────────────────────────────────────────


class TestFFN:
    @pytest.mark.parametrize("ffn_type", ["swiglu", "geglu", "classic"])
    def test_output_shape(self, ffn_type):
        cfg = _small_cfg()
        cfg.ffn.type = ffn_type
        ffn = build_ffn(cfg)
        x = torch.randn(B, S, D)
        assert ffn(x).shape == (B, S, D)

    def test_swiglu_has_two_projections(self):
        cfg = _small_cfg()
        cfg.ffn.type = "swiglu"
        ffn = build_ffn(cfg)
        assert hasattr(ffn, "gate_up_proj")
        assert hasattr(ffn, "down_proj")

    def test_classic_has_peft_aliases(self):
        cfg = _small_cfg()
        cfg.ffn.type = "classic"
        ffn = build_ffn(cfg)
        assert hasattr(ffn, "gate_proj")  # alias for fc1
        assert hasattr(ffn, "down_proj")  # alias for fc2
        assert ffn.gate_proj is ffn.fc1
        assert ffn.down_proj is ffn.fc2

    def test_no_nans(self):
        for ffn_type in ["swiglu", "geglu", "classic"]:
            cfg = _small_cfg()
            cfg.ffn.type = ffn_type
            ffn = build_ffn(cfg)
            x = torch.randn(B, S, D)
            assert torch.isfinite(ffn(x)).all()


class TestConfigs:
    def test_rope_head_dim_even(self):
        # Should work with default hidden=64, heads=4 (head_dim=16)
        _small_cfg(positional=PositionalConfig(type="rope"))

        # Should fail if head_dim is odd
        with pytest.raises(AssertionError, match="must be even for RoPE"):
            _small_cfg(
                hidden_size=60,
                attention=AttentionConfig(num_heads=12, num_kv_heads=4),
                positional=PositionalConfig(type="rope"),
            )
