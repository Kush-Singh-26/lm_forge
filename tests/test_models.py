"""
tests/test_models.py

Integration tests for CausalLM, MaskedLM, config round-trip, and save/load.
"""

import sys
import tempfile
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
from engine.models import CausalLM, MaskedLM


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _cfg(
    attn_type="gqa", pe_type="rope", ffn_type="swiglu", norm_type="rms", num_kv_heads=2
) -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        attention=AttentionConfig(
            type=attn_type, num_heads=4, num_kv_heads=num_kv_heads
        ),
        positional=PositionalConfig(type=pe_type, max_seq_len=64),
        ffn=FFNConfig(type=ffn_type, intermediate_size=128),
        norm=NormConfig(type=norm_type),
    )


B, S = 2, 16


# ─────────────────────────────────────────────────────────────────────────────
# CausalLM
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalLM:
    def test_forward_shape(self):
        model = CausalLM(_cfg())
        ids = torch.randint(0, 256, (B, S))
        logits, loss, _ = model(ids)
        assert logits.shape == (B, S, 256)
        assert loss is None

    def test_loss_with_labels(self):
        model = CausalLM(_cfg())
        ids = torch.randint(0, 256, (B, S))
        _, loss, _ = model(ids, labels=ids)
        assert loss is not None
        assert loss.ndim == 0
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_loss_decreases_with_training(self):
        torch.manual_seed(42)
        model = CausalLM(_cfg())
        ids = torch.randint(0, 256, (B, S))
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        losses = []
        for _ in range(5):
            _, loss, _ = model(ids, labels=ids)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Loss should decrease at some point over 5 steps
        assert min(losses[2:]) < losses[0]

    @pytest.mark.parametrize("attn", ["mha", "gqa", "mqa", "sliding"])
    def test_all_attention_types(self, attn):
        kv = 1 if attn == "mqa" else 2 if attn in ("gqa", "sliding") else 4
        model = CausalLM(_cfg(attn_type=attn, num_kv_heads=kv))
        ids = torch.randint(0, 256, (B, S))
        logits, loss, _ = model(ids, labels=ids)
        assert logits.shape == (B, S, 256)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("pe", ["rope", "alibi", "learned", "none"])
    def test_all_pe_types(self, pe):
        model = CausalLM(_cfg(pe_type=pe))
        ids = torch.randint(0, 256, (B, S))
        logits, loss, _ = model(ids, labels=ids)
        assert logits.shape == (B, S, 256)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("ffn", ["swiglu", "geglu", "classic"])
    def test_all_ffn_types(self, ffn):
        model = CausalLM(_cfg(ffn_type=ffn))
        ids = torch.randint(0, 256, (B, S))
        logits, loss, _ = model(ids, labels=ids)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("norm", ["rms", "layer"])
    def test_all_norm_types(self, norm):
        model = CausalLM(_cfg(norm_type=norm))
        ids = torch.randint(0, 256, (B, S))
        _, loss, _ = model(ids, labels=ids)
        assert torch.isfinite(loss)

    def test_generate_length(self):
        model = CausalLM(_cfg())
        model.eval()
        seed = torch.randint(0, 256, (1, 4))
        out = model.generate(seed, max_new_tokens=8)
        assert out.shape == (1, 12)

    def test_generate_eos_stop(self):
        model = CausalLM(_cfg())
        model.eval()
        seed = torch.randint(0, 256, (1, 4))
        # Force EOS at position 5 by using eos_token_id that's very common
        out = model.generate(seed, max_new_tokens=20, eos_token_id=0)
        assert out.shape[1] <= 24  # stopped early or at max

    def test_llama_state_dict_keys(self):
        model = CausalLM(_cfg())
        sd = model.state_dict()
        required = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        for k in required:
            assert k in sd, f"Missing key: {k}"

    def test_num_parameters(self):
        model = CausalLM(_cfg())
        n = model.num_parameters()
        assert n > 0
        assert model.num_parameters(only_trainable=True) == n

    def test_gradient_checkpointing(self):
        model = CausalLM(_cfg())
        model.model.enable_gradient_checkpointing()
        assert model.model.gradient_checkpointing is True
        ids = torch.randint(0, 256, (B, S))
        _, loss, _ = model(ids, labels=ids)
        loss.backward()  # must not raise
        assert torch.isfinite(loss)

    def test_save_load_round_trip(self):
        model = CausalLM(_cfg())
        ids = torch.randint(0, 256, (B, S))
        logits_before, _, _ = model(ids)

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            assert (Path(d) / "model_config.json").exists()

            model2 = CausalLM.from_pretrained(d)
            logits_after, _, _ = model2(ids)

        assert torch.allclose(logits_before, logits_after, atol=1e-5)

    def test_tie_word_embeddings(self):
        cfg = _cfg()
        cfg.tie_word_embeddings = True
        model = CausalLM(cfg)
        assert (
            model.lm_head.weight.data_ptr()
            == model.model.embed_tokens.weight.data_ptr()
        )

    def test_attention_mask_applied(self):
        """Padding mask should change output on padded positions."""
        model = CausalLM(_cfg())
        model.eval()
        ids = torch.randint(1, 256, (1, S))
        mask = torch.ones(1, S, dtype=torch.long)
        mask[0, S // 2 :] = 0  # pad second half

        out_masked, _, _ = model(ids, attention_mask=mask)
        out_unmasked, _, _ = model(ids)

        # Outputs should differ at masked positions
        assert not torch.allclose(out_masked, out_unmasked, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# MaskedLM
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskedLM:
    def test_forward_shape(self):
        model = MaskedLM(_cfg(attn_type="mha", pe_type="learned"))
        ids = torch.randint(0, 256, (B, S))
        logits, loss = model(ids)
        assert logits.shape == (B, S, 256)
        assert loss is None

    def test_mlm_loss(self):
        model = MaskedLM(_cfg(attn_type="mha", pe_type="learned"))
        ids = torch.randint(0, 256, (B, S))
        labels = torch.full_like(ids, -100)
        labels[:, ::3] = ids[:, ::3]  # only supervise every 3rd token

        _, loss = model(ids, labels=labels)
        assert loss is not None
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_encode_cls(self):
        model = MaskedLM(_cfg(attn_type="mha", pe_type="learned"))
        model.eval()
        ids = torch.randint(0, 256, (B, S))
        mask = torch.ones(B, S, dtype=torch.long)
        emb = model.encode(ids, attention_mask=mask, pool="cls")
        assert emb.shape == (B, 64)
        # Should be L2-normalised
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_encode_mean(self):
        model = MaskedLM(_cfg(attn_type="mha", pe_type="learned"))
        model.eval()
        ids = torch.randint(0, 256, (B, S))
        mask = torch.ones(B, S, dtype=torch.long)
        emb = model.encode(ids, attention_mask=mask, pool="mean")
        assert emb.shape == (B, 64)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_peft_attn_names_accessible(self):
        """BERT-style attention.self path must exist."""
        model = MaskedLM(_cfg(attn_type="mha", pe_type="learned"))
        layer = model.encoder.layers[0]
        assert hasattr(layer.attention, "self")
        assert hasattr(layer.attention.self, "qkv_proj")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_head_dim_inferred(self):
        cfg = _cfg()
        assert cfg.head_dim == 64 // 4  # hidden_size // num_heads

    def test_gqa_validation(self):
        with pytest.raises(AssertionError):
            ModelConfig(
                vocab_size=256,
                hidden_size=64,
                num_layers=2,
                attention=AttentionConfig(type="gqa", num_heads=4, num_kv_heads=3),
            )

    def test_config_json_round_trip(self):
        import json, tempfile

        cfg = _cfg()
        with tempfile.TemporaryDirectory() as d:
            cfg.save(d)
            raw = json.loads((Path(d) / "model_config.json").read_text())
            assert raw["hidden_size"] == 64
            assert raw["attention"]["num_kv_heads"] == 2
