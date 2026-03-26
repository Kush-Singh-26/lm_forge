"""
engine/config/schema.py

Nested, YAML-loadable config.  Every field maps directly to a config.yaml key.
Nested sections (attention, positional, ffn, norm) each carry a `type` field
that the component registries use to pick the right implementation.

Example config.yaml structure:
    experiment:
      name: "exp_001"
      hub:
        repo_id: "your-name/lm-forge-exp001"
        push_every: 500
    model:
      vocab_size: 32000
      hidden_size: 512
      num_layers: 6
      attention:
        type: "gqa"
        num_heads: 8
        num_kv_heads: 2
      positional:
        type: "rope"
        theta: 10000.0
      ffn:
        type: "swiglu"
        intermediate_size: 1376
      norm:
        type: "rms"
        eps: 1e-5
    training:
      max_steps: 10000
      batch_size: 8
      grad_accum: 4
      lr: 3e-4
"""

from __future__ import annotations

import json
import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs  (each maps to a YAML section)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AttentionConfig:
    type: str = "gqa"  # "mha" | "gqa" | "mqa" | "sliding"
    num_heads: int = 8
    num_kv_heads: int = 8  # < num_heads → GQA; 1 → MQA
    dropout: float = 0.0
    # sliding window only
    window_size: int = 512
    # Flash Attention 2 (requires: pip install flash-attn --no-build-isolation)
    # Only supported on CUDA with fp16/bf16.  Set dtype accordingly.
    flash_attn: bool = False

    def __post_init__(self):
        if self.type == "mqa":
            self.num_kv_heads = 1
        if self.type == "mha":
            self.num_kv_heads = self.num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )


@dataclass
class PositionalConfig:
    type: str = "rope"  # "rope" | "alibi" | "learned" | "none"
    # RoPE
    theta: float = 10_000.0
    # Learned absolute — matches max_seq_len of model
    max_seq_len: int = 2048


@dataclass
class FFNConfig:
    type: str = "swiglu"  # "swiglu" | "geglu" | "classic"
    intermediate_size: int = 1376
    dropout: float = 0.0
    bias: bool = False


@dataclass
class NormConfig:
    type: str = "rms"  # "rms" | "layer"
    eps: float = 1e-5


@dataclass
class ModelConfig:
    # ── shape ────────────────────────────────────────────────────────────
    vocab_size: int = 32_000
    hidden_size: int = 512
    num_layers: int = 6
    max_seq_len: int = 2048

    # ── sub-components (override via YAML) ───────────────────────────────
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    positional: PositionalConfig = field(default_factory=PositionalConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    norm: NormConfig = field(default_factory=NormConfig)

    # ── misc ─────────────────────────────────────────────────────────────
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # ── derived (auto-computed, not set in YAML) ─────────────────────────
    head_dim: int = 0

    def __post_init__(self):
        # Cascade max_seq_len into positional config if not explicitly set
        if self.positional.max_seq_len == 2048 and self.max_seq_len != 2048:
            self.positional.max_seq_len = self.max_seq_len
        assert self.hidden_size % self.attention.num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )
        self.head_dim = self.hidden_size // self.attention.num_heads

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_config.json").write_text(self.to_json())


@dataclass
class TrainConfig:
    max_steps: int = 10_000
    batch_size: int = 8
    seq_len: int = 512
    grad_accum: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_ratio: float = 0.04  # fraction of max_steps used for warmup
    max_grad_norm: float = 1.0
    dtype: str = "float32"  # "float32" | "float16" | "bfloat16"
    backend: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500
    compile: bool = False

    # ── DataLoader ───────────────────────────────────────────────────────────
    # None = auto-detect (0 on Windows, 2 in Colab, 4 elsewhere)
    num_workers: int = -1  # -1 = auto-detect
    pin_memory: bool = True  # auto-disabled on CPU in DeviceManager

    # ── Optimizer ────────────────────────────────────────────────────────────
    # fused AdamW: ~5-10% faster on CUDA, requires torch >= 2.0.
    # Automatically disabled on CPU/MPS where it's not supported.
    fused_adamw: bool = True

    # ── HF Native ────────────────────────────────────────────────────────────
    # Dictionary of keyword arguments passed directly to transformers.TrainingArguments
    # when using the HF trainer. e.g. { "bf16": True, "optim": "adamw_torch_fused" }
    hf_args: dict[str, Any] = field(default_factory=dict)

    # ── Data ─────────────────────────────────────────────────────────────────
    # Path to a pre-tokenized dataset directory (train.bin + val.bin + meta.json)
    # produced by engine/data/pretokenize.py.
    # Leave empty to use the experiment's own dataset code.
    data_dir: str = ""
    # HF Hub dataset repo to pull from if data_dir files are missing.
    data_hub_repo: str = ""

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.max_steps * self.warmup_ratio))

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass
class HubConfig:
    repo_id: str = ""  # "username/repo-name" — empty = no push
    push_every: int = 500  # push checkpoint every N opt steps
    private: bool = True
    token_env: str = "HF_TOKEN"  # env var name for the HF write token


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def _merge(dataclass_instance, overrides: dict) -> None:
    """
    Recursively apply a dict of overrides to a dataclass instance in-place.
    Unknown keys are silently ignored so future YAML fields don't break old
    engine versions.
    """
    fields = {f.name: f for f in dataclass_instance.__dataclass_fields__.values()}  # type: ignore[attr-defined]

    for key, value in overrides.items():
        if key not in fields:
            continue
        current = getattr(dataclass_instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dataclass_instance, key, value)

    # Re-run __post_init__ if present to recompute derived fields
    if hasattr(dataclass_instance, "__post_init__"):
        try:
            dataclass_instance.__post_init__()
        except Exception:
            pass  # best-effort


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML file.

    Missing keys fall back to dataclass defaults, so a minimal config.yaml
    only needs to specify the fields that differ from defaults.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}

    cfg = ExperimentConfig()
    if "experiment" in raw:
        exp_raw = dict(raw["experiment"])
        cfg.name = exp_raw.pop("name", cfg.name)
        if "hub" in exp_raw:
            _merge(cfg.hub, exp_raw["hub"])

    if "model" in raw:
        _merge(cfg.model, raw["model"])

    if "training" in raw:
        _merge(cfg.training, raw["training"])

    return cfg
