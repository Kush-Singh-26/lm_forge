"""
engine/config/schema.py

Nested, YAML-loadable config.  Every field maps directly to a config.yaml key.
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
    # RoPE Scaling (NTK-aware)
    scaling_type: Optional[str] = None  # None | "ntk"
    factor: float = 1.0
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

    # ── tokens ───────────────────────────────────────────────────────────
    # Token IDs (defaults to SmolLM tokenizer)
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0

    # ── derived (auto-computed, not set in YAML) ─────────────────────────
    head_dim: int = 0

    def __post_init__(self):
        # Cascade max_seq_len into positional config if it hasn't been explicitly changed
        # or if the model's max_seq_len was explicitly changed but positional wasn't.
        if self.positional.max_seq_len == 2048 and self.max_seq_len != 2048:
            self.positional.max_seq_len = self.max_seq_len

        # Ensure hidden_size is divisible by num_heads
        if self.attention.num_heads > 0:
            assert self.hidden_size % self.attention.num_heads == 0, (
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.attention.num_heads})"
            )
            self.head_dim = self.hidden_size // self.attention.num_heads

            if self.positional.type == "rope":
                assert self.head_dim % 2 == 0, (
                    f"head_dim ({self.head_dim}) must be even for RoPE. "
                    f"Check hidden_size ({self.hidden_size}) and num_heads ({self.attention.num_heads})."
                )

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
    num_workers: int = -1  # -1 = auto-detect
    pin_memory: bool = True

    # ── Optimizer ────────────────────────────────────────────────────────────
    fused_adamw: bool = True

    # ── HF Native ────────────────────────────────────────────────────────────
    hf_args: dict[str, Any] = field(default_factory=dict)

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir: str = ""
    data_hub_repo: str = ""
    shuffle_buffer: int = 10000  # Size of the streaming shuffle buffer
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"

    def __post_init__(self):
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)
        if self.num_workers == -1:
            import os

            self.num_workers = os.cpu_count() or 0

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.max_steps * self.warmup_ratio))

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass
class HubConfig:
    repo_id: str = ""  # "username/repo-name" — empty = no push
    use_hub_checkpoints: bool = True
    checkpoint_limit: int = 2
    push_every: int = 500
    private: bool = True
    token_env: str = "HF_TOKEN"


@dataclass
class LoggingConfig:
    report_to: list[str] = field(
        default_factory=lambda: ["none"]
    )  # ["wandb", "tensorboard"]
    wandb_project: str = "lm_forge"
    wandb_entity: Optional[str] = None
    log_model: str = "checkpoint"  # "checkpoint" | "end" | "none"


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        m = self.model
        t = self.training
        if m.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if m.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if t.lr <= 0:
            raise ValueError("lr must be > 0")
        if t.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if t.seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if t.seq_len > m.max_seq_len:
            raise ValueError(
                f"training.seq_len ({t.seq_len}) exceeds model.max_seq_len "
                f"({m.max_seq_len}). Increase max_seq_len or reduce seq_len."
            )

        if m.attention.flash_attn and m.positional.type == "alibi":
            raise ValueError(
                "Flash Attention 2 does not natively support additive bias (ALiBi). "
                "Disable flash_attn or use RoPE instead."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def _merge(dataclass_instance, overrides: dict, prefix: str = "") -> list[str]:
    """
    Recursively merge overrides into a dataclass.
    Returns a list of unknown keys found.
    """
    fields = {f.name: f for f in dataclass_instance.__dataclass_fields__.values()}
    unknown_keys = []
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in fields:
            unknown_keys.append(full_key)
            continue
        current = getattr(dataclass_instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            unknown_keys.extend(_merge(current, value, prefix=full_key))
        else:
            setattr(dataclass_instance, key, value)
    if hasattr(dataclass_instance, "__post_init__"):
        dataclass_instance.__post_init__()
    return unknown_keys


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    cfg = ExperimentConfig()
    unknown = []
    if "experiment" in raw:
        exp_raw = dict(raw["experiment"])
        cfg.name = exp_raw.pop("name", cfg.name)
        if "hub" in exp_raw:
            unknown.extend(_merge(cfg.hub, exp_raw["hub"], prefix="experiment.hub"))
    if "model" in raw:
        unknown.extend(_merge(cfg.model, raw["model"], prefix="model"))
    if "training" in raw:
        unknown.extend(_merge(cfg.training, raw["training"], prefix="training"))
    if "hub" in raw:
        unknown.extend(_merge(cfg.hub, raw["hub"], prefix="hub"))
    if "logging" in raw:
        unknown.extend(_merge(cfg.logging, raw["logging"], prefix="logging"))

    if unknown:
        import warnings

        warnings.warn(
            f"Found unknown configuration keys in YAML: {', '.join(unknown)}. "
            "These will be ignored. Check for typos!"
        )

    cfg.validate()
    return cfg
