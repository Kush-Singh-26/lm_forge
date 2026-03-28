"""
lm_forge engine — modular small LM training.

HF Native primary entry point.
"""

from engine.config import (
    load_experiment_config,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    HubConfig,
    AttentionConfig,
    PositionalConfig,
    FFNConfig,
    NormConfig,
    LMForgeConfig,
)
from engine.models import (
    CausalLM,
    MaskedLM,
    BaseLM,
    DecoderLayer,
    EncoderLayer,
    HFCausalLM,
)
from engine.components import (
    build_norm,
    build_pe,
    build_attention,
    build_ffn,
    list_pe_types,
    list_attention_types,
    list_ffn_types,
)
from engine.data import (
    CLMCollator,
    MLMCollator,
    prepare_dataset,
    SyntheticDataset,
    build_streaming_dataset,
    build_fineweb_edu,
    build_stack_v2,
)
from engine.tokenizer import BPETokenizer, HFBPETokenizer
from engine.utils import (
    Profiler,
    AblationRunner,
    generate_model_card,
    ProfilingCallback,
)

__version__ = "1.0.0"
