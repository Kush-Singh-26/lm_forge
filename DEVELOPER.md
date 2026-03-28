# Developer Guide — lm_forge

Welcome to the internal engineering guide for `lm_forge`. This document explains the architecture, the registry system, and how to extend the framework.

## 1. Core Architecture

`lm_forge` is built on a **Registry-based Component System**. Instead of hardcoding attention or MLP implementations inside the model, we use factories to build them based on configuration.

### Key Components:
- **`engine/config/schema.py`**: The "Source of Truth" for all parameters. Uses nested dataclasses that map 1:1 to YAML keys.
- **`engine/components/`**: Modular implementations of specific layers (Attention, Norm, FFN, Positional).
- **`engine/models/`**: The high-level model structures (`DecoderModel`, `CausalLM`, `MaskedLM`).
- **`engine/data/`**: Streaming and memory-mapped data pipelines.

## 2. The Registry System

Every component (e.g., a new attention mechanism) is registered using a decorator.

```python
# engine/components/attention/my_new_attn.py

@register("my_new_attn")
class MyNewAttention(BaseAttention):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # ... implementation ...
```

To use it, simply update your YAML config:
```yaml
model:
  attention:
    type: "my_new_attn"
```

The `build_attention(cfg)` factory will automatically find and instantiate your class.

## 3. High-Performance Optimizations

We use several strategies to maximize GPU utilization:
- **Fused Projections**: QKV projections are fused into a single `nn.Linear` to reduce kernel launches.
- **PyTorch Native SDPA**: Automatically selects the fastest available kernel (Flash Attention 2, Memory-Efficient, or math) for GQA, MQA, and Sliding Window.
- **Vectorized Collators**: Data masking and padding use vectorized PyTorch operations instead of Python loops.
- **Zero-Copy Data Prep**: Uses NumPy memory-mapping and view-based reshaping to avoid unnecessary copies.

## 4. Evaluation Suite

Standardized evaluation is built into the CLI:
```bash
# Calculate Perplexity
python main.py eval --model checkpoints/my_model --dataset wikitext --split validation

# Run Zero-Shot Benchmarks
python main.py eval --model checkpoints/my_model --tasks hellaswag,arc_easy
```

## 5. Distributed Training

For training models >500M parameters, use `accelerate` with our provided FSDP config:
```bash
accelerate launch --config_file configs/accelerate/fsdp.yaml experiments/pretrain_base/train.py --config ...
```

## 6. Development Workflow

1. **Add a component**: Register it in the appropriate `engine/components/` subdirectory.
2. **Update Schema**: If your component needs new parameters, add them to `AttentionConfig` or `FFNConfig` in `engine/config/schema.py`.
3. **Write a test**: Add a functional test in `tests/test_components.py`.
4. **Benchmark**: Use `python main.py profile` to ensure no performance regression.
