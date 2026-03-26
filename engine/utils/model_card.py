"""
engine/utils/model_card.py

Auto-generate a HuggingFace model card (README.md) from experiment config
and training results.

Produces a properly formatted Model Card with:
  • Model description (auto-generated from config)
  • Architecture table
  • Training details
  • Usage code snippet
  • Evaluation results table
  • Citation block
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from engine.config.schema import ExperimentConfig


_CARD_TEMPLATE = """\
---
license: mit
tags:
  - lm-forge
  - causal-lm
  - decoder-only
  - {attention_type}
  - {positional_type}
language:
  - en
library_name: lm_forge
---

# {model_name}

> Trained with [lm_forge](https://github.com/{github_user}/lm_forge) — a modular small LM training engine.

## Model description

A **{param_str} decoder-only causal language model** with the following architecture:

| Component | Choice | Detail |
|---|---|---|
| Attention | `{attention_type}` | {attn_detail} |
| Positional encoding | `{positional_type}` | {pe_detail} |
| Feed-forward | `{ffn_type}` | hidden={intermediate_size} |
| Normalisation | `{norm_type}` | eps={norm_eps} |
| Layers | {num_layers} | hidden_size={hidden_size} |
| Vocab size | {vocab_size} | |
| Max seq len | {max_seq_len} | |

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `{lr}` |
| LR schedule | Cosine with warmup |
| Warmup ratio | `{warmup_ratio}` |
| Weight decay | `{weight_decay}` |
| Batch size | {batch_size} × {grad_accum} = **{effective_batch}** tokens |
| Seq len | {seq_len} |
| Max steps | {max_steps} |
| dtype | `{dtype}` |
| Trained on | {train_date} |

{eval_section}

## Usage

```python
# Install
# pip install torch pyyaml huggingface_hub safetensors
# pip install transformers  # for HF Trainer / pipeline support

import sys
sys.path.insert(0, "path/to/lm_forge")

from engine.models import HFCausalLM

HFCausalLM.register()

# Load from Hub
model = HFCausalLM.from_pretrained("{hub_repo_id}")

# Inference
import torch
input_ids = torch.tensor([[2, 100, 200]])   # BOS + some tokens
output = model.generate(input_ids, max_new_tokens=64, temperature=0.8, top_k=50)
```

### PEFT / LoRA fine-tuning

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # LLaMA-compatible names
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
```

## Architecture notes

{arch_notes}

## Citation

```bibtex
@misc{{{citation_key},
  title  = {{{model_name}}},
  author = {{{author}}},
  year   = {{{year}}},
  url    = {{https://huggingface.co/{hub_repo_id}}}
}}
```
"""


_EVAL_SECTION_TEMPLATE = """\
## Evaluation results

| Metric | Value |
|---|---|
{eval_rows}
"""

_ARCH_NOTES = {
    "gqa": "Uses **Grouped Query Attention** (GQA) — fewer KV heads reduce the KV cache size at inference, lowering memory bandwidth requirements without significantly impacting quality.",
    "mha": "Uses standard **Multi-Head Attention** (MHA) — all heads have independent KV projections.",
    "mqa": "Uses **Multi-Query Attention** (MQA) — all query heads share a single KV head, maximising inference speed at the cost of some quality.",
    "sliding": "Uses **Sliding Window Attention** — each token attends only to the nearest W tokens, enabling efficient very long sequence processing.",
}

_PE_NOTES = {
    "rope": "**RoPE** (Rotary Position Embedding) is applied inside attention by rotating query/key vectors, giving strong length generalisation.",
    "alibi": "**ALiBi** (Attention with Linear Biases) adds head-specific linear position penalties to attention logits — no learned positional parameters.",
    "learned": "Uses **learned absolute positional embeddings** (GPT-2 / BERT style) — one embedding vector per position up to max_seq_len.",
    "none": "No explicit positional encoding — the model learns implicit position information from causal masking.",
}


def generate_model_card(
    exp: ExperimentConfig,
    output_path: str | Path,
    author: str = "Unknown",
    github_user: str = "your-username",
    eval_results: Optional[dict] = None,
    extra_notes: str = "",
) -> Path:
    """
    Generate a README.md model card and write it to output_path.

    Args:
        exp          : The ExperimentConfig used for training.
        output_path  : Where to write the README.md.
        author       : Your name / GitHub username.
        github_user  : GitHub username for the repo link.
        eval_results : Dict of {metric_name: value} for the eval table.
                       e.g. {"eval_loss": 2.34, "perplexity": 10.4}
        extra_notes  : Additional text appended to the Architecture notes.

    Returns:
        Path to the written README.md.
    """
    m = exp.model
    t = exp.training

    # Parameter count estimate
    n_params = _estimate_params(m)
    param_str = (
        f"{n_params / 1e6:.0f}M" if n_params >= 1e6 else f"{n_params / 1e3:.0f}K"
    )

    # Attention detail
    attn_detail = f"{m.attention.num_heads}Q / {m.attention.num_kv_heads}KV heads, head_dim={m.head_dim}"
    if m.attention.type == "sliding":
        attn_detail += f", window={m.attention.window_size}"

    # PE detail
    pe_detail_map = {
        "rope": f"theta={m.positional.theta:.0f}",
        "alibi": "no learned params",
        "learned": f"max_seq_len={m.positional.max_seq_len}",
        "none": "—",
    }
    pe_detail = pe_detail_map.get(m.positional.type, "")

    # Eval section
    eval_section = ""
    if eval_results:
        rows = "\n".join(f"| {k} | {v} |" for k, v in eval_results.items())
        eval_section = _EVAL_SECTION_TEMPLATE.format(eval_rows=rows)

    # Arch notes
    attn_note = _ARCH_NOTES.get(m.attention.type, "")
    pe_note = _PE_NOTES.get(m.positional.type, "")
    arch_notes = f"{attn_note}\n\n{pe_note}"
    if extra_notes:
        arch_notes += f"\n\n{extra_notes}"

    hub_repo = exp.hub.repo_id or f"{github_user}/{exp.name}"

    card = _CARD_TEMPLATE.format(
        model_name=exp.name,
        param_str=param_str,
        attention_type=m.attention.type,
        positional_type=m.positional.type,
        ffn_type=m.ffn.type,
        norm_type=m.norm.type,
        attn_detail=attn_detail,
        pe_detail=pe_detail,
        intermediate_size=m.ffn.intermediate_size,
        norm_eps=m.norm.eps,
        num_layers=m.num_layers,
        hidden_size=m.hidden_size,
        vocab_size=m.vocab_size,
        max_seq_len=m.max_seq_len,
        lr=t.lr,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        batch_size=t.batch_size,
        grad_accum=t.grad_accum,
        effective_batch=t.effective_batch_size,
        seq_len=t.seq_len,
        max_steps=t.max_steps,
        dtype=t.dtype,
        train_date=datetime.now().strftime("%Y-%m-%d"),
        eval_section=eval_section,
        hub_repo_id=hub_repo,
        arch_notes=arch_notes,
        citation_key=exp.name.replace("-", "_"),
        author=author,
        year=datetime.now().year,
        github_user=github_user,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card)
    return out


def _estimate_params(m) -> int:
    """Rough parameter count from config (no model instantiation needed)."""
    h = m.hidden_size
    v = m.vocab_size
    n = m.num_layers
    ffn_h = m.ffn.intermediate_size
    q_heads = m.attention.num_heads
    kv_heads = m.attention.num_kv_heads
    head_d = h // q_heads

    embed = v * h
    q_proj = h * (q_heads * head_d)
    kv_proj = h * (kv_heads * head_d) * 2  # k + v
    o_proj = (q_heads * head_d) * h
    attn = (q_proj + kv_proj + o_proj) * n

    ffn_mult = 3 if m.ffn.type in ("swiglu", "geglu") else 2
    ffn = h * ffn_h * ffn_mult * n

    norms = h * 2 * n + h  # per-layer norms + final norm
    lm_head = 0 if m.tie_word_embeddings else v * h

    return embed + attn + ffn + norms + lm_head
