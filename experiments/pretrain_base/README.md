# Pretrain Base Model — 85M Parameter Pretraining

Pretrain an 85M parameter decoder-only transformer on two datasets sequentially, then fine-tune on domain-specific data (PowerShell).

## What This Does

| Phase | Dataset | Tokens | Purpose |
|-------|---------|--------|---------|
| Phase 1 | `HuggingFaceFW/fineweb-edu` | ~10B | Grammar, logic, world knowledge, sentence structure |
| Phase 2 | `bigcode/the-stack-v2` | ~5–10B | Code syntax, variables, loops, script patterns |
| Phase 3 | PS1 corpus (separate experiment) | fine-tune | PowerShell generation |

**Model:** 16 layers, hidden 512, GQA (8Q/2KV), RoPE, SwiGLU, RMSNorm, SmolLM tokenizer (49,152 vocab).

---

## Prerequisites

```bash
pip install torch transformers datasets huggingface_hub safetensors
```

Optional (recommended):
```bash
pip install wandb accelerate
```

---

## Step 0 — Smoke Test (Verify Everything Works)

Run a 50-step CPU smoke test with synthetic data. Takes ~1 minute.

```bash
cd lm_forge
python experiments/pretrain_base/train.py --phase 1 --smoke --steps 50
```

If this completes without errors, you're ready for real training.

---

## Step 1 — Phase 1: General Knowledge (fineweb-edu)

### On Modal (A100/H100) — Recommended

```bash
python experiments/pretrain_base/train.py --phase 1
```

This will:
- Stream fineweb-edu from HuggingFace (no download needed)
- Train for 38,000 steps (~10B tokens)
- Use BF16 mixed precision
- Save checkpoints every 1,000 steps to `checkpoints/pretrain_phase1/`
- Estimated time: **8–16 hours** on A100

### On Google Colab (T4)

```bash
python experiments/pretrain_base/train.py --phase 1 --t4
```

The `--t4` flag automatically applies:
- `batch_size=8, grad_accum=32` (effective batch stays 256)
- `dtype=float16` with GradScaler

**Colab session management:**
Colab free tier disconnects after ~4–6 hours. The script auto-resumes from the latest checkpoint.

```
Session 1: steps 0 → ~3,000       (checkpoint saved at step 3000)
Session 2: auto-resumes → ~6,000
Session 3: auto-resumes → ~9,000
...
Session 12+: reaches 38,000
```

Each session:
```bash
# In Colab notebook cell:
!python experiments/pretrain_base/train.py --phase 1 --t4
```

### Verify Phase 1

After completion, check:
```bash
ls checkpoints/pretrain_phase1/final/
# Should contain: model.safetensors, config.json, tokenizer.json, ...
```

The script also runs a quick generation test automatically.

---

## Step 2 — Phase 2: Code (the-stack-v2)

Phase 2 starts from the Phase 1 checkpoint. The model already knows English grammar and reasoning — now it learns code.

### On Modal

```bash
python experiments/pretrain_base/train.py --phase 2 \
    --resume checkpoints/pretrain_phase1/final
```

Key differences from Phase 1:
- **Lower learning rate:** 1e-4 (vs 3e-4 in Phase 1) — prevents forgetting general knowledge
- **Code dataset:** Python, JavaScript, TypeScript, Java, C++, C, Go, Rust
- **Same architecture, same optimizer state** — just new data and LR

### On T4

```bash
python experiments/pretrain_base/train.py --phase 2 \
    --resume checkpoints/pretrain_phase1/final --t4
```

Same auto-resume behavior as Phase 1.

### Verify Phase 2

```bash
ls checkpoints/pretrain_phase2/final/
```

Generation test should produce code-like output:
```
Prompt: def fibonacci(n):
Output: def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

---

## GPU Switching (Modal → T4 or Vice Versa)

No code changes needed. HF Trainer handles everything automatically.

### Modal → T4

1. Finish your Modal session (checkpoint is saved)
2. Copy checkpoint to Colab (or use HF Hub)
3. Run with `--t4`:

```bash
python experiments/pretrain_base/train.py --phase 1 \
    --resume checkpoints/pretrain_phase1/checkpoint-15000 --t4
```

**What happens automatically:**
- Model weights loaded → cast to FP16
- Optimizer state (Adam moments) loaded → stays in FP32
- Scheduler state loaded → resumes at correct step
- Fresh GradScaler created for FP16
- Training continues seamlessly

### T4 → Modal

```bash
python experiments/pretrain_base/train.py --phase 1 \
    --resume checkpoints/pretrain_phase1/checkpoint-20000
```

- FP16 weights → cast to BF16
- Optimizer state loaded
- Scheduler continues
- BF16 doesn't need GradScaler

### Key Rule

**Never change these between GPU switches:**
- Model architecture (defined in config YAML — stays same)
- Learning rate (defined in config YAML — stays same)
- Optimizer type (AdamW — stays same)

**Only change these (the `--t4` flag handles it):**
- `batch_size` and `grad_accum` (keep effective batch constant)
- `dtype` (BF16 ↔ FP16)

---

## Checkpoint Structure

Each checkpoint contains everything needed to resume:

```
checkpoints/pretrain_phase1/checkpoint-1000/
├── model.safetensors          # model weights
├── optimizer.pt               # Adam state (moments)
├── scheduler.pt               # cosine schedule + step count
├── trainer_state.json         # global_step, logging history
├── config.json                # model architecture config
├── tokenizer.json             # SmolLM tokenizer
├── tokenizer_config.json
├── generation_config.json
└── rng_states.pth             # RNG states for reproducibility
```

**Important:** `trainer_state.json` contains `global_step`. This is what tells the scheduler where to resume. Never delete it.

### Checkpoint Cleanup

By default, only the last 5 checkpoints are kept (`save_total_limit: 5`).

To manually keep a specific checkpoint:
```bash
cp -r checkpoints/pretrain_phase1/checkpoint-10000 checkpoints/pretrain_phase1/keep_step10000
```

---

## Custom Configuration

### Change model size

Edit `config_phase1.yaml`:
```yaml
model:
  hidden_size: 768       # larger
  num_layers: 24         # deeper
  ffn:
    intermediate_size: 2048
```

### Change training duration

```yaml
training:
  max_steps: 100000      # train longer
```

Or use CLI:
```bash
python experiments/pretrain_base/train.py --phase 1 --steps 10000
```

### Use different code languages

Edit `train.py` in the `get_datasets()` function:
```python
code_languages = ["Python", "JavaScript"]  # only these two
```

### Enable W&B logging

In config YAML:
```yaml
training:
  hf_args:
    report_to: ["wandb"]
```

---

## Monitoring

### Console output

The script logs every 10 steps:
```
{'loss': 3.421, 'learning_rate': 0.00015, 'epoch': 0.0, 'step': 100}
```

### Expected loss curves

| Phase | Start loss | End loss | Notes |
|-------|-----------|----------|-------|
| Phase 1 | ~4.5–5.0 | ~2.8–3.2 | General knowledge acquisition |
| Phase 2 | ~2.5–3.0 | ~1.8–2.2 | Code is more structured (lower loss) |

If loss plateaus above 3.5 in Phase 1, train longer.

### Tokens processed

Track tokens manually:
```
tokens_processed = global_step × batch_size × grad_accum × seq_len
```

Example at step 20,000:
```
20,000 × 32 × 8 × 1024 = ~5.2B tokens
```

---

## Troubleshooting

### "CUDA out of memory"

```bash
# Reduce batch size
python experiments/pretrain_base/train.py --phase 1 --t4

# Or edit config: batch_size: 4, grad_accum: 64
```

### Streaming connection drops

HF datasets streaming auto-reconnects. If it fails repeatedly:
```bash
# Use --no-streaming to download a small slice locally
python experiments/pretrain_base/train.py --phase 1 --no-streaming --steps 1000
```

### "Checkpoint not found" on resume

Make sure the path is correct:
```bash
ls checkpoints/pretrain_phase1/
# Find the latest checkpoint-XXXXX directory
```

### Loss spikes after GPU switch

This can happen for the first ~100 steps as the optimizer adapts to new precision. It should stabilize. If it doesn't:
- Check that effective batch size is the same
- Ensure you didn't accidentally change the learning rate
- Verify `trainer_state.json` has the correct `global_step`

### Generation quality is poor

This is expected during pretraining. The model learns structure first, coherence later. Quality improves significantly after fine-tuning (Phase 3 — PS1 data).

---

## After Pretraining → Fine-Tuning

Once both phases are complete:

```bash
# Final pretrained model location
ls checkpoints/pretrain_phase2/final/
```

Use this as the base for PS1 fine-tuning (existing `experiments/powershell_gen/` experiment, but loading from this checkpoint instead of random init).

---

## File Overview

```
experiments/pretrain_base/
├── README.md                  # this file
├── config_phase1.yaml         # Phase 1 config (fineweb-edu)
├── config_phase2.yaml         # Phase 2 config (the-stack-v2)
├── train.py                   # main training script
├── data_streaming.py          # streaming dataset builders
└── colab_train.ipynb          # Colab T4 notebook
```

---

## Quick Reference

```bash
# Smoke test (CPU, 50 steps)
python experiments/pretrain_base/train.py --phase 1 --smoke

# Phase 1 (GPU)
python experiments/pretrain_base/train.py --phase 1

# Phase 1 (T4)
python experiments/pretrain_base/train.py --phase 1 --t4

# Phase 1 (quick test, 500 steps)
python experiments/pretrain_base/train.py --phase 1 --steps 500

# Phase 2 (from Phase 1 checkpoint)
python experiments/pretrain_base/train.py --phase 2 --resume checkpoints/pretrain_phase1/final

# Phase 2 (T4, from specific checkpoint)
python experiments/pretrain_base/train.py --phase 2 --resume checkpoints/pretrain_phase1/checkpoint-20000 --t4

# Custom config
python experiments/pretrain_base/train.py --phase 1 --config my_config.yaml

# Offline mode (downloads a small slice instead of streaming)
python experiments/pretrain_base/train.py --phase 1 --no-streaming --steps 1000
```
