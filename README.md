# lm_forge

A modular small language model training engine with native Hugging Face support.

Designed for the **Colab Nomad** pattern: Colab sessions are disposable, GitHub holds the code, HuggingFace Hub holds the weights. Kill the session at any time and resume exactly where you left off.

---

## What this is

A clean-room implementation of every component in a decoder-only transformer, with a plugin registry system that lets you swap attention, positional encoding, FFN, and normalisation by changing one field in a YAML config — zero code changes.

**HF Native:** `lm_forge` is built around the Hugging Face ecosystem. It uses `transformers.Trainer`, `datasets.Dataset`, and `PreTrainedModel` as its primary interfaces.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/lm_forge
cd lm_forge
pip install -r requirements.txt
```

**Run the HF Native training example:**
```bash
python experiments/hf_native_example/train.py --steps 100 --cpu
```

---

## Architecture

```
lm_forge/
├── engine/                         # the library — import this
│   ├── config/
│   │   ├── schema.py               # nested YAML-loadable dataclasses
│   │   └── hf_config.py            # PretrainedConfig bridge for HF
│   ├── components/
│   │   ├── attention/              # mha · gqa · mqa · sliding
│   │   ├── positional/             # rope · alibi · learned · none
│   │   ├── ffn/                    # swiglu · geglu · classic
│   │   └── norm/                   # rms · layer
│   ├── models/
│   │   ├── decoder.py              # CausalLM — core decoder model
│   │   └── hf_model.py             # HFCausalLM — PreTrainedModel wrapper
│   ├── data/
│   │   ├── hf_utils.py             # prepare_dataset() — tokenize & pack
│   │   └── collators.py            # CLMCollator · MLMCollator
│   ├── tokenizer/
│   │   ├── bpe.py                  # byte-level BPE from scratch
│   │   └── hf_tokenizer.py         # HFBPETokenizer wrapper
│   ├── utils/
│   │   ├── hf_callbacks.py         # ProfilingCallback, HubCheckpointCallback
│   │   ├── hub_checkpoint_utils.py # HubCheckpointManager for no-bloat sync
│   │   └── profiler.py             # MFU, throughput measurement logic
│   └── legacy/                     # Deprecated components (Trainer, HubSync, etc.)
├── experiments/
│   ├── hf_native_example/          # Recommended starting point
│   │   ├── config.yaml             
│   │   └── train.py                # Uses transformers.Trainer
│   └── exp_001_gqa_rope/           # GQA + RoPE experiment
├── scripts/
│   └── verify_hf_native.py         # Integration smoke test
└── tests/                          # pytest suite
```

---

## HF Native Workflow

The recommended workflow uses Hugging Face `datasets` and `transformers.Trainer`. See **`experiments/hf_native_example/`** for a simple starting point and **`experiments/pretrain_base/`** for a complete 160M parameter pretraining pipeline.

### 1. Model Configuration
`lm_forge` uses a YAML config that maps directly to HF's `PretrainedConfig` via `LMForgeConfig`.

```yaml
model:
  attention:
    type: "gqa"
  positional:
    type: "rope"
  ffn:
    type: "swiglu"
  norm:
    type: "rms"
```

### 2. Data Preparation
Use `prepare_dataset()` to tokenize and pack any HF dataset on-the-fly.

```python
from engine import prepare_dataset
from datasets import load_dataset

raw_ds = load_dataset("roneneldan/TinyStories", split="train")
train_ds = prepare_dataset(raw_ds, tokenizer, seq_len=1024)
```

### 3. Training
Train using the standard `transformers.Trainer` with our native `HFCausalLM`.

```python
from engine import HFCausalLM
from engine.utils import HubCheckpointCallback
from transformers import Trainer

model = HFCausalLM(hf_config)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    callbacks=[ProfilingCallback(), HubCheckpointCallback(exp)] # MFU + Zero-Bloat Hub Sync
)
trainer.train()
```

---

## The Colab Nomad Pattern (Zero Bloat)

`lm_forge` is optimized for training on transient hardware (Google Colab, Modal, etc.) using a **Zero-Bloat Hub Checkpointing** strategy.

### Why not standard Git?
Traditional Git repositories are versioned. Saving a 2GB checkpoint every hour creates massive hidden history.

### Our Solution
1. **Isolated Branching**: All checkpoints are stored on a dedicated `checkpoints` branch. Your `main` branch stays 100% clean.
2. **Mutable Storage**: The `HubCheckpointCallback` automatically deletes old checkpoints from the Hub after each successful save, keeping only the most recent $N$ (default is 2).
3. **Automatic Resumption**: Training scripts automatically scan the `checkpoints` branch, download the latest state, and resume bit-perfectly.

### Configuration
In your `config.yaml`:

```yaml
hub:
  repo_id: "your-username/your-repo"
  use_hub_checkpoints: true
  checkpoint_limit: 2
  push_every: 500  # Sync every N steps
```

---

## Features

| Feature | Status | Notes |
|---|---|---|
| **HF Ecosystem** | ✅ Full | Works with `pipeline`, `AutoModel`, `PEFT/LoRA`, `Accelerate` |
| **KV Caching** | ✅ Native | Full support for incremental decoding in `forward()` |
| **Profiling** | ✅ Callback | `ProfilingCallback` logs MFU, tok/s, and memory to Trainer |
| **Zero Bloat Hub** | ✅ Native | `HubCheckpointCallback` for clean, mutable checkpointing |
| **Component Swap** | ✅ Registry | Swap attention/FFN/PE variants via YAML config |
| **LoRA Support** | ✅ Native | Projection names mirror LLaMA for easy PEFT integration |

---

## Legacy Components

While `lm_forge` is now HF-native, the original custom components (for educational purposes or low-dependency environments) are available in `engine/legacy/`:
- `Trainer`: Custom training loop with HubSync.
- `MemmapDataset`: uint16 binary dataset format.
- `HubSync`: Automated checkpoint syncing to HF Hub.

---

## Requirements

```
torch >= 2.0.0
transformers >= 4.46.0
datasets >= 2.18.0
huggingface_hub >= 0.26.0
pyyaml >= 6.0
```

---

## License

MIT
