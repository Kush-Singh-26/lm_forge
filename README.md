# Forge: The Nomad Training Utility

Forge is a streamlined orchestrator and state-management tool designed for **Nomad Training**—the ability to seamlessly hop between local dev, serverless GPUs (Modal), and free notebook providers (Colab, Kaggle) without losing a single sample of progress.

## Why Forge?

Traditional cloud orchestrators (SkyPilot, Lightning) often stick your data in cloud buckets (S3/GCS) or persistent volumes. **Forge uses the Hugging Face Hub as your universal state backbone**, enabling zero-infrastructure resumption anywhere.

### Key Capabilities

*   **ForgeTrainer™**: A boilerplate-free drop-in replacement for `transformers.Trainer`.
*   **Atomic Syncing**: Checkpoints are uploaded to a dedicated `checkpoints` branch on HF Hub and verified with a `latest.json` pointer.
*   **Data Cursor Tracking**: Automatic stateful resumption for `IterableDatasets`. Forge calculates the exact sample skip count across different GPU counts.
*   **Memory Prober**: Auto-tunes batch sizes to fit VRAM limits while preserving your target Global Batch Size via inverse Gradient Accumulation scaling.
*   **Lazy Data Fetching**: Proactively redirects HF cache to local node NVMe/scratch space for maximum streaming throughput.
*   **Multi-Provider CLI**: One-click launch on Modal or local hardware, with snippet generation for Colab, Kaggle, and SageMaker.

---

## Quick Start

### 1. Installation

```bash
pip install -e .
```

### 2. Initialize Project

```bash
forge init --name my-gpt --repo-id your-user/my-gpt-checkpoints
```

This generates a `forge.yaml` with pre-configured profiles for **Colab**, **Kaggle**, and **Modal**.

### 3. "Boilerplate-Free" Integration

Simply swap `Trainer` for `ForgeTrainer` in your script:

```python
from forge import ForgeTrainer

trainer = ForgeTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset, # IterableDatasets are auto-resumed!
    # All Forge magic (Hub sync, Prober, Profiling) is now active
)
trainer.train()
```

### 4. The Nomad Workflow

**Ship to Modal:**
```bash
forge run modal --script train.py
```

**Move to Kaggle:**
1. Run `forge notebook kaggle` to get your setup snippet.
2. Paste into Kaggle. Forge will automatically `pull` the latest state from your Hub and resume.

**Check Status:**
```bash
forge status
```

---

## Advanced Config (`forge.yaml`)

```yaml
name: "my-gpt"
state:
  repo_id: "user/repo"
  branch: "checkpoints"
  checkpoint_limit: 2
profiles:
  colab:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 16
    probe_memory: true  # Auto-adjust if it OOMs
  modal:
    per_device_train_batch_size: 8
    data_cache: true    # Use fast local scratch space
```

## Community & Contribution

Forge is built for the open-source ML community. If you have suggestions for new "Nomad Targets" or features, please open an issue!
