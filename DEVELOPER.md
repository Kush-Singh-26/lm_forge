# Developer Guide — Forge

Welcome to the engineering guide for `Forge`. This document explains the architecture of the Nomad Training system and how to extend it.

## 1. Core Philosophy: Nomad Training

Forge is designed for **transient compute**. Unlike traditional training loops that assume a stable cluster, Forge assumes the trainer might be preempted, moved between providers (e.g., from Modal to Colab), or auto-scaled.

The "Source of Truth" for state is not the local disk, but the **Hugging Face Hub** (specifically a `checkpoints` branch).

## 2. System Architecture

### `forge/state/hub_manager.py`
The heart of the state system. It handles:
- **Atomic Commits**: Creating verified pointers (`latest.json`) to specific checkpoint shards.
- **Branch Management**: Isynchronizing state to a non-default branch to keep the `main` repo clean.
- **Shipment**: Pruning training artifacts (optimizer states) and exporting a clean model to the `main` branch.

### `forge/integration/hf.py` (`ForgeTrainer`)
A subclass of the Hugging Face `Trainer`. It overrides the core loop to inject:
- **Resumption Logic**: Automatically identifying the latest remote checkpoint before training starts.
- **ForgeCallback**: A custom `TrainerCallback` that triggers Hub synchronization and TUI updates.

### `forge/data/streaming.py`
Provides utilities for `IterableDatasets` to ensure deterministic resumption. It tracks `samples_seen` cumulatively to handle batch-size adjustments without losing the data cursor.

### `forge/integration/prober.py`
Contains the `MemoryProber`, which performs binary search on batch sizes if an OOM occurs, then scales `gradient_accumulation_steps` to preserve the user's target global batch size.

## 3. Extending Forge

### Adding a New Provider
To add a new compute provider (e.g., Lambda Labs, RunPod):
1. Create a new class in `forge/providers/`.
2. Implement the `run(script, args)` method.
3. Register the provider in the `run` command within `forge/cli.py`.

### Modifying the Dashboard
The TUI is implemented in `forge/integration/dash.py` using `rich`. It reads `forge_status.json` which is updated by the `ForgeCallback` during training.

## 4. Development Workflow

1. **Local Testing**: Always run `python main.py run local --script examples/minimal_train/train.py` to verify the integration.
2. **Resumption Testing**: Use `tests/test_forge_resumption.py` to ensure that `samples_seen` is correctly tracked across simulated crashes.
3. **Benchmarking**: Use `forge profile` to check MFU on your current hardware before shipping changes to the training loop.
