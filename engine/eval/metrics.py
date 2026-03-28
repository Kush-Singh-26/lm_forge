"""
engine/eval/metrics.py

Evaluation metrics for language models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Optional


@torch.no_grad()
def calculate_perplexity(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Calculate the perplexity of a model on a given dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Check if the model has a forward that returns (logits, loss, ...)
    # or just logits.
    # Our CausalLM returns (logits, loss, presents).

    for batch in tqdm(dataloader, desc="Calculating Perplexity"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        logits, loss, _ = model(input_ids=input_ids, labels=labels)

        # We want the cross-entropy loss sum (not averaged by batch size)
        # but averaged by token count.
        # HFTrainer usually averages across batch * seq_len (minus -100).

        if loss is not None:
            # Hugging Face Trainer loss is already averaged by number of non-ignored labels.
            # We want to re-weight it correctly.

            # Count tokens that are NOT -100
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()
