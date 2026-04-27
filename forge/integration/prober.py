import torch
from transformers import Trainer, TrainingArguments, DataCollator
from typing import Any, Dict, Optional, Tuple
import gc

class ForgeMemoryProber:
    """
    Auto-tunes batch sizes to fit VRAM limits across different GPUs.
    """
    @staticmethod
    def probe_batch_size(
        model: torch.nn.Module,
        train_dataset: Any,
        data_collator: DataCollator,
        target_total_batch: int,
        initial_per_device_batch: int = 1,
        max_steps: int = 5,
        fp16: bool = True,
        bf16: bool = False,
    ) -> Tuple[int, int]:
        """
        Tests batch sizes to find the best fit.
        Returns: (per_device_batch_size, grad_accumulation_steps)
        """
        print(f"[Forge.Prober] Probing memory limits for {target_total_batch} target batch size...")
        
        # We start with the user's initial guess and scale down if it OOMs, 
        # or we could scale UP to find the limit. Let's try scaling UP from 1.
        
        current_batch = initial_per_device_batch
        best_batch = 1
        
        while current_batch <= target_total_batch:
            try:
                # 1. Clear cache
                gc.collect()
                torch.cuda.empty_cache()
                
                # 2. Run a mini-training pass
                args = TrainingArguments(
                    output_dir="/tmp/forge_probe",
                    per_device_train_batch_size=current_batch,
                    max_steps=max_steps,
                    fp16=fp16,
                    bf16=bf16,
                    report_to="none",
                    logging_steps=1
                )
                
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    data_collator=data_collator
                )
                
                print(f"[Forge.Prober] Testing per_device_batch_size={current_batch}...")
                trainer.train()
                
                # If we get here, it succeeded
                best_batch = current_batch
                
                # Try doubling for the next test
                if current_batch * 2 > target_total_batch:
                    break
                current_batch *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[Forge.Prober] OOM at batch_size={current_batch}. Using best successful: {best_batch}")
                    break
                else:
                    raise e
        
        grad_accum = max(1, target_total_batch // best_batch)
        print(f"[Forge.Prober] Final recommendation: per_device={best_batch}, grad_accum={grad_accum}")
        return best_batch, grad_accum
