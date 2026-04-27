"""
forge/data/streaming.py

Streaming data helpers for Nomad Training.
Provides utilities for sharding and stateful resumption of streaming datasets.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Any, List
from datasets import IterableDataset

class ForgeDataHelper:
    """
    Utilities for preparing streaming datasets for Nomad Training.
    """

    @staticmethod
    def prepare_streaming_dataset(
        dataset: IterableDataset,
        rank: int = 0,
        world_size: int = 1,
        skip_samples: int = 0,
    ) -> IterableDataset:
        """
        Shards and fast-forwards an IterableDataset for stateful resumption.
        """
        ds = dataset

        if world_size > 1:
            print(f"[Forge.Data] Sharding stream: rank {rank}/{world_size}")
            ds = ds.shard(num_shards=world_size, index=rank)

        if skip_samples > 0:
            print(f"[Forge.Data] Fast-forwarding: skipping {skip_samples:,} samples...")
            ds = ds.skip(skip_samples)
            
        return ds

    @staticmethod
    def setup_fast_cache():
        """
        Detects fastest local storage (/scratch, /tmp, or NVMe) 
        and points HF_HOME there.
        """
        import os
        # Order of preference for fast scratch space
        potential_fast_disks = ["/scratch", "/tmp", os.environ.get("LOCAL_NVME", "")]
        for disk in potential_fast_disks:
            if disk and os.path.exists(disk):
                print(f"[Forge.Data] Fast storage detected at {disk}. Redirecting HF cache...")
                cache_dir = os.path.join(disk, ".cache", "huggingface")
                os.environ["HF_HOME"] = cache_dir
                os.makedirs(cache_dir, exist_ok=True)
                return cache_dir
        return None

    @staticmethod
    def combine_datasets(
        datasets_dict: dict[str, IterableDataset],
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ) -> IterableDataset:
        """
        Combines multiple streaming datasets using 'interleave_datasets'.
        This is the 'Hydra' pattern for mixed-dataset streaming.
        """
        from datasets import interleave_datasets
        
        ds_list = list(datasets_dict.values())
        names = list(datasets_dict.keys())
        
        print(f"[Forge.Data] Interleaving {len(ds_list)} datasets: {names}")
        if weights:
            print(f"[Forge.Data] Using mixture weights: {weights}")
            
        return interleave_datasets(
            ds_list,
            probabilities=weights,
            seed=seed,
            stopping_strategy="all_exhausted"
        )

    @classmethod
    def get_resume_stats(cls, checkpoint_dir: str | Path) -> dict:
        """
        Reads data_state.json from a checkpoint directory to get resumption stats.
        """
        import json
        from pathlib import Path
        
        state_file = Path(checkpoint_dir) / "data_state.json"
        if not state_file.exists():
            # If not in the direct folder, try looking in subdirectories (checkpoint-XXXX)
            path = Path(checkpoint_dir)
            if path.exists():
                checkpoints = []
                for d in path.iterdir():
                    if d.is_dir() and d.name.startswith("checkpoint-"):
                        suffix = d.name.split("checkpoint-")[-1]
                        if suffix.isdigit():
                            checkpoints.append((int(suffix), d))
                
                checkpoints.sort(key=lambda x: x[0])
                if checkpoints:
                    state_file = checkpoints[-1][1] / "data_state.json"
        
        if not state_file.exists():
            return {"samples_seen": 0, "global_step": 0}
            
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Forge.Data] WARNING: Failed to read data state: {e}")
            return {"samples_seen": 0, "global_step": 0}
