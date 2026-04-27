"""
forge/config.py

Configuration schema for the Forge Training Assist tool.
Uses Pydantic for validation and type-safety.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml
from pydantic import BaseModel, Field, ConfigDict

class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    repo_id: str = Field(description="Hugging Face Hub repository ID (e.g. 'user/model')")
    branch: str = Field(default="checkpoints", description="Branch to store checkpoints on")
    checkpoint_limit: int = Field(default=2, gt=0, description="Number of recent checkpoints to keep")
    push_every: int = Field(default=500, gt=0, description="Steps between Hub syncs")
    token_env: str = Field(default="HF_TOKEN", description="Env var containing HF token")
    private: bool = Field(default=True, description="Whether the Hub repo is private")

class ProfileConfig(BaseModel):
    """Overrides for TrainingArguments based on environment."""
    model_config = ConfigDict(extra="allow") # Allow arbitrary TrainingArguments
    
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    max_steps: Optional[int] = None
    probe_memory: bool = Field(default=False, description="Auto-tune batch size if OOM occurs")
    data_cache: bool = Field(default=False, description="Pre-fetch dataset shards to local NVMe")

class ModalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    gpu: str = Field(default="A100", description="GPU type (A100, H100, L4, etc.)")
    gpu_count: int = Field(default=1, gt=0)
    cpu: int = Field(default=8)
    memory: int = Field(default=32768, description="Memory in MB")
    timeout: int = Field(default=86400, description="Job timeout in seconds")

class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    modal: ModalConfig = Field(default_factory=ModalConfig)
    # Future providers (RunPod, Lambda, etc.) can be added here

class ForgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Project/Experiment name")
    state: StateConfig
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    profiles: Dict[str, ProfileConfig] = Field(default_factory=dict)
    
    @classmethod
    def load(cls, path: str | Path) -> ForgeConfig:
        path = Path(path)
        if not path.exists():
            # Return a default config if not found, or handle error
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        return cls(**data)

    def get_active_profile(self) -> Optional[ProfileConfig]:
        """Detect environment and return matching profile."""
        env = self.detect_env()
        return self.profiles.get(env) or self.profiles.get("default")

    @staticmethod
    def detect_env() -> str:
        """Detects the current training environment."""
        if os.environ.get("MODAL_IMAGE_ID"):
            return "modal"
        
        # Kaggle Detection
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            return "kaggle"
            
        # SageMaker Studio Lab Detection
        if os.environ.get("SAGEMAKER_STUDIO_LAB_ROOT") or Path("/opt/ml").exists():
            return "sagemaker"
            
        try:
            import google.colab
            return "colab"
        except ImportError:
            return "local"

    def save(self, path: str | Path):
        path = Path(path)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
