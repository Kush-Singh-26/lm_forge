import os
import json
from pathlib import Path
import pytest
from transformers import TrainingArguments, TrainerState, TrainerControl
from forge.integration.hf import ForgeCallback, ForgeTrainer
from forge.config import ForgeConfig, StateConfig

def test_callback_samples_seen(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Create a dummy forge.yaml
    config_path = tmp_path / "forge.yaml"
    cfg = ForgeConfig(name="test", state=StateConfig(repo_id="test/repo"))
    cfg.save(config_path)
    
    callback = ForgeCallback(config_path=config_path)
    args = TrainingArguments(output_dir=str(output_dir), per_device_train_batch_size=2, gradient_accumulation_steps=2)
    state = TrainerState()
    control = TrainerControl()
    
    # Simulate a few steps
    callback.on_step_end(args, state, control) # 2 * 2 * 1 = 4 samples
    callback.on_step_end(args, state, control) # +4 = 8 samples
    
    assert callback.samples_seen == 8
    
    # Simulate save
    state.global_step = 2
    checkpoint_dir = output_dir / "checkpoint-2"
    checkpoint_dir.mkdir()
    
    # We need a dummy model file for is_valid_checkpoint to pass if we were using hub_manager.pull_latest
    # but here we just check if it saves data_state.json
    callback.on_save(args, state, control)
    
    state_file = checkpoint_dir / "data_state.json"
    assert state_file.exists()
    with open(state_file, "r") as f:
        saved_data = json.load(f)
        assert saved_data["samples_seen"] == 8

def test_callback_resumption(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    checkpoint_dir = output_dir / "checkpoint-10"
    checkpoint_dir.mkdir()
    
    # Pre-populate data_state.json
    with open(checkpoint_dir / "data_state.json", "w") as f:
        json.dump({"samples_seen": 1000, "global_step": 10}, f)
        
    config_path = tmp_path / "forge.yaml"
    cfg = ForgeConfig(name="test", state=StateConfig(repo_id="test/repo"))
    cfg.save(config_path)
    
    callback = ForgeCallback(config_path=config_path)
    args = TrainingArguments(output_dir=str(output_dir))
    state = TrainerState()
    
    # Should load 1000 from the latest checkpoint in output_dir
    callback.on_train_begin(args, state, TrainerControl())
    assert callback.samples_seen == 1000
