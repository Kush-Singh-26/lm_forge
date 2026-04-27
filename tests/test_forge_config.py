import os
from pathlib import Path
from forge.config import ForgeConfig, StateConfig

def test_detect_env_local():
    if "MODAL_IMAGE_ID" in os.environ:
        del os.environ["MODAL_IMAGE_ID"]
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        del os.environ["KAGGLE_KERNEL_RUN_TYPE"]
    assert ForgeConfig.detect_env() == "local"

def test_detect_env_modal():
    os.environ["MODAL_IMAGE_ID"] = "test"
    assert ForgeConfig.detect_env() == "modal"
    del os.environ["MODAL_IMAGE_ID"]

def test_config_load_save(tmp_path):
    config_path = tmp_path / "forge.yaml"
    cfg = ForgeConfig(
        name="test-project",
        state=StateConfig(repo_id="user/repo")
    )
    cfg.save(config_path)
    
    loaded = ForgeConfig.load(config_path)
    assert loaded.name == "test-project"
    assert loaded.state.repo_id == "user/repo"
