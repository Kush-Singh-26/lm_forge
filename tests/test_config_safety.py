import yaml
import warnings
from pathlib import Path
import pytest
from engine.config.schema import load_experiment_config


def test_config_typo_warning():
    """
    Verifies that load_experiment_config raises a warning for unknown configuration keys.
    """
    bad_yaml = """
experiment:
  name: "typo-test"
model:
  hidden_size: 512
  layers: 6  # Should be 'num_layers'
training:
  lr: 0.0003
  learning_rate: 0.1  # Typo
"""
    tmp_path = Path("tmp_bad_config.yaml")
    tmp_path.write_text(bad_yaml)

    # We use pytest to catch the warning
    with pytest.warns(UserWarning, match="Found unknown configuration keys in YAML"):
        cfg = load_experiment_config(tmp_path)

    tmp_path.unlink()


if __name__ == "__main__":
    test_config_typo_warning()
