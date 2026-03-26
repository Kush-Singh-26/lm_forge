import argparse
import os
import torch
from engine.config import load_config
from engine.legacy.training import Trainer
from engine.models import CausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="experiments/powershell_gen/config.yaml"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize model
    model = CausalLM(config.model)

    # Initialize trainer
    trainer = Trainer(config, model)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
