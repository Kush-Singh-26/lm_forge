from setuptools import setup, find_packages

setup(
    name="lm_forge_engine",
    version="0.2.0",
    description="Modular LM training engine — Colab Nomad edition",
    packages=find_packages(include=["engine", "engine.*"]),
    python_requires=">=3.10",
    install_requires=["torch>=2.0.0", "pyyaml>=6.0"],
    extras_require={
        "hub":      ["huggingface_hub>=0.20.0", "safetensors>=0.4.0"],
        "peft":     ["peft>=0.8.0", "transformers>=4.37.0"],
        "dev":      ["pytest", "huggingface_hub>=0.20.0", "safetensors>=0.4.0"],
    },
)
