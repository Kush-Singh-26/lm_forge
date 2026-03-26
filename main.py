import sys
from pathlib import Path

# Set up project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print("Welcome to lm_forge (HF Native)!")
    print("Run the native HF example with:")
    print("  python experiments/hf_native_example/train.py --steps 100 --cpu")


if __name__ == "__main__":
    main()
