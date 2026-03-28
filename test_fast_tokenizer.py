from engine.tokenizer.bpe import BPETokenizer
import os


def test_fast_tokenizer():
    texts = [
        "Hello world!",
        "Hello there, how are you?",
        "This is a test corpus for BPE.",
    ]

    # 1. Train
    tok = BPETokenizer()
    tok.train(texts, vocab_size=300)
    print(f"Vocab size after training: {len(tok)}")

    # 2. Encode
    encoded = tok.encode("Hello world!")
    print(f"Encoded: {encoded}")

    # 3. Decode
    decoded = tok.decode(encoded)
    print(f"Decoded: '{decoded}'")
    assert decoded.strip() == "Hello world!"

    # 4. Save/Load
    save_path = "test_tokenizer.json"
    tok.save(save_path)
    tok2 = BPETokenizer.load(save_path)

    encoded2 = tok2.encode("Hello world!")
    assert encoded == encoded2
    print("Save/Load verified.")

    # 5. Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
    print("Test passed!")


if __name__ == "__main__":
    test_fast_tokenizer()
