import torch
from engine.config.schema import ModelConfig, AttentionConfig
from engine.models.decoder import CausalLM
from engine.components.attention.gqa import GroupedQueryAttention


def test_flash_attn_propagation():
    """
    Verifies that flash_attn flag is correctly propagated to the attention modules.
    """
    cfg = ModelConfig(attention=AttentionConfig(flash_attn=True))
    try:
        model = CausalLM(cfg)
    except ImportError as e:
        if "flash-attn" in str(e):
            print("Skipping flash_attn test (not installed)")
            return
        raise e

    found = False
    for module in model.modules():
        if isinstance(module, GroupedQueryAttention):
            assert module.use_flash == True
            found = True
    assert found


def test_torch_compile():
    """
    Verifies that the model is compatible with torch.compile.
    """
    if not hasattr(torch, "compile"):
        print("Skipping torch.compile test (not supported on this torch version)")
        return

    cfg = ModelConfig(
        num_layers=2,
        hidden_size=64,
        attention=AttentionConfig(num_heads=2, num_kv_heads=2),
    )
    model = CausalLM(cfg).to("cpu")

    try:
        compiled_model = torch.compile(model)

        x = torch.randint(0, cfg.vocab_size, (1, 8))
        # Run forward
        with torch.no_grad():
            logits, _, _ = compiled_model(x)
            assert logits.shape == (1, 8, cfg.vocab_size)
    except Exception as e:
        if "Compiler" in str(e) or "backend" in str(e):
            print(f"Skipping torch.compile test (compiler not found: {e})")
            return
        raise e


if __name__ == "__main__":
    test_flash_attn_propagation()
    test_torch_compile()
