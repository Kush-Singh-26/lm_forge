import torch
from engine.components.positional.alibi import ALiBi
from engine.config.schema import PositionalConfig

cfg = PositionalConfig(max_seq_len=2048)
alibi = ALiBi(cfg, num_heads=8)

# Simulating decoder.py passing kv_len
hs = torch.zeros(1, 1, 512)
pe_out = alibi(hs, seq_len=513) # past_key_values of 512 + 1 new token

print("Attn bias shape:", pe_out.attn_bias.shape)
assert pe_out.attn_bias.shape == (1, 8, 513, 513)

pe_out = alibi(hs, seq_len=3000) # testing expansion
print("Expanded attn bias shape:", pe_out.attn_bias.shape)
assert pe_out.attn_bias.shape == (1, 8, 3000, 3000)
print("ALiBi forward test PASSED!")
