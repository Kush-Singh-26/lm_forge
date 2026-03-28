"""
engine/models/decoder.py

Decoder-only Causal LM — assembled from registry components.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from engine.config.schema import ModelConfig
from engine.components.norm import build_norm
from engine.components.positional import build_pe
from engine.components.attention import build_attention
from engine.components.ffn import build_ffn
from engine.models.base import BaseLM


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.post_attention_layernorm = build_norm(
            cfg.norm.type, cfg.hidden_size, cfg.norm.eps
        )
        self.self_attn = build_attention(cfg)
        self.mlp = build_ffn(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe_out,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Union[Tuple, object]] = None,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden_states),
            pe_out=pe_out,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            layer_idx=layer_idx,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present


class DecoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pe = build_pe(
            cfg.positional,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.attention.num_heads,
        )
        if hasattr(self.pe, "set_head_dim"):
            self.pe.set_head_dim(cfg.head_dim)

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = build_norm(cfg.norm.type, cfg.hidden_size, cfg.norm.eps)
        self.gradient_checkpointing = False
        self._mask_cache: dict = {}

    def reset_cache(self) -> None:
        self._mask_cache.clear()

    def _get_causal_mask(
        self, S: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Returns a (S, S) causal mask (upper triangular with -inf).
        Caches a max-length mask to avoid thrashing.
        """
        norm_device = str(torch.device(device.type, device.index or 0))
        # Use a large enough default or the model's max_seq_len
        max_len = max(S, self.cfg.max_seq_len)
        key = ("causal", max_len, dtype, norm_device)

        if key not in self._mask_cache:
            # Clear if we have too many devices/dtypes cached (rare)
            if len(self._mask_cache) > 8:
                self._mask_cache.clear()

            mask = torch.full(
                (max_len, max_len),
                torch.finfo(dtype).min / 2,
                device=device,
                dtype=dtype,
            )
            mask = torch.triu(mask, diagonal=1)
            self._mask_cache[key] = mask

        return self._mask_cache[key][:S, :S]

    def _get_sliding_window_mask(
        self, S: int, kv_len: int, window: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Returns a (S, kv_len) sliding window mask.
        """
        norm_device = str(torch.device(device.type, device.index or 0))
        # Cache by kv_len and window
        key = ("sliding", kv_len, window, dtype, norm_device)

        if key not in self._mask_cache:
            if len(self._mask_cache) > 8:
                self._mask_cache.clear()

            # Build the full kv_len x kv_len mask
            i = torch.arange(kv_len, device=device).unsqueeze(1)
            j = torch.arange(kv_len, device=device).unsqueeze(0)
            in_window = (j > i - window) & (j <= i)
            mask = torch.full(
                (kv_len, kv_len),
                torch.finfo(dtype).min / 2,
                device=device,
                dtype=dtype,
            )
            mask = mask.masked_fill(in_window, 0.0)
            self._mask_cache[key] = mask

        return self._mask_cache[key][-S:, :]

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[List, object]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Union[List, object]]]:
        B, S = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        kv_len = S
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                kv_len = past_key_values.get_seq_length() + S
            elif isinstance(past_key_values, list):
                kv_len += past_key_values[0][0].shape[2]

        pe_out = self.pe(hidden_states, seq_len=kv_len, position_ids=position_ids)
        if pe_out.hidden_states is not None:
            hidden_states = pe_out.hidden_states

        # ── Attention Masking ──────────────────────────────────────────
        # Optimized SDPA Path: Pass None and set is_causal=True inside attention
        # whenever we don't have padding OR ALiBi bias OR sliding window.
        additive = None
        has_alibi = pe_out.attn_bias is not None
        has_padding = attention_mask is not None
        is_sliding = self.cfg.attention.type == "sliding"

        # 1. Determine if we need to materialize a mask.
        # We MUST materialize if we have ALiBi (to add it later) OR if we have padding
        # OR if we are using sliding window (since SDPA doesn't support it).
        if has_alibi or has_padding or is_sliding:
            if is_sliding:
                additive = self._get_sliding_window_mask(
                    S,
                    kv_len,
                    self.cfg.attention.window_size,
                    hidden_states.device,
                    hidden_states.dtype,
                )
                additive = additive.view(1, 1, S, kv_len)
            elif S > 1:
                if past_key_values is None:
                    additive = self._get_causal_mask(
                        S, hidden_states.device, hidden_states.dtype
                    )
                    additive = additive.view(1, 1, S, S)
                else:
                    # (S, kv_len) mask: zeros followed by causal suffix
                    causal_suffix = self._get_causal_mask(
                        S, hidden_states.device, hidden_states.dtype
                    )
                    additive = torch.zeros(
                        (S, kv_len),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    additive[:, -S:] = causal_suffix
                    additive = additive.view(1, 1, S, kv_len)
            else:
                # S = 1, just zeros (1, 1, 1, kv_len) to be filled with padding mask if needed
                additive = torch.zeros(
                    1,
                    1,
                    1,
                    kv_len,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

            # 2. Add padding mask if present
            if has_padding:
                fill_value = torch.finfo(hidden_states.dtype).min / 2
                pad_mask = (attention_mask == 0).view(B, 1, 1, kv_len)
                # pad_mask is (B, 1, 1, kv_len), additive is (1, 1, S/1, kv_len)
                # Broadcasting handles the batch and sequence dimension.
                additive = additive.masked_fill(pad_mask, fill_value)

        # ── Execution ──────────────────────────────────────────────────
        presents = (
            []
            if (use_cache and past_key_values is None)
            or isinstance(past_key_values, list)
            else None
        )

        for i, layer in enumerate(self.layers):
            if isinstance(past_key_values, list):
                pkv = past_key_values[i]
            else:
                # Cache object or None
                pkv = past_key_values

            if self.gradient_checkpointing and self.training and pkv is None:

                def _inner(hs, mask, layer=layer, pe_out_captured=pe_out):
                    out, _ = layer(
                        hs,
                        pe_out=pe_out_captured,
                        attention_mask=mask,
                        past_key_value=None,
                        use_cache=False,
                    )
                    return out

                hidden_states = torch.utils.checkpoint.checkpoint(
                    _inner, hidden_states, additive, use_reentrant=False
                )
                present = None
            else:
                hidden_states, present = layer(
                    hidden_states,
                    pe_out=pe_out,
                    attention_mask=additive,
                    past_key_value=pkv,
                    use_cache=use_cache,
                    layer_idx=i,
                )
            if presents is not None:
                presents.append(present)

        final_presents = presents if presents is not None else past_key_values

        return self.norm(hidden_states), final_presents


class CausalLM(BaseLM):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.model = DecoderModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tie_weights()
        self.post_init()

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[List, object]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Union[List, object]]]:
        hidden, presents = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache
        )
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, presents

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self.eval()
        B, S_start = input_ids.shape
        max_total = S_start + max_new_tokens

        # Pre-allocate buffer for better performance
        # We'll fill this as we go.
        output_ids = torch.full(
            (B, max_total),
            self.config.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output_ids[:, :S_start] = input_ids

        past = []
        S_curr = S_start

        for i in range(max_new_tokens):
            cur = (
                output_ids[:, :S_curr]
                if not past
                else output_ids[:, S_curr - 1 : S_curr]
            )

            logits, _, past = self(cur, past_key_values=past or None, use_cache=True)
            next_logits = logits[:, -1] / max(temperature, 1e-8)

            if top_k > 0:
                vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < vals[:, -1:]] = float("-inf")

            tok = torch.multinomial(
                torch.softmax(next_logits, -1), 1, generator=generator
            )

            output_ids[:, S_curr] = tok.squeeze(-1)
            S_curr += 1

            if eos_token_id is not None and (tok == eos_token_id).all():
                break

        return output_ids[:, :S_curr]
