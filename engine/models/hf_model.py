"""
engine/models/hf_model.py

HFCausalLM — The primary model class for lm_forge, inheriting from HF's PreTrainedModel.
This provides native support for the entire Hugging Face ecosystem.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from engine.config.hf_config import LMForgeConfig
from engine.models.decoder import CausalLM


class HFCausalLM(PreTrainedModel, GenerationMixin):
    """
    Native HF-compatible model. Use this for training with HF Trainer,
    saving/loading via .from_pretrained(), and pushing to the Hub.
    """

    config_class = LMForgeConfig
    base_model_prefix = "lm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _tied_weights_keys = {"lm.lm_head.weight": "lm.model.embed_tokens.weight"}

    def __init__(self, config: LMForgeConfig) -> None:
        super().__init__(config)
        model_cfg = config.to_model_config()
        self.lm = CausalLM(model_cfg)
        self.post_init()



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", True)
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else getattr(self.config, "use_cache", True)
        )
        # KV caching is incompatible with gradient checkpointing and wastes
        # VRAM during training — always disable it.
        if self.training:
            use_cache = False

        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds is not supported. Pass input_ids.")

        logits, loss, presents = self.lm(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if not return_dict:
            out = (logits,)
            if presents is not None:
                out = out + (presents,)
            if loss is not None:
                out = (loss,) + out
            return out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.lm.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.lm.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm.lm_head = new_embeddings

    def tie_weights(self, **kwargs) -> None:
        if self.config.tie_word_embeddings:
            self.lm.lm_head.weight = self.lm.model.embed_tokens.weight

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # Convert HF's DynamicCache to legacy tuple format if needed
        if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
            try:
                legacy_cache = past_key_values.to_legacy_cache()
            except Exception:
                legacy_cache = past_key_values
            # Check if cache is populated — shape check handles both None and
            # empty tensors with shape (batch, heads, 0, head_dim) in recent transformers
            cache_empty = all(
                layer_cache is None
                or (layer_cache[0] is None and layer_cache[1] is None)
                or (layer_cache[0].shape[2] == 0 and layer_cache[1].shape[2] == 0)
                for layer_cache in legacy_cache
            )
            if cache_empty:
                # Cache not yet populated, treat as no cache
                past_key_values = None
            else:
                past_key_values = legacy_cache

        # only last token for inputs_ids if past is defined
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # For generation, we retain the full attention_mask (length = kv_len)
        # to ensure the model doesn't attend to padding tokens during decoding.

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
            )
        return reordered_past

    @classmethod
    def register(cls):
        """Register with HF Auto classes for seamless AutoModel loading."""
        AutoConfig.register("lm_forge", LMForgeConfig)
        AutoModelForCausalLM.register(LMForgeConfig, cls)
