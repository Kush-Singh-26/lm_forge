"""
engine/models/hf_model.py

HFCausalLM — The primary model class for lm_forge, inheriting from HF's PreTrainedModel.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, List

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
    Native HF-compatible model. Supports transformers.Cache (v5 compatible).
    """

    config_class = LMForgeConfig
    base_model_prefix = "lm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _tied_weights_keys = {"lm.lm_head.weight": "lm.model.embed_tokens.weight"}
    _keys_to_ignore_on_load_missing = [r"lm\.lm_head\.weight"]

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
        past_key_values: Optional[Union[List, object]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.training:
            use_cache = False

        if inputs_embeds is not None:
            raise ValueError(
                "inputs_embeds is not supported by lm_forge. Pass input_ids."
            )

        logits, loss, presents = self.lm(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if not return_dict:
            output = (logits,)
            if presents is not None:
                output += (presents,)
            if loss is not None:
                output = (loss,) + output
            return output

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
        self.lm.tie_weights()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm.model.enable_gradient_checkpointing()

    def gradient_checkpointing_disable(self):
        self.lm.model.gradient_checkpointing = False

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, list):
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = past_key_values[0].shape[2]

            # Only last token if past is defined
            if input_ids.shape[1] > 1:
                input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if isinstance(past_key_values, list):
            reordered_past = []
            for layer_past in past_key_values:
                reordered_past.append(
                    tuple(
                        past_state.index_select(0, beam_idx.to(past_state.device))
                        for past_state in layer_past
                    )
                )
            return reordered_past
        elif hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        return past_key_values

    @classmethod
    def register(cls):
        AutoConfig.register("lm_forge", LMForgeConfig)
        AutoModelForCausalLM.register(LMForgeConfig, cls)
