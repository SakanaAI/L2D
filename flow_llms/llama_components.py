import copy
import math
import numpy as np
from dataclasses import dataclass
import typing as tp
from transformers.cache_utils import Cache, DynamicCache, StaticCache

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaModel,
    LlamaDecoderLayer,
    AttentionMaskConverter,
)

import hydra
from omegaconf import DictConfig


from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,)


from transformers.generation.utils import *
from .base import FlowLanguageModel, FrequencyEmbedder, ConditioningHead
from .utils import apply_rotary_to_vector
from .base_components import BatchedLinear, LoRA


class FlowLlamaRMSNorm(LlamaRMSNorm):

    def __init__(
            self,
            hidden_size: int,
            freeze_modulation_at_flow_start: bool,
            eps=1e-6,
    ):

        LlamaRMSNorm.__init__(self=self, hidden_size=hidden_size, eps=eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.freeze_modulation_at_flow_start = freeze_modulation_at_flow_start

    def forward(
            self,
            hidden_states,

            modulate_weights: bool = False,

            modulation_weight: tp.Optional[torch.Tensor] = None,

            modulation_weight_at_flow_start: tp.Optional[torch.Tensor] = None,
    ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon)

        if modulate_weights:
            weight = self.weight + modulation_weight
            if self.freeze_modulation_at_flow_start:

                weight = weight - modulation_weight_at_flow_start
        else:
            weight = self.weight

        return weight*hidden_states.to(input_dtype)


class FlowLlamaMLP(LlamaMLP):
    def __init__(self,
                 config,
                 memory_efficient: bool = True,
                 separate_flow_params: bool = False,
                 separate_flow_params_with_lora: bool = False,
                 flow_lora_rank: int = 8,
                 flow_lora_alpha: tp.Optional[float] = None,
                 ):
        LlamaMLP.__init__(self=self, config=config)
        self.memory_efficient = memory_efficient
        self.separate_flow_params = separate_flow_params
        self.separate_flow_params_with_lora = separate_flow_params_with_lora
        self.flow_lora_rank = flow_lora_rank
        self.flow_lora_alpha = flow_lora_alpha

        if self.separate_flow_params:
            if self.separate_flow_params_with_lora:
                self.flow_gate_proj = LoRA(
                    self.hidden_size, self.intermediate_size,
                    wrapped_layer=self.gate_proj,
                    rank=self.flow_lora_rank, alpha=self.flow_lora_alpha)
                self.flow_up_proj = LoRA(
                    self.hidden_size, self.intermediate_size,
                    wrapped_layer=self.up_proj,
                    rank=self.flow_lora_rank, alpha=self.flow_lora_alpha)
                self.flow_down_proj = LoRA(
                    self.intermediate_size, self.hidden_size,
                    wrapped_layer=self.down_proj,
                    rank=self.flow_lora_rank, alpha=self.flow_lora_alpha)
            else:
                self.flow_gate_proj = nn.Linear(
                    self.hidden_size, self.intermediate_size,
                    bias=config.mlp_bias)
                self.flow_up_proj = nn.Linear(
                    self.hidden_size, self.intermediate_size,
                    bias=config.mlp_bias)
                self.flow_down_proj = nn.Linear(
                    self.intermediate_size, self.hidden_size,
                    bias=config.mlp_bias)
        else:
            self.flow_gate_proj = self.gate_proj
            self.flow_up_proj = self.up_proj
            self.flow_down_proj = self.down_proj
        if not self.memory_efficient:
            assert not separate_flow_params

    def mlp_forward(self, x, gate_proj, up_proj, down_proj):

        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = gate_proj.weight.split(slice, dim=0)
            up_proj_slices = up_proj.weight.split(slice, dim=0)
            down_proj_slices = down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(
                    self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(
                self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(
                slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = down_proj(self.act_fn(gate_proj(x)) * up_proj(x))

        return down_proj

    def forward(self,
                x,
                flow_hidden_states: Optional[torch.Tensor] = None,
                use_flow: bool = False,
                compute_only_flow: bool = False,
                ):
        if use_flow or compute_only_flow:
            if self.memory_efficient or compute_only_flow:
                if compute_only_flow:
                    x_out = None
                else:
                    x_out = self.mlp_forward(
                        x=x,
                        gate_proj=self.gate_proj,
                        up_proj=self.up_proj,
                        down_proj=self.down_proj,
                    )
                flow_hidden_states = self.mlp_forward(
                    x=flow_hidden_states,
                    gate_proj=self.flow_gate_proj,
                    up_proj=self.flow_up_proj,
                    down_proj=self.flow_down_proj,
                )
            else:

                x = torch.cat([x, flow_hidden_states], dim=0)
                x_out, flow_hidden_states = torch.chunk(
                    input=x, chunks=2, dim=0)
            return x_out, flow_hidden_states
        else:
            x_out = LlamaMLP.forward(self=self, x=x)
            return x_out, flow_hidden_states


class FlowLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(
            self,
            config: LlamaConfig,
            layer_idx: Optional[int] = None,
            memory_efficient: bool = True,
            separate_flow_params: bool = False,
            separate_flow_params_with_lora: bool = False,
            flow_lora_rank: int = 8,
            flow_lora_alpha: tp.Optional[float] = None,
    ):
        LlamaAttention.__init__(self=self, config=config, layer_idx=layer_idx)
        self.memory_efficient = memory_efficient
        self.separate_flow_params = separate_flow_params
        self.separate_flow_params_with_lora = separate_flow_params_with_lora
        self.flow_lora_rank = flow_lora_rank
        self.flow_lora_alpha = flow_lora_alpha
        if self.memory_efficient:
            if layer_idx == 0:
                print('WARNING: the flow_hidden_states do not attend to'
                      ' themselves in the current memory efficient'
                      ' implementation')
        else:

            raise NotImplementedError

        if separate_flow_params:
            if self.separate_flow_params_with_lora:
                self.flow_q_proj = LoRA(
                    self.hidden_size,
                    self.num_heads*self.head_dim,
                    wrapped_layer=self.q_proj,
                    rank=self.flow_lora_rank, alpha=self.flow_lora_alpha,
                )
                self.flow_o_proj = LoRA(
                    self.num_heads*self.head_dim,
                    self.hidden_size,
                    wrapped_layer=self.o_proj,
                    rank=self.flow_lora_rank, alpha=self.flow_lora_alpha,
                )
            else:
                self.flow_q_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads*self.head_dim,
                    bias=config.attention_bias
                )
                self.flow_o_proj = nn.Linear(
                    self.num_heads*self.head_dim,
                    self.hidden_size,
                    bias=config.attention_bias
                )
        else:
            self.flow_q_proj = self.q_proj
            self.flow_o_proj = self.o_proj

    def forward_only_flow(
        self,

        flow_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Cache = None,
        output_attentions: bool = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]] = None,
    ):

        if output_attentions:
            raise NotImplementedError

        bsz, q_len, _ = flow_hidden_states.size()

        flow_query_states = self.flow_q_proj(flow_hidden_states)
        flow_query_states = flow_query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(flow_query_states, position_ids)
        else:
            cos, sin = position_embeddings
        flow_query_states = apply_rotary_to_vector(
            v=flow_query_states, cos=cos, sin=sin)

        key_states, value_states = past_key_value[self.layer_idx]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if flow_query_states.device.type == "cuda" and causal_mask is not None:
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            flow_query_states = flow_query_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        flow_attn_output = torch.nn.functional.scaled_dot_product_attention(
            flow_query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        flow_attn_output = flow_attn_output.transpose(1, 2).contiguous()
        flow_attn_output = flow_attn_output.view(bsz, q_len, -1)
        flow_attn_output = self.flow_o_proj(flow_attn_output)

        return None, flow_attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        flow_hidden_states: Optional[torch.Tensor] = None,
        use_flow: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]] = None,
        compute_only_flow: bool = False,
        **kwargs,
    ) -> Tuple[
            torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            raise NotImplementedError

        if compute_only_flow:
            return self.forward_only_flow(

                flow_hidden_states=flow_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if use_flow:
            bsz, flow_q_len, _ = flow_hidden_states.size()

            flow_query_states = self.flow_q_proj(flow_hidden_states)
            flow_query_states = flow_query_states.view(
                bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            q=query_states, k=key_states, cos=cos, sin=sin)

        if use_flow:
            flow_query_states = apply_rotary_to_vector(
                v=flow_query_states, cos=cos, sin=sin)
        if past_key_value is not None:

            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}

            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            if use_flow:
                flow_query_states = flow_query_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if use_flow:
            flow_attn_output = torch.nn.functional.scaled_dot_product_attention(
                flow_query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            flow_attn_output = flow_attn_output.transpose(1, 2).contiguous()
            flow_attn_output = flow_attn_output.view(bsz, q_len, -1)

            flow_attn_output = self.flow_o_proj(flow_attn_output)
        else:
            flow_attn_output = None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, flow_attn_output, None, past_key_value


FLOW_LLAMA_ATTENTION_CLASSES = {


    "sdpa": FlowLlamaSdpaAttention,
}


class FlowLlamaDecoderLayer(nn.Module):
    def __init__(
            self,
            config: LlamaConfig,
            layer_idx: int,
            modulate_hidden_states: bool,
            full_dit_modulation: bool,
            timestep_modulation_hidden_size: int,
            freeze_modulation_at_flow_start: bool,
            memory_efficient_implementation: bool = True,
            separate_flow_params: bool = False,
            separate_flow_params_with_lora: bool = False,
            flow_lora_rank: int = 8,
            flow_lora_alpha: tp.Optional[float] = None,
    ):
        nn.Module.__init__(self=self,)
        self.hidden_size = config.hidden_size

        self.modulate_hidden_states = modulate_hidden_states
        self.full_dit_modulation = full_dit_modulation
        self.freeze_modulation_at_flow_start = freeze_modulation_at_flow_start

        self.self_attn: FlowLlamaSdpaAttention = FLOW_LLAMA_ATTENTION_CLASSES[
            config._attn_implementation](
                config=config,
                layer_idx=layer_idx,
                memory_efficient=memory_efficient_implementation,
                separate_flow_params=separate_flow_params,
                separate_flow_params_with_lora=separate_flow_params_with_lora,
                flow_lora_rank=flow_lora_rank,
                flow_lora_alpha=flow_lora_alpha,
        )

        self.mlp = FlowLlamaMLP(
            config,
            memory_efficient=memory_efficient_implementation,
            separate_flow_params=separate_flow_params,
            separate_flow_params_with_lora=separate_flow_params_with_lora,
            flow_lora_rank=flow_lora_rank,
            flow_lora_alpha=flow_lora_alpha,
        )

        self.input_layernorm = FlowLlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            freeze_modulation_at_flow_start=freeze_modulation_at_flow_start,
        )
        self.post_attention_layernorm = FlowLlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            freeze_modulation_at_flow_start=freeze_modulation_at_flow_start,
        )

        if full_dit_modulation:
            num_hidden_components = 6
        else:

            num_hidden_components = 2

        self.modulation_head = ConditioningHead(
            hidden_dim=config.hidden_size,
            num_hidden_components=num_hidden_components,
            num_layers=2,
            conditioning_dim=timestep_modulation_hidden_size,
        )

    def init_flow_weights(self,):
        def _basic_init(module):
            if isinstance(module, LoRA):
                nn.init.normal_(module.A, std=module.rescale_stddev.item())
                nn.init.zeros_(module.B)
            elif isinstance(module, (nn.Linear, BatchedLinear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.modulation_head.apply(_basic_init)
        nn.init.constant_(self.modulation_head.mlp[-1].weight, 0)
        nn.init.constant_(self.modulation_head.mlp[-1].bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        flow_hidden_states: torch.Tensor,


        use_flow: bool,
        modulation_embeddings: torch.Tensor,
        modulation_embeddings_at_flow_start: tp.Optional[torch.Tensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.LongTensor] = None,
        past_key_value: tp.Optional[Cache] = None,
        output_attentions: tp.Optional[bool] = False,
        use_cache: tp.Optional[bool] = False,
        cache_position: tp.Optional[torch.LongTensor] = None,
        position_embeddings: tp.Optional[
            tp.Tuple[torch.Tensor, torch.Tensor]] = None,
        compute_only_flow: bool = False,
        **kwargs,
    ) -> tp.Tuple[torch.FloatTensor,
                  tp.Optional[tp.Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        modulate_hidden_states = (
            self.modulate_hidden_states and use_flow and
            (not compute_only_flow)
        )
        use_flow = use_flow or compute_only_flow

        if not compute_only_flow:
            residual = hidden_states
        if use_flow:
            flow_residual = flow_hidden_states
            modulation_weights = self.modulation_head(modulation_embeddings)
            if self.full_dit_modulation:
                (input_mod_weight, input_mod_shift, input_mod_rescale,
                 post_attn_mod_weight, post_attn_mod_shift,
                 post_attn_mod_rescale) = torch.chunk(
                     modulation_weights, chunks=6, dim=-1)
            else:
                input_mod_weight, post_attn_mod_weight = torch.chunk(
                    modulation_weights, chunks=2, dim=-1)
        else:
            input_mod_weight = None
            post_attn_mod_weight = None

        if use_flow and self.freeze_modulation_at_flow_start:
            mod_weights_at_flow_start = self.modulation_head(
                modulation_embeddings_at_flow_start)
            if self.full_dit_modulation:
                (input_mod_weight_at_flow_start,
                 input_mod_shift_at_flow_start,
                 input_mod_rescale_at_flow_start,
                 post_attn_mod_weight_at_flow_start,
                 post_attn_mod_shift_at_flow_start,
                 post_attn_mod_rescale_at_flow_start,) = torch.chunk(
                    mod_weights_at_flow_start, chunks=6, dim=-1)
            else:
                (input_mod_weight_at_flow_start,
                 post_attn_mod_weight_at_flow_start) = torch.chunk(
                    mod_weights_at_flow_start, chunks=2, dim=-1)
        else:
            input_mod_weight_at_flow_start = None
            post_attn_mod_weight_at_flow_start = None

        if not compute_only_flow:
            hidden_states = self.input_layernorm(
                hidden_states=hidden_states,
                modulate_weights=modulate_hidden_states,
                modulation_weight=input_mod_weight,
                modulation_weight_at_flow_start=(
                    input_mod_weight_at_flow_start),
            )

        if use_flow:
            flow_hidden_states = self.input_layernorm(
                hidden_states=flow_hidden_states,
                modulate_weights=True,
                modulation_weight=input_mod_weight,
                modulation_weight_at_flow_start=(
                    input_mod_weight_at_flow_start),
            )
            if self.full_dit_modulation:
                if self.freeze_modulation_at_flow_start:
                    shift = input_mod_shift - input_mod_shift_at_flow_start
                    rescale = (1.0 + input_mod_rescale -
                               input_mod_rescale_at_flow_start)
                else:
                    shift = input_mod_shift
                    rescale = input_mod_rescale
                flow_hidden_states = flow_hidden_states + shift
                if modulate_hidden_states:
                    hidden_states = hidden_states + shift

        (hidden_states, flow_hidden_states, self_attn_weights,
         present_key_value) = self.self_attn(
            hidden_states=hidden_states,
            flow_hidden_states=flow_hidden_states,
            use_flow=use_flow,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            compute_only_flow=compute_only_flow,
            **kwargs,
        )

        if use_flow:
            if self.full_dit_modulation:
                flow_hidden_states = flow_hidden_states*rescale
                if modulate_hidden_states:
                    hidden_states = hidden_states*rescale
            flow_hidden_states = flow_residual + flow_hidden_states

        if not compute_only_flow:
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(
                hidden_states=hidden_states,
                modulate_weights=modulate_hidden_states,
                modulation_weight=post_attn_mod_weight,
                modulation_weight_at_flow_start=(
                    post_attn_mod_weight_at_flow_start),
            )

        if use_flow:
            flow_residual = flow_hidden_states
            flow_hidden_states = self.post_attention_layernorm(
                hidden_states=flow_hidden_states,
                modulate_weights=True,
                modulation_weight=post_attn_mod_weight,
                modulation_weight_at_flow_start=(
                    post_attn_mod_weight_at_flow_start),
            )
            if self.full_dit_modulation:
                if self.freeze_modulation_at_flow_start:
                    shift = (post_attn_mod_shift -
                             post_attn_mod_shift_at_flow_start)
                    rescale = (1.0 + post_attn_mod_rescale -
                               post_attn_mod_rescale_at_flow_start)
                else:
                    shift = post_attn_mod_shift
                    rescale = post_attn_mod_rescale
                flow_hidden_states = flow_hidden_states + shift
                if modulate_hidden_states:
                    hidden_states = hidden_states + shift
        hidden_states, flow_hidden_states = self.mlp(
            x=hidden_states,
            flow_hidden_states=flow_hidden_states,
            use_flow=use_flow,
            compute_only_flow=compute_only_flow,
        )
        if use_flow:
            if self.full_dit_modulation:
                flow_hidden_states = flow_hidden_states*rescale
                if modulate_hidden_states:
                    hidden_states = hidden_states*rescale
            flow_hidden_states = flow_residual + flow_hidden_states

        if not compute_only_flow:
            hidden_states = residual + hidden_states

        outputs = (hidden_states, flow_hidden_states)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
