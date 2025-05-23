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

from torchdiffeq import odeint

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
from .llama_components import (
    FlowLlamaDecoderLayer, FlowLlamaMLP, FlowLlamaSdpaAttention,
    FlowLlamaRMSNorm
)
from .base_components import (
    SimpleMLP, BatchedLinear,
    FlowCausalLMOutputWithPast,
    FlowModelOutputWithPast,
    FlowInferenceOutput,
)
from .utils import *


class FlowLlamaModel(LlamaModel, FlowLanguageModel):
    def __init__(
            self,
            config: LlamaConfig,



            flow_representation_space: str = 'mapping',

            flow_representation_dim: tp.Optional[int] = None,
            flow_representation_num_layers: tp.Optional[int] = None,

            flow_representation_normalize: bool = True,

            flow_representation_rescaling: tp.Optional[
                tp.Literal["div", "mult", "none"]] = None,



            noise_rescaling: float = 1.0,



            flow_to_lm_translation_depth: tp.Optional[int] = None,
            flow_to_lm_hidden_size: tp.Optional[int] = None,


            flow_to_lm_timestep_rescaling: tp.Optional[float] = None,


            flow_to_lm_rescale_in_float32: bool = False,



            preserve_behavior_at_flow_start: bool = True,

            modulate_hidden_states: bool = False,
            full_dit_modulation: bool = False,
            timestep_modulation_num_layers: int = 2,
            timestep_modulation_freq_embedding_size: int = 256,

            timestep_modulation_hidden_size: tp.Optional[int] = None,



            guidance_modulation_num_classes: tp.Optional[int] = None,


            guidance_modulation_training_dropout: float = 0.2,



            freeze_modulation_at_flow_start: bool = False,


            separate_flow_params: bool = False,


            separate_flow_params_with_lora: bool = False,
            flow_lora_rank: int = 8,

            flow_lora_alpha: tp.Optional[float] = None,


            ode_kwargs: tp.Optional[dict] = dict(
                atol=1e-5,
                rtol=1e-5,
                method='midpoint'
            ),



            nstep_final_timestep: float = 1.0,



            nstep_x1_estimation: tp.Literal["average", "sample"] = "average",


            nstep_normalize_x1_predictions: bool = False,


            nstep_clamp_predictions: bool = False,


            nstep_temperature_schedule: tp.Optional[
                float | tp.Callable | str] = None,


            nstep_guidance_parameter: tp.Optional[float] = None,
    ):
        LlamaPreTrainedModel.__init__(self, config)
        self.main_model_hidden_dim = config.hidden_size

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        FlowLanguageModel.__init__(
            self=self,
            config=config,
            main_model_hidden_dim=self.main_model_hidden_dim,
            flow_representation_space=flow_representation_space,
            flow_representation_dim=flow_representation_dim,
            flow_representation_num_layers=flow_representation_num_layers,
            flow_representation_normalize=flow_representation_normalize,
            flow_representation_rescaling=flow_representation_rescaling,
            noise_rescaling=noise_rescaling,
            flow_to_lm_translation_depth=flow_to_lm_translation_depth,
            flow_to_lm_hidden_size=flow_to_lm_hidden_size,
            flow_to_lm_timestep_rescaling=flow_to_lm_timestep_rescaling,
            flow_to_lm_rescale_in_float32=flow_to_lm_rescale_in_float32,
            preserve_behavior_at_flow_start=preserve_behavior_at_flow_start,
            modulate_hidden_states=modulate_hidden_states,
            full_dit_modulation=full_dit_modulation,
            timestep_modulation_num_layers=timestep_modulation_num_layers,
            timestep_modulation_freq_embedding_size=(
                timestep_modulation_freq_embedding_size),
            timestep_modulation_hidden_size=timestep_modulation_hidden_size,
            guidance_modulation_num_classes=guidance_modulation_num_classes,
            guidance_modulation_training_dropout=(
                guidance_modulation_training_dropout),
            freeze_modulation_at_flow_start=freeze_modulation_at_flow_start,
            separate_flow_params=separate_flow_params,
            separate_flow_params_with_lora=separate_flow_params_with_lora,
            flow_lora_rank=flow_lora_rank,
            flow_lora_alpha=flow_lora_alpha,
            vocab_size=self.vocab_size,
            ode_kwargs=ode_kwargs,
            nstep_final_timestep=nstep_final_timestep,
            nstep_x1_estimation=nstep_x1_estimation,
            nstep_normalize_x1_predictions=nstep_normalize_x1_predictions,
            nstep_clamp_predictions=nstep_clamp_predictions,
            nstep_temperature_schedule=nstep_temperature_schedule,
            nstep_guidance_parameter=nstep_guidance_parameter,
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [FlowLlamaDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                modulate_hidden_states=modulate_hidden_states,
                full_dit_modulation=full_dit_modulation,
                timestep_modulation_hidden_size=(
                    self.timestep_modulation_hidden_size),
                freeze_modulation_at_flow_start=freeze_modulation_at_flow_start,
                separate_flow_params=separate_flow_params,
                separate_flow_params_with_lora=separate_flow_params_with_lora,
                flow_lora_rank=flow_lora_rank,
                flow_lora_alpha=flow_lora_alpha,
            )
                for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.modulation_head = ConditioningHead(
            hidden_dim=config.hidden_size,
            num_hidden_components=1,
            num_layers=2,
            conditioning_dim=self.timestep_modulation_hidden_size,
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        self.post_init()

    def init_flow_weights(self,):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, BatchedLinear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        FlowLanguageModel.init_flow_weights(self=self,)
        for decoder_layer in self.layers:
            decoder_layer.init_flow_weights()
        self.modulation_head.apply(_basic_init)
        nn.init.constant_(self.modulation_head.mlp[-1].weight, 0)
        nn.init.constant_(self.modulation_head.mlp[-1].bias, 0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timesteps: torch.FloatTensor = None,
        class_labels: torch.LongTensor = None,
        flow_representation_embeds: torch.FloatTensor = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.LongTensor] = None,
        past_key_values: tp.Optional[tp.Union[
            Cache, tp.List[torch.FloatTensor]]] = None,
        inputs_embeds: tp.Optional[torch.FloatTensor] = None,
        use_cache: tp.Optional[bool] = None,
        output_attentions: tp.Optional[bool] = None,
        output_hidden_states: tp.Optional[bool] = None,
        return_dict: tp.Optional[bool] = None,
        cache_position: tp.Optional[torch.LongTensor] = None,
        output_cached_final_hidden_states: bool = False,


        cached_final_hidden_states: tp.Optional[torch.FloatTensor] = None,
    ) -> tp.Union[tp.Tuple, BaseModelOutputWithPast]:

        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions)

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (use_cache if use_cache is not None
                     else self.config.use_cache)
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        if (input_ids is None) and (inputs_embeds is not None):
            raise ValueError()

        use_flow = flow_representation_embeds is not None
        compute_only_flow = cached_final_hidden_states is not None

        if compute_only_flow:
            assert past_key_values is not None and use_flow
            num_flow_embeds = flow_representation_embeds.shape[-2]

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:

            inputs_embeds = self.embed_tokens(input_ids)

        if self.use_guidance and (class_labels is None):

            class_labels = torch.zeros(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            ) + self.guidance_modulation_num_classes
        if use_flow:
            if self.flow_to_lm_rescale_timestep:
                flow_inputs_embeds = self.flow_to_lm_encoder(
                    x=flow_representation_embeds,
                    timesteps=timesteps)
            else:
                flow_inputs_embeds = self.flow_to_lm_encoder(
                    flow_representation_embeds)
        else:
            flow_inputs_embeds = None

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a Tuple "
                "and this is deprecated and will be removed in v4.43. "
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None
                else 0)

            if compute_only_flow:
                cache_position = torch.arange(
                    past_seen_tokens - num_flow_embeds, past_seen_tokens,
                    device=flow_representation_embeds.device
                )
            else:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens +
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device
                )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if compute_only_flow:
            assert position_ids.shape[-1] == num_flow_embeds
            causal_mask = self._update_causal_mask(
                attention_mask, flow_inputs_embeds, cache_position,
                past_key_values, output_attentions
            )

            position_embeddings = self.rotary_emb(
                flow_inputs_embeds, position_ids)
        else:
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values,
                output_attentions
            )

            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        flow_hidden_states = flow_inputs_embeds

        if use_flow:

            modulation_embeddings = self.timestep_embedder(timesteps).to(
                dtype=inputs_embeds.dtype)
            if self.use_guidance:
                class_embeddings = self.data_label_embedder(class_labels)
                modulation_embeddings = modulation_embeddings + class_embeddings
            if (self.freeze_modulation_at_flow_start or
                    self.preserve_behavior_at_flow_start):
                timesteps_at_flow_start = torch.zeros_like(timesteps)
                modulation_embeddings_at_flow_start = self.timestep_embedder(
                    timesteps_at_flow_start).to(dtype=inputs_embeds.dtype)
                if self.use_guidance:
                    modulation_embeddings_at_flow_start = (
                        modulation_embeddings_at_flow_start + class_embeddings)
            else:
                modulation_embeddings_at_flow_start = None
        else:
            modulation_embeddings = None
            modulation_embeddings_at_flow_start = None

        all_hidden_states = () if output_hidden_states else None
        all_flow_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                all_flow_hidden_states += (flow_hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    flow_hidden_states,
                    use_flow,
                    modulation_embeddings,
                    modulation_embeddings_at_flow_start,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    compute_only_flow,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    flow_hidden_states=flow_hidden_states,
                    use_flow=use_flow,
                    modulation_embeddings=modulation_embeddings,
                    modulation_embeddings_at_flow_start=(
                        modulation_embeddings_at_flow_start),
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    compute_only_flow=compute_only_flow,
                )

            hidden_states = layer_outputs[0]
            flow_hidden_states = layer_outputs[1]

            if use_cache:
                next_decoder_cache = layer_outputs[
                    3 if output_attentions else 2]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        if output_cached_final_hidden_states:
            if compute_only_flow:
                cached_final_hidden_states_to_return = (
                    cached_final_hidden_states)
            else:
                cached_final_hidden_states_to_return = hidden_states
        else:
            cached_final_hidden_states_to_return = None

        if use_flow:
            final_flow_rescaling = self.modulation_head(modulation_embeddings)
            if (self.freeze_modulation_at_flow_start or
                    self.preserve_behavior_at_flow_start):
                final_flow_rescale_at_flow_start = self.modulation_head(
                    modulation_embeddings_at_flow_start)
                final_flow_rescaling = (
                    final_flow_rescaling - final_flow_rescale_at_flow_start)
            flow_hidden_states = flow_hidden_states*final_flow_rescaling
            if compute_only_flow:

                hidden_states = cached_final_hidden_states + flow_hidden_states
            else:
                hidden_states = hidden_states + flow_hidden_states

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [
                hidden_states, next_cache, all_hidden_states, all_self_attns,
                cached_final_hidden_states_to_return] if v is not None)
        return FlowModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cached_final_hidden_states=cached_final_hidden_states_to_return,
        )


class FlowLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
            self,
            model_or_model_id: LlamaForCausalLM | str,
            base_params_to_freeze: tp.Optional[
                tp.Literal["all", "none"]] = None,
            model_loading_kwargs: tp.Optional[dict] = None,


            default_generation_nstep: tp.Optional[int] = None,
            noise_schedule: tp.Literal[
                "cosmap", "identity"] | Callable = "identity",

            minimum_training_noise: tp.Optional[float] = None,


            minimum_training_noise_units: tp.Literal[
                "time", "stddev"] | Callable = "time",




            velocity_schedule: tp.Literal[
                "constant", "edm"] | Callable = "constant",



            flow_representation_space: str = 'mapping',

            flow_representation_dim: tp.Optional[int] = None,
            flow_representation_num_layers: tp.Optional[int] = None,

            flow_representation_normalize: bool = True,

            flow_representation_rescaling: tp.Optional[
                tp.Literal["div", "mult", "none"]] = None,



            noise_rescaling: float = 1.0,



            flow_to_lm_translation_depth: tp.Optional[int] = None,
            flow_to_lm_hidden_size: tp.Optional[int] = None,


            flow_to_lm_timestep_rescaling: tp.Optional[float] = None,


            flow_to_lm_rescale_in_float32: bool = False,



            preserve_behavior_at_flow_start: bool = True,

            modulate_hidden_states: bool = False,
            full_dit_modulation: bool = False,
            timestep_modulation_num_layers: int = 2,
            timestep_modulation_freq_embedding_size: int = 256,

            timestep_modulation_hidden_size: tp.Optional[int] = None,



            guidance_modulation_num_classes: tp.Optional[int] = None,


            guidance_modulation_training_dropout: float = 0.2,


            freeze_modulation_at_flow_start: bool = False,


            separate_flow_params: bool = False,


            separate_flow_params_with_lora: bool = False,
            flow_lora_rank: int = 8,

            flow_lora_alpha: tp.Optional[float] = None,


            ode_kwargs: tp.Optional[dict] = dict(
                atol=1e-5,
                rtol=1e-5,
                method='midpoint'
            ),

            nstep_final_timestep: float = 1.0,



            nstep_x1_estimation: tp.Literal["average", "sample"] = "average",


            nstep_normalize_x1_predictions: bool = False,


            nstep_clamp_predictions: bool = False,


            nstep_temperature_schedule: tp.Optional[
                float | tp.Callable | str] = None,


            nstep_guidance_parameter: tp.Optional[float] = None,


            reinit_flow_params: bool = False,
    ):

        if model_loading_kwargs is None:
            model_loading_kwargs = {}

        if isinstance(model_or_model_id, str):
            if model_loading_kwargs is None:
                model_loading_kwargs = {}
            elif 'device_map' in model_loading_kwargs:
                assert model_loading_kwargs['device_map'] == 'cpu'
            model_loading_kwargs['device_map'] == 'cpu'
            model = LlamaForCausalLM.from_pretrained(
                model_or_model_id, **model_loading_kwargs)
        else:
            model: LlamaForCausalLM = model_or_model_id

        if base_params_to_freeze is None:
            base_params_to_freeze = 'none'
        self.base_params_to_freeze = base_params_to_freeze.lower()

        assert self.base_params_to_freeze in ['all', 'none']

        config: LlamaConfig = copy.deepcopy(model.config)
        LlamaPreTrainedModel.__init__(self, config)
        if timestep_modulation_freq_embedding_size is None:
            timestep_modulation_freq_embedding_size = self.config.hidden_size
        self.model = FlowLlamaModel(
            config=self.config,
            flow_representation_space=flow_representation_space,
            flow_representation_dim=flow_representation_dim,
            flow_representation_num_layers=flow_representation_num_layers,
            flow_representation_normalize=flow_representation_normalize,
            flow_representation_rescaling=flow_representation_rescaling,
            noise_rescaling=noise_rescaling,
            flow_to_lm_translation_depth=flow_to_lm_translation_depth,
            flow_to_lm_hidden_size=flow_to_lm_hidden_size,
            flow_to_lm_timestep_rescaling=flow_to_lm_timestep_rescaling,
            flow_to_lm_rescale_in_float32=flow_to_lm_rescale_in_float32,
            preserve_behavior_at_flow_start=preserve_behavior_at_flow_start,
            modulate_hidden_states=modulate_hidden_states,
            full_dit_modulation=full_dit_modulation,
            timestep_modulation_num_layers=timestep_modulation_num_layers,
            timestep_modulation_freq_embedding_size=(
                timestep_modulation_freq_embedding_size),
            timestep_modulation_hidden_size=timestep_modulation_hidden_size,
            guidance_modulation_num_classes=guidance_modulation_num_classes,
            guidance_modulation_training_dropout=(
                guidance_modulation_training_dropout),
            freeze_modulation_at_flow_start=freeze_modulation_at_flow_start,
            separate_flow_params=separate_flow_params,
            separate_flow_params_with_lora=separate_flow_params_with_lora,
            flow_lora_rank=flow_lora_rank,
            flow_lora_alpha=flow_lora_alpha,
            ode_kwargs=ode_kwargs,
            nstep_final_timestep=nstep_final_timestep,
            nstep_x1_estimation=nstep_x1_estimation,
            nstep_normalize_x1_predictions=nstep_normalize_x1_predictions,
            nstep_clamp_predictions=nstep_clamp_predictions,
            nstep_temperature_schedule=nstep_temperature_schedule,
            nstep_guidance_parameter=nstep_guidance_parameter,
        )

        self.reinit_flow_params = reinit_flow_params

        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False)

        self.to(dtype=model.dtype)
        checkpoint = model.state_dict()

        self.base_model_param_keys = list(checkpoint.keys())

        self.load_partial_state_dict(checkpoint)
        self.tie_weights()
        del model

        self.layers_to_freeze = []

        if self.base_params_to_freeze == 'all':
            self.layers_to_freeze += self.base_model_param_keys

        self.init_flow_weights_and_freeze_model()

        self.generation_mode = False
        if default_generation_nstep is None:
            self.default_generation_nstep = 0
        else:
            self.default_generation_nstep = default_generation_nstep

        if noise_schedule == "cosmap":
            noise_schedule = cosmap
        elif noise_schedule == "identity":
            noise_schedule = identity
        elif not callable(noise_schedule):
            raise ValueError(f"unknown noise schedule {noise_schedule}")

        self.noise_schedule = noise_schedule
        self.noise_rescaling = noise_rescaling

        self.minimum_training_noise = minimum_training_noise
        self.minimum_training_noise_units = minimum_training_noise_units.lower()
        assert self.minimum_training_noise_units in ['time']

        if self.minimum_training_noise is None:
            self.maximum_timestep = 1.0
            self.minimum_training_noise = 0
        elif self.minimum_training_noise == 0:
            self.maximum_timestep = 1.0
        elif self.minimum_training_noise_units == 'time':
            assert (self.minimum_training_noise < 1.0 and
                    self.minimum_training_noise > 0.0)
            self.maximum_timestep = 1 - self.minimum_training_noise
        else:
            raise NotImplementedError

        if velocity_schedule == 'constant':
            velocity_schedule = constant_velocity_schedule
        elif velocity_schedule == 'edm':
            velocity_schedule = edm_velocity_schedule
        elif not callable(velocity_schedule):
            raise ValueError(f"unknown velocity schedule {velocity_schedule}")
        self.velocity_schedule = velocity_schedule

        self.use_guidance = self.model.use_guidance
        self.guidance_modulation_num_classes = (
            self.model.guidance_modulation_num_classes)
        self.guidance_modulation_training_dropout = (
            self.model.guidance_modulation_training_dropout)

    @torch.no_grad()
    def compute_velocity(
        self,
        input_ids: torch.LongTensor = None,
        timesteps: torch.Tensor = None,
        class_labels: torch.LongTensor = None,
        flow_representation_embeds: tp.Optional[torch.FloatTensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.LongTensor] = None,
        past_key_values: tp.Optional[tp.Union[
            Cache, tp.List[torch.FloatTensor]]] = None,
        inputs_embeds: tp.Optional[torch.FloatTensor] = None,
        cached_final_hidden_states: tp.Optional[torch.FloatTensor] = None,

        sample_from_logits: bool = False,
        normalize_x1_predictions: tp.Optional[bool] = None,
        clamp_predictions: bool = False,
        temperature: tp.Optional[float] = None,
        velocity_dtype: torch.dtype = torch.float64,



        debug_labels: tp.Optional[torch.LongTensor] = None,
        **model_kwargs,
    ):
        debug_with_labels = debug_labels is not None

        model_output = self.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds.to(self.dtype),
            timesteps=timesteps,
            class_labels=class_labels,
            flow_representation_embeds=flow_representation_embeds.to(
                dtype=self.dtype),
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
            past_key_values=past_key_values,
            cached_final_hidden_states=cached_final_hidden_states,
            disable_recursive_nstep_call=True,
            **model_kwargs,
        )
        logits = model_output.logits

        if debug_with_labels:
            new_debug_labels = debug_labels
            min_dtype = torch.finfo(logits.dtype).min
            logits.fill_(min_dtype)
            debug_labels_one_hot = F.one_hot(
                new_debug_labels, num_classes=logits.shape[-1])
            logits = torch.where(debug_labels_one_hot > 0, 1.0, logits)

        x1_predictions = (
            self.model.get_weighted_flow_representation_for_logits(
                logits=logits,
                sample_from_logits=sample_from_logits,
                normalize_weighted_outputs=normalize_x1_predictions,
                clamp_predictions=clamp_predictions,

                temperature=temperature,
            )
        ).to(dtype=velocity_dtype)

        x1_xt_difference = x1_predictions - flow_representation_embeds.to(
            dtype=velocity_dtype)

        flow = self.velocity_schedule(
            x1_xt_difference=x1_xt_difference, t=timesteps)
        return flow

    @torch.no_grad()
    def nstep_inference(
        self,
        input_ids,
        class_labels: tp.Optional[torch.LongTensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.LongTensor] = None,
        initial_flow_representation_embeds: Optional[torch.Tensor] = None,
        start_timestep: float = 0.0,
        ode_kwargs: tp.Optional[dict] = None,
        final_timestep: Optional[float] = None,
        ode_steps: int = 16,

        only_last_token_prediction: bool = False,
        precompute_non_flow_path: bool = False,
        past_key_values: tp.Optional[tp.Union[
            Cache, tp.List[torch.FloatTensor]]] = None,
        cached_final_hidden_states: tp.Optional[torch.FloatTensor] = None,
        x1_estimation: tp.Optional[
            tp.Literal["average", "sample"]] = None,
        normalize_x1_predictions: tp.Optional[bool] = None,
        clamp_predictions: tp.Optional[bool] = None,
        temperature_schedule: tp.Optional[float | Callable] = None,
        guidance_parameter: tp.Optional[float] = None,


        debug_flow_magnitudes: bool = False,



        debug_labels: tp.Optional[torch.LongTensor] = None,

        record_evaluated_steps=False,
        **model_kwargs,
    ):

        assert ode_steps >= 2

        if ode_kwargs is None:
            ode_kwargs = self.model.ode_kwargs
        if final_timestep is None:
            final_timestep = self.model.nstep_final_timestep
        if x1_estimation is None:
            x1_estimation = self.model.nstep_x1_estimation
        sample_from_logits = False
        if x1_estimation == "average":
            sample_from_logits = False
        elif x1_estimation == "sample":
            sample_from_logits = True
        else:
            raise NotImplementedError
        if normalize_x1_predictions is None:
            normalize_x1_predictions = self.model.nstep_normalize_x1_predictions
        if clamp_predictions is None:
            if hasattr(self.model, 'nstep_clamp_predictions'):
                clamp_predictions = self.model.nstep_clamp_predictions
            else:
                clamp_predictions = False
        if temperature_schedule is None:
            temperature_schedule = self.model.nstep_temperature_schedule
            if temperature_schedule is None:
                temperature_schedule = 0.0
            get_temperature = self.model.get_nstep_temperature
            fixed_temperature = self.model.fixed_temperature
        else:
            assert isinstance(temperature_schedule, float) or isinstance(
                temperature_schedule, int)

            def get_temperature(t):
                return temperature_schedule
            fixed_temperature = True

        use_guidance = self.use_guidance and class_labels is not None
        if use_guidance:
            if guidance_parameter is None:
                guidance_parameter = self.model.nstep_guidance_parameter
            class_labels = class_labels.view(-1, 1)

        was_training = self.training
        self.eval()

        bsz, seq_len = input_ids.shape

        inputs_embeds: torch.Tensor = self.model.embed_tokens(input_ids)
        device = inputs_embeds.device

        if initial_flow_representation_embeds is None:
            flow_representation_embeds = torch.randn(
                bsz, seq_len, self.model.flow_representation_dim,
                device=device,
            )*self.noise_rescaling
        else:

            flow_representation_embeds = (
                initial_flow_representation_embeds.float())
            flow_representation_embeds = flow_representation_embeds.expand(
                bsz, seq_len, self.model.flow_representation_dim)

        precompute_cache = precompute_non_flow_path and (
            past_key_values is None or cached_final_hidden_states is None)
        if precompute_cache:
            precomputed_output = self.forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
                past_key_values=past_key_values,
                output_cached_final_hidden_states=True,
                disable_recursive_nstep_call=True,
                **model_kwargs
            )
            past_key_values = precomputed_output.past_key_values
            cached_final_hidden_states = (
                precomputed_output.cached_final_hidden_states)

        if only_last_token_prediction:

            flow_representation_embeds = flow_representation_embeds[:, -1:]
            if cached_final_hidden_states is not None:
                cached_final_hidden_states = cached_final_hidden_states[:, -1:]

        timesteps = torch.linspace(
            start_timestep, final_timestep,
            ode_steps,
            device=device
        )

        potential_boundary_step = False

        if ode_kwargs['method'] == 'euler':
            timesteps_delta = timesteps[-1] - timesteps[-2]

        elif ode_kwargs['method'] == 'midpoint':
            timesteps_delta = (timesteps[-1] - timesteps[-2])/2
        elif ode_kwargs['method'] in ['adaptive_heun', 'fehlberg2', 'bosh3']:
            timesteps_delta = 0
            potential_boundary_step = True
        elif 'heun' in ode_kwargs['method'] or 'rk' in ode_kwargs['method']:
            if ode_kwargs['method'] == 'heun':
                ode_kwargs['method'] = 'heun2'
            timesteps_delta = timesteps[-1] - timesteps[-2]
            potential_boundary_step = True
        else:
            potential_boundary_step = True
            timesteps_delta = 0.0

        if record_evaluated_steps:
            self._internal_tracked_steps = []
        else:
            self._internal_tracked_steps = None

        def ode_fn(t, x):

            if record_evaluated_steps:
                self._internal_tracked_steps.append(float(t))

            if potential_boundary_step:
                if t >= 1:
                    return torch.zeros_like(x)

            temperature = get_temperature(t=t+timesteps_delta)
            flow = self.compute_velocity(
                input_ids=input_ids,
                timesteps=t,
                inputs_embeds=inputs_embeds,
                class_labels=class_labels,
                flow_representation_embeds=x,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cached_final_hidden_states=cached_final_hidden_states,
                sample_from_logits=sample_from_logits,
                normalize_x1_predictions=normalize_x1_predictions,
                clamp_predictions=clamp_predictions,
                temperature=temperature,
                velocity_dtype=torch.float64,
                debug_labels=debug_labels,
                **model_kwargs,
            )

            if use_guidance:
                unconditional_flow = self.compute_velocity(
                    input_ids=input_ids,
                    timesteps=t,
                    inputs_embeds=inputs_embeds,
                    class_labels=None,
                    flow_representation_embeds=x,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cached_final_hidden_states=cached_final_hidden_states,
                    sample_from_logits=sample_from_logits,
                    normalize_x1_predictions=normalize_x1_predictions,
                    clamp_predictions=clamp_predictions,
                    temperature=temperature,
                    velocity_dtype=torch.float64,
                    debug_labels=debug_labels,
                    **model_kwargs,
                )
                flow = guidance_parameter*flow + (
                    1 - guidance_parameter)*unconditional_flow

            return flow

        timesteps = torch.linspace(
            start_timestep, final_timestep,
            ode_steps,
            device=device
        )

        trajectory = odeint(
            ode_fn, flow_representation_embeds.to(
                dtype=torch.float64), timesteps, **ode_kwargs)

        self.train(was_training)

        return FlowInferenceOutput(
            flow_trajectory=trajectory,
            timestep_trajectory=timesteps,
            cached_final_hidden_states=cached_final_hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            tracked_steps=self._internal_tracked_steps,
        )

    def get_flow_representation_for_tokens(
            self,
            input_ids,
            inputs_embeds: tp.Optional[torch.Tensor] = None,
    ):
        return self.model.get_flow_representation_for_tokens(
            input_ids=input_ids, inputs_embeds=inputs_embeds)

    def load_partial_state_dict(self, state_dict, keys_subset=None):
        current_state = self.state_dict()

        not_optimized_flow_params = []
        for new_param_name in current_state.keys():

            if (self.model.separate_flow_params and "flow_" in new_param_name
                    and not self.reinit_flow_params):
                old_param_name = new_param_name.replace("flow_", "")
            else:
                old_param_name = new_param_name
            if (
                (keys_subset is None or old_param_name in keys_subset)
                and old_param_name in state_dict
            ):
                old_param = state_dict[old_param_name]
                current_state[new_param_name].copy_(old_param.data)
            elif 'flow_' in new_param_name:
                not_optimized_flow_params.append(new_param_name)

    def init_flow_weights_and_freeze_model(self,):
        self.model.init_flow_weights()
        for name, param in self.named_parameters():
            if name in self.layers_to_freeze:
                param.requires_grad = False

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[
            int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        self.generation_mode = True

        generation_outputs = LlamaForCausalLM.generate(
            self=self,
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

        self.generation_mode = False

        return generation_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):

        model_inputs = LlamaForCausalLM.prepare_inputs_for_generation(
            self=self,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        return model_inputs

    def sample_timestep_and_flow_embeds(
            self,
            labels: torch.LongTensor,
            timesteps: torch.Tensor = None,
            ignore_index: tp.Optional[int] = -100,
    ):

        if ignore_index is not None:
            target_labels = torch.where(
                labels == ignore_index,
                input=torch.zeros_like(labels),
                other=labels,
            )
        else:
            target_labels = labels

        batch_dim, seq_dim = target_labels.shape

        if timesteps is None:

            timesteps = torch.rand(
                size=[batch_dim, seq_dim],
                device=target_labels.device,
            )
            timesteps = self.noise_schedule(timesteps)*self.maximum_timestep
        else:

            t_shape = timesteps.shape
            t_dims = len(t_shape)
            if t_dims < 2:
                timesteps = timesteps.view(-1, 1)
            timesteps = timesteps.expand(batch_dim, seq_dim)

        targets_flow_representation_embeds = (
            self.model.get_flow_representation_for_tokens(
                input_ids=target_labels,)
        )

        noise = torch.randn_like(
            targets_flow_representation_embeds)*self.noise_rescaling

        unsqueezed_timesteps = timesteps.unsqueeze(-1)
        flow_representation_embeds = (
            (1.0 - unsqueezed_timesteps)*noise +
            targets_flow_representation_embeds*unsqueezed_timesteps
        )
        return timesteps, flow_representation_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timesteps: torch.Tensor = None,
        class_labels: torch.LongTensor = None,

        flow_representation_embeds: tp.Optional[torch.FloatTensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.LongTensor] = None,
        past_key_values: tp.Optional[
            Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: tp.Optional[torch.FloatTensor] = None,
        labels: tp.Optional[torch.LongTensor] = None,


        shifted_labels: bool = False,
        compute_loss_for_labels: bool = True,
        use_cache: tp.Optional[bool] = None,
        output_attentions: tp.Optional[bool] = None,
        output_hidden_states: tp.Optional[bool] = None,
        return_dict: tp.Optional[bool] = None,
        cache_position: tp.Optional[torch.LongTensor] = None,

        ignore_index: int = -100,
        output_cached_final_hidden_states: bool = False,


        cached_final_hidden_states: tp.Optional[torch.FloatTensor] = None,


        disable_recursive_nstep_call: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if (self.default_generation_nstep > 0 and self.generation_mode
                and not disable_recursive_nstep_call):
            assert not self.training
            assert flow_representation_embeds is None
            nstep_output = self.nstep_inference(
                input_ids=input_ids,
                class_labels=class_labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                initial_flow_representation_embeds=flow_representation_embeds,
                ode_steps=self.default_generation_nstep,

                precompute_non_flow_path=True,
                past_key_values=past_key_values,
                cached_final_hidden_states=cached_final_hidden_states,
                only_last_token_prediction=True,
            )
            flow_trajectory = nstep_output.flow_trajectory
            timestep_trajectory = nstep_output.timestep_trajectory
            cached_final_hidden_states = nstep_output.cached_final_hidden_states
            past_key_values = nstep_output.past_key_values
            position_ids = nstep_output.position_ids

            flow_representation_embeds = flow_trajectory[-1].to(self.dtype)
            timesteps = timestep_trajectory[-1]

        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        if self.use_guidance and class_labels is not None:

            if input_ids is not None:
                seq_len = input_ids.shape[1]
            else:
                seq_len = inputs_embeds.shape[1]
            class_labels = class_labels.view(-1, 1).expand(
                class_labels.shape[0], seq_len)

            if self.training:

                drop_ids = (
                    torch.rand(
                        *class_labels.shape, device=class_labels.device)
                    < self.guidance_modulation_training_dropout
                )

                class_labels = torch.where(
                    drop_ids,
                    self.guidance_modulation_num_classes,
                    class_labels)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        batch_dim, seq_dim, _ = inputs_embeds.shape

        if timesteps is not None:
            t_shape = timesteps.shape
            t_dims = len(t_shape)
            if t_dims < 2:
                timesteps = timesteps.view(-1, 1)
            timesteps = timesteps.expand(batch_dim, seq_dim)

        if labels is not None:
            if timesteps is None:

                timesteps = torch.rand(
                    size=[batch_dim, seq_dim],
                    device=inputs_embeds.device,
                )
                timesteps = self.noise_schedule(
                    timesteps)*self.maximum_timestep

            if shifted_labels:
                flow_labels = labels
                timesteps_to_interpolate = timesteps.view(
                    batch_dim, seq_dim, 1)
            else:

                shift_labels = labels[..., 1:].contiguous()
                flow_labels = shift_labels

                ixs_for_inputs = torch.arange(
                    seq_dim - 1, device=timesteps.device)
                ixs_for_inputs = ixs_for_inputs.view(1, seq_dim - 1).expand(
                    batch_dim, seq_dim - 1)

                timesteps_to_interpolate = torch.gather(
                    input=timesteps, dim=1, index=ixs_for_inputs).view(
                        batch_dim, seq_dim - 1, 1)
            timesteps_to_interpolate = timesteps_to_interpolate.to(
                dtype=self.dtype)

            target_labels = torch.where(
                flow_labels == ignore_index,
                input=torch.zeros_like(flow_labels),
                other=flow_labels,
            )

            targets_flow_representation_embeds = (
                self.model.get_flow_representation_for_tokens(
                    input_ids=target_labels,)
            )

            noise = torch.randn_like(
                targets_flow_representation_embeds)*self.noise_rescaling

            flow_representation_embeds = (
                (1.0 - timesteps_to_interpolate)*noise +
                targets_flow_representation_embeds*timesteps_to_interpolate
            ).to(dtype=self.dtype)

            if not shifted_labels:

                flow_representation_embeds = F.pad(
                    input=flow_representation_embeds,



                    pad=(0, 0, 0, 1),
                    mode='constant',
                    value=0.,
                )
        elif flow_representation_embeds is not None:

            flow_representation_embeds = flow_representation_embeds.to(
                device=self.device)

        outputs = self.model(
            input_ids=input_ids,
            timesteps=timesteps,
            class_labels=class_labels,
            flow_representation_embeds=flow_representation_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            output_cached_final_hidden_states=output_cached_final_hidden_states,
            cached_final_hidden_states=cached_final_hidden_states,
        )

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in
                      range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None and compute_loss_for_labels:

            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            if shifted_labels:
                loss_logits = logits.contiguous()
                loss_labels = labels.contiguous()
            if not shifted_labels:

                loss_logits = logits[..., :-1, :].contiguous()
                loss_labels = shift_labels

            loss_logits = loss_logits.view(-1, self.config.vocab_size)
            loss_labels = loss_labels.view(-1)

            loss_labels = loss_labels.to(loss_logits.device)
            loss = loss_fct(loss_logits, loss_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FlowCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cached_final_hidden_states=outputs.cached_final_hidden_states,
            timesteps=timesteps,
        )
