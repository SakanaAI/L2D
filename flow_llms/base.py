import abc
import typing as tp
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math
from .base_components import SimpleMLP, BatchedLinear
from .utils import polynomial_temperature


class FrequencyEmbedder(nn.Module):
    def __init__(
            self, num_layers, conditioning_dim, frequency_embedding_size=256):
        super().__init__()
        assert num_layers > 0
        layers = []
        layer_input_dim = frequency_embedding_size
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(layer_input_dim, conditioning_dim, bias=True),
                nn.SiLU()]
            layer_input_dim = conditioning_dim
        layers += [nn.Linear(layer_input_dim, conditioning_dim, bias=True)]
        self.mlp = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.frequency_embedding_size = frequency_embedding_size
        assert self.frequency_embedding_size % 2 == 0

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        dtype = t.dtype
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class OneHotEmbedder(nn.Module):

    def __init__(self, num_classes, embedding_dim, embedding_dropout):
        super().__init__()
        use_cfg_embedding = embedding_dropout > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, embedding_dim)
        self.num_classes = num_classes
        self.dropout_prob = embedding_dropout
        self.use_dropout = self.dropout_prob > 0

    def token_drop(self, labels, force_drop_ids=None):

        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device)
                < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, x, drop_ids=False, force_drop_ids=None):
        if (drop_ids and self.use_dropout) or (force_drop_ids is not None):
            x = self.token_drop(x, force_drop_ids)
        embeddings = self.embedding_table(x)
        return embeddings


class MappingEmbedder(nn.Module):

    def __init__(self, input_dim, embedding_dim, depth=1, bias=True,):
        nn.Module.__init__(
            self=self,
        )
        if depth == 0:
            assert input_dim == embedding_dim
            self.embedder = nn.Identity()
        else:
            self.embedder = SimpleMLP(
                input_dim=input_dim,
                hidden_dim=embedding_dim,
                output_dim=embedding_dim,
                activation=nn.SiLU,
                num_layers=depth,
                bias=bias,
                final_non_linearity=False,
            )
        self.depth = depth
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    def forward(self, x, **kwargs):
        x = self.embedder(x)
        return x


class FlowToLMEmbedder(nn.Module):

    def __init__(
            self,
            flow_representation_dim,
            main_model_hidden_dim,
            hidden_size,
            depth,
            flow_representation_normalize,
            flow_representation_rescaling,
            noise_rescaling,
            flow_to_lm_timestep_rescaling,
            flow_to_lm_rescale_in_float32,
            bias=True,


            main_model_tokens_per_flow=1,
    ):
        nn.Module.__init__(
            self=self,
        )
        if main_model_tokens_per_flow > 1:
            raise NotImplementedError

        input_dim = flow_representation_dim
        output_dim = main_model_hidden_dim*main_model_tokens_per_flow
        if depth == 0:
            assert input_dim == output_dim
            self.embedder = nn.Identity()
        else:
            self.embedder = SimpleMLP(
                input_dim=input_dim,
                hidden_dim=hidden_size,
                output_dim=output_dim,
                activation=nn.SiLU,
                num_layers=depth,
                bias=bias,
                final_non_linearity=False,
            )
        self.flow_to_lm_timestep_rescaling = flow_to_lm_timestep_rescaling
        self.flow_to_lm_rescale_in_float32 = flow_to_lm_rescale_in_float32

        self.var_x0_dist = noise_rescaling**2
        assert self.flow_to_lm_timestep_rescaling >= 0
        if self.flow_to_lm_timestep_rescaling > 0:
            assert flow_representation_normalize

        self.var_x1_dist = 1/flow_representation_dim
        if flow_representation_rescaling is not None:
            if flow_representation_rescaling == 'div':
                self.var_x1_dist = 1/flow_representation_dim**2
            elif flow_representation_rescaling == 'mult':
                self.var_x1_dist = 1
            elif flow_representation_rescaling == 'none':
                self.var_x1_dist = 1/flow_representation_dim
            else:
                raise NotImplementedError

    def forward(self, x, timesteps, **kwargs):

        if self.flow_to_lm_rescale_in_float32:
            x_input_dtype = x.dtype
            x = x.to(dtype=torch.float32)
        if self.flow_to_lm_timestep_rescaling > 0:
            bsz, seq_dim, _ = x.shape
            timesteps = timesteps.view(bsz, seq_dim, 1)
            variance_at_timesteps = (
                torch.square(1 - timesteps)*self.var_x0_dist +
                torch.square(timesteps)*self.var_x1_dist)
            std_at_timesteps = torch.sqrt(variance_at_timesteps)
            x = x/std_at_timesteps.to(dtype=x.dtype)
            x = x*self.flow_to_lm_timestep_rescaling
        if self.flow_to_lm_rescale_in_float32:
            x = x.to(dtype=x_input_dtype)
        x = self.embedder(x)
        return x


class ConditioningHead(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_hidden_components,
            num_layers,
            conditioning_dim,
    ):
        super().__init__()
        assert num_layers > 0
        layers = []
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(conditioning_dim, conditioning_dim, bias=True),
                nn.SiLU()]
        output_dim = hidden_dim*num_hidden_components
        layers += [nn.Linear(conditioning_dim, output_dim, bias=True)]
        self.mlp = nn.Sequential(*layers)
        self.num_layers = num_layers

    def forward(self, c):
        t_emb = self.mlp(c)
        return t_emb


class FlowLanguageModel(abc.ABC):

    embed_tokens: nn.Module

    def __init__(
            self,
            config,
            main_model_hidden_dim: int,




            flow_representation_space: str = 'mapping',

            flow_representation_dim: tp.Optional[int] = None,
            flow_representation_num_layers: tp.Optional[int] = None,

            flow_representation_normalize: bool = True,

            flow_representation_rescaling: tp.Optional[
                tp.Literal["div", "mult", "none"]] = None,



            noise_rescaling: float = 1.0,










            self_attention_on_flow_sequence: bool = False,



            flow_to_lm_translation_depth: int = 2,
            flow_to_lm_hidden_size: tp.Optional[int] = None,


            flow_to_lm_timestep_rescaling: tp.Optional[float] = None,


            flow_to_lm_rescale_in_float32: bool = False,


            preserve_behavior_at_flow_start: bool = False,

            modulate_hidden_states: bool = False,
            full_dit_modulation: bool = False,
            timestep_modulation_num_layers: int = 2,
            timestep_modulation_freq_embedding_size: int = 256,

            timestep_modulation_hidden_size: tp.Optional[int] = None,



            guidance_modulation_num_classes: tp.Optional[int] = None,


            guidance_modulation_training_dropout: float = 0.2,


            freeze_modulation_at_flow_start: bool = False,


            vocab_size: tp.Optional[int] = None,


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

        self.config = config
        self.main_model_hidden_dim = main_model_hidden_dim
        self.flow_to_lm_translation_depth = flow_to_lm_translation_depth
        if flow_to_lm_hidden_size is None:
            flow_to_lm_hidden_size = main_model_hidden_dim
        self.flow_to_lm_hidden_size = flow_to_lm_hidden_size
        self.flow_to_lm_timestep_rescaling = flow_to_lm_timestep_rescaling
        self.flow_to_lm_rescale_timestep = False
        if self.flow_to_lm_timestep_rescaling is not None:
            if self.flow_to_lm_timestep_rescaling > 0:
                self.flow_to_lm_rescale_timestep = True
            elif self.flow_to_lm_timestep_rescaling == -1:

                raise NotImplementedError
            elif self.flow_to_lm_timestep_rescaling < 0:
                raise NotImplementedError
        self.flow_to_lm_rescale_in_float32 = flow_to_lm_rescale_in_float32

        self.flow_representation_space = flow_representation_space.lower()
        assert self.flow_representation_space in [
            'embedding', 'mapping', 'one-hot']
        self.flow_representation_from_embeds = not (
            self.flow_representation_space == 'one-hot')
        self.flow_representation_dim = flow_representation_dim
        if self.flow_representation_dim is None:
            assert main_model_hidden_dim is not None
            self.flow_representation_dim = main_model_hidden_dim
        self.flow_representation_num_layers = flow_representation_num_layers
        self.flow_representation_normalize = flow_representation_normalize
        self.flow_representation_rescaling = flow_representation_rescaling

        if self.flow_representation_rescaling is not None:
            flow_representation_rescaling = (
                flow_representation_rescaling.lower())
            self.flow_representation_rescaling = flow_representation_rescaling
            if flow_representation_rescaling == 'div':
                self.flow_representation_rescale = 1/math.sqrt(
                    self.flow_representation_dim)
            elif flow_representation_rescaling == 'mult':
                self.flow_representation_rescale = math.sqrt(
                    self.flow_representation_dim)
            elif flow_representation_rescaling == 'none':
                self.flow_representation_rescale = 1
            else:
                raise NotImplementedError
        else:
            self.flow_representation_rescale = 1

        self.noise_rescaling = noise_rescaling
        assert self.noise_rescaling > 0.0

        self.self_attention_on_flow_sequence = self_attention_on_flow_sequence
        if self.self_attention_on_flow_sequence:

            raise NotImplementedError

        self.make_flow_representation_embedder(vocab_size=vocab_size)

        self.preserve_behavior_at_flow_start = preserve_behavior_at_flow_start

        if timestep_modulation_hidden_size is None:
            timestep_modulation_hidden_size = main_model_hidden_dim
        self.modulate_hidden_states = modulate_hidden_states
        self.full_dit_modulation = full_dit_modulation
        self.timestep_modulation_num_layers = timestep_modulation_num_layers
        self.timestep_modulation_freq_embedding_size = (
            timestep_modulation_freq_embedding_size)
        self.timestep_modulation_hidden_size = timestep_modulation_hidden_size
        self.freeze_modulation_at_flow_start = freeze_modulation_at_flow_start
        self.separate_flow_params = separate_flow_params
        self.separate_flow_params_with_lora = separate_flow_params_with_lora

        if self.separate_flow_params_with_lora:
            assert self.separate_flow_params

        self.flow_lora_rank = flow_lora_rank
        self.flow_lora_alpha = flow_lora_alpha

        self.timestep_embedder = FrequencyEmbedder(
            num_layers=timestep_modulation_num_layers,
            conditioning_dim=timestep_modulation_hidden_size,
            frequency_embedding_size=timestep_modulation_freq_embedding_size,
        )

        self.use_guidance = False
        if guidance_modulation_num_classes is not None:
            if guidance_modulation_num_classes > 0:
                self.use_guidance = True

        self.guidance_modulation_num_classes = guidance_modulation_num_classes
        self.guidance_modulation_training_dropout = (
            guidance_modulation_training_dropout)
        if self.use_guidance:
            assert self.guidance_modulation_training_dropout > 0
            self.data_label_embedder = nn.Embedding(

                num_embeddings=self.guidance_modulation_num_classes + 1,

                embedding_dim=self.timestep_modulation_hidden_size,
            )

        if self.flow_to_lm_rescale_timestep:

            self.flow_to_lm_encoder = FlowToLMEmbedder(
                flow_representation_dim=self.flow_representation_dim,
                main_model_hidden_dim=main_model_hidden_dim,
                hidden_size=self.flow_to_lm_hidden_size,
                depth=self.flow_to_lm_translation_depth,
                flow_representation_normalize=(
                    self.flow_representation_normalize),
                flow_representation_rescaling=(
                    self.flow_representation_rescaling),
                noise_rescaling=self.noise_rescaling,
                flow_to_lm_timestep_rescaling=(
                    self.flow_to_lm_timestep_rescaling),
                flow_to_lm_rescale_in_float32=(
                    self.flow_to_lm_rescale_in_float32),
                bias=True,

                main_model_tokens_per_flow=1,
            )
        else:

            self.flow_to_lm_encoder = SimpleMLP(
                input_dim=self.flow_representation_dim,
                hidden_dim=self.flow_to_lm_hidden_size,
                output_dim=main_model_hidden_dim,
                activation=nn.SiLU,
                num_layers=self.flow_to_lm_translation_depth,
            )

        if self.flow_to_lm_translation_depth == 0:
            assert self.flow_representation_dim == main_model_hidden_dim
        if self.preserve_behavior_at_flow_start:
            if self.modulate_hidden_states:
                assert self.freeze_modulation_at_flow_start
        elif self.freeze_modulation_at_flow_start:
            assert self.preserve_behavior_at_flow_start

        self.nstep_final_timestep = nstep_final_timestep
        assert nstep_final_timestep >= 0.0 and nstep_final_timestep <= 1.0
        self.nstep_x1_estimation = nstep_x1_estimation.lower()
        assert self.nstep_x1_estimation in ['average', 'sample']
        self.nstep_normalize_x1_predictions = nstep_normalize_x1_predictions
        self.nstep_clamp_predictions = nstep_clamp_predictions
        self.nstep_temperature_schedule = nstep_temperature_schedule

        if self.nstep_temperature_schedule is None:
            self.nstep_temperature_schedule = 0.0
        if isinstance(self.nstep_temperature_schedule, float) or isinstance(
                self.nstep_temperature_schedule, int):
            self.fixed_temperature = True
            self.nstep_temperature_schedule_fn = None
        else:
            self.fixed_temperature = False
            if isinstance(self.nstep_temperature_schedule, str):
                self.nstep_temperature_schedule = (
                    self.nstep_temperature_schedule.lower())
                if self.nstep_temperature_schedule.startswith('polynomial'):

                    _, p, end_t = self.nstep_temperature_schedule.split('_')
                    p = float(p)
                    end_t = float(end_t)

                    def temperature_schedule_fn(t):
                        return polynomial_temperature(t=t, p=p, end_t=end_t)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            self.nstep_temperature_schedule_fn = temperature_schedule_fn

        self.nstep_guidance_parameter = nstep_guidance_parameter

        if self.use_guidance:
            if self.nstep_guidance_parameter is None:
                self.nstep_guidance_parameter = 1.0
            else:

                assert self.nstep_guidance_parameter >= 0.0

        if ode_kwargs is None:
            ode_kwargs = {}
        self.ode_kwargs = ode_kwargs
        self.adaptive_step_size = False

    def get_nstep_temperature(self, t, override_temperature=False):
        if self.fixed_temperature:
            return self.nstep_temperature_schedule
        else:
            return self.nstep_temperature_schedule_fn(t=t)

    def init_flow_weights(self,):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, BatchedLinear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.timestep_embedder.to(dtype=torch.float32)
        self.timestep_embedder.apply(_basic_init)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[-1].weight, std=0.02)

        if self.use_guidance:

            nn.init.normal_(self.data_label_embedder.weight, std=0.02)

        self.flow_token_embedder.apply(_basic_init)
        if self.flow_representation_space == 'one-hot':
            nn.init.normal_(
                self.flow_token_embedder.embedding_table.weight, std=0.02)

        self.flow_to_lm_encoder.apply(_basic_init)

    def make_flow_representation_embedder(self, vocab_size):
        if self.flow_representation_space == 'embedding':
            if self.flow_representation_num_layers is not None:
                assert self.flow_representation_num_layers == 0
            assert self.flow_representation_dim == self.main_model_hidden_dim
            self.flow_token_embedder = MappingEmbedder(
                input_dim=self.main_model_hidden_dim,
                embedding_dim=self.flow_representation_dim,
                depth=0,
            )
        elif self.flow_representation_space == 'mapping':
            assert self.flow_representation_num_layers is not None
            assert self.flow_representation_num_layers >= 1
            self.flow_token_embedder = MappingEmbedder(
                input_dim=self.main_model_hidden_dim,
                embedding_dim=self.flow_representation_dim,
                depth=self.flow_representation_num_layers,
            )
        else:
            assert vocab_size is not None
            self.flow_token_embedder = OneHotEmbedder(
                num_classes=vocab_size,
                embedding_dim=self.flow_representation_dim,

                embedding_dropout=0,
            )

    def get_flow_representation_for_tokens(
            self,
            input_ids: torch.Tensor,
            inputs_embeds: tp.Optional[torch.Tensor] = None,
    ):
        if self.flow_representation_from_embeds:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            flow_representation = self.flow_token_embedder(inputs_embeds)
        else:
            flow_representation = self.flow_token_embedder(input_ids)
        if self.flow_representation_normalize:
            flow_representation = F.normalize(
                flow_representation, dim=-1)*self.flow_representation_rescale
        return flow_representation

    def get_weighted_flow_representation_for_logits(

            self,
            logits,
            sample_from_logits: bool = False,
            normalize_weighted_outputs: bool = False,
            clamp_predictions: bool = False,
            temperature: float = 1.0,
    ):

        if self.flow_representation_from_embeds:
            base_embedding_table = self.embed_tokens.weight
            flow_embedding_table = self.flow_token_embedder(
                base_embedding_table)
        else:
            flow_embedding_table = (
                self.flow_token_embedder.embedding_table.weight)
        if self.flow_representation_normalize:
            flow_embedding_table = F.normalize(
                flow_embedding_table,
                dim=-1)*self.flow_representation_rescale

        all_representations = flow_embedding_table.float()
        if temperature <= 0.0:
            max_temperature_idx = torch.argmax(logits, dim=-1)
            weighted_representation = all_representations[max_temperature_idx]
        else:
            probabilities = torch.softmax(logits.float()/temperature, dim=-1)
            if sample_from_logits:
                batch_dims = probabilities.shape[:-1]
                flat_probs = probabilities.view(-1, probabilities.shape[-1])

                samples = torch.multinomial(
                    flat_probs, num_samples=1, replacement=True).view(-1)

                weighted_representation = all_representations[samples].view(
                    *batch_dims, all_representations.shape[-1])
            else:
                if clamp_predictions:

                    distances = torch.einsum(
                        "mv,vn,wn->mw",
                        probabilities,
                        all_representations,
                        all_representations,
                    )

                    closest_indices = torch.argmax(distances, dim=-1)
                    weighted_representation = all_representations[
                        closest_indices]
                else:

                    weighted_representation = probabilities@all_representations

            if normalize_weighted_outputs and not clamp_predictions:
                weighted_representation = F.normalize(
                    weighted_representation,
                    dim=-1,)*self.flow_representation_rescale
        return weighted_representation
