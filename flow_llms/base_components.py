import abc
from typing import Optional
import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,)

import typing as tp
from typing import Tuple
from dataclasses import dataclass


@dataclass
class FlowModelOutputWithPast(BaseModelOutputWithPast):
    cached_final_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class FlowCausalLMOutputWithPast(CausalLMOutputWithPast):
    cached_final_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    timesteps: Optional[torch.FloatTensor] = None


@dataclass
class FlowInferenceOutput(ModelOutput):
    flow_trajectory: tp.List[torch.Tensor] = None
    timestep_trajectory: tp.List[torch.Tensor] = None
    cached_final_hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    position_ids: Optional[torch.LongTensor] = None
    tracked_steps: Optional[list] = None


class BatchedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_parallel: int,
        batched_inputs: bool,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(BatchedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_parallel = num_parallel
        self.batched_inputs = batched_inputs
        if num_parallel is None or (num_parallel == 1):
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(
                    out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(
                torch.empty((num_parallel, in_features,
                            out_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty((num_parallel, 1, out_features),
                                **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward_expand(self, input):
        pre_in = input.expand(self.num_parallel, -1, -1)
        return torch.baddbmm(self.bias, pre_in, self.weight)

    def forward(self, input):
        if self.num_parallel is None or (self.num_parallel == 1):
            return F.linear(input, self.weight, self.bias)

        if not self.batched_inputs:
            *batch_dims, _ = input.shape
            input = torch.flatten(input, start_dim=0, end_dim=-2).unsqueeze(0)
            out = self.forward_expand(input)
        else:
            _, *batch_dims, _ = input.shape
            input = torch.flatten(input, start_dim=1, end_dim=-2)
            out = torch.baddbmm(self.bias, input, self.weight)
        out = out.unflatten(dim=1, sizes=batch_dims)
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, num_parallel={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.num_parallel,
            self.bias is not None,
        )


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation,
        num_layers=2,
        bias=True,
        final_non_linearity=False,
        num_parallel=None,
        batched_inputs=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.final_non_linearity = final_non_linearity
        self.num_parallel = num_parallel
        self.batched_inputs = batched_inputs

        def make_linear(in_dims, out_dims, batched_inputs=None):
            if num_parallel is None:
                return nn.Linear(in_dims, out_dims, bias=bias)
            else:
                return BatchedLinear(
                    in_dims,
                    out_dims,
                    num_parallel=num_parallel,
                    batched_inputs=batched_inputs,
                    bias=bias,
                )

        in_dims = self.input_dim
        out_dims = self.hidden_dim
        batched_in = batched_inputs
        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers += [make_linear(in_dims,
                                        out_dims, batched_in), activation()]
            in_dims = self.hidden_dim
            batched_in = True
        out_dims = self.output_dim
        self.layers += [make_linear(in_dims, out_dims, batched_in)]
        if self.final_non_linearity:
            self.layers += [activation()]
        self.layers_module = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers_module(x)


class LoRA(torch.nn.Module):
    def __init__(self, in_dim, out_dim, wrapped_layer, rank, alpha=None,):
        super().__init__()
        self.wrapped_layer = wrapped_layer
        self.rescale_stddev = 1/torch.tensor(rank).float()
        if alpha is None:
            alpha = rank*2
        elif alpha == -1:
            alpha = rank*2
        self.alpha = alpha
        self.total_rescale = float(self.alpha*self.rescale_stddev)
        assert self.alpha > 0

        self.A = torch.nn.Parameter(
            torch.randn(in_dim, rank)*self.rescale_stddev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        delta = (x@self.A@self.B)*self.total_rescale
        return self.wrapped_layer(x) + delta
