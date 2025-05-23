import abc
from typing import Optional
import torch
from torch import nn
import math


def cosmap(t):

    return 1.0 - (1.0/(torch.tan(torch.pi / 2 * t) + 1))


def polynomial_temperature(t, p, end_t):
    scaling = 1/end_t
    x = 1 - scaling*t

    temperature = torch.pow(x, p)
    temperature = torch.where(t >= end_t, torch.zeros_like(t), temperature)
    return temperature


def identity(t):
    return t


def constant_velocity_schedule(x1_xt_difference, t):

    velocity = x1_xt_difference/(1-t)
    return velocity


def edm_velocity_schedule(x1_xt_difference, t):

    velocity = x1_xt_difference/(1-t)**2
    return velocity


def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))


def rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_to_vector(
        v, cos, sin, position_ids=None, unsqueeze_dim=1) -> torch.Tensor:

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    v_embed = (v * cos) + (rotate_half(v) * sin)
    return v_embed
