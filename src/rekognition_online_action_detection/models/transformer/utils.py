# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
assert torch.__version__ >= '1.6.0'
import torch.nn as nn


class DynamicTanh(nn.Module):
    """Element-wise tanh(α·x) + affine; optional LayerNorm replacement (DyT)."""

    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


def build_norm_layer(d_model, norm_type='layernorm', dyt_alpha_init=0.5):
    norm_type = (norm_type or 'layernorm').lower()
    if norm_type == 'layernorm':
        return nn.LayerNorm(d_model)
    if norm_type == 'dyt':
        return DynamicTanh(d_model, channels_last=True, alpha_init_value=dyt_alpha_init)
    raise ValueError("NORM_TYPE must be 'layernorm' or 'dyt', got {!r}".format(norm_type))


def layer_norm(d_model, condition=True, norm_type='layernorm', dyt_alpha_init=0.5):
    if not condition:
        return None
    return build_norm_layer(d_model, norm_type=norm_type, dyt_alpha_init=dyt_alpha_init)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
