#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : non_local.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch, and implemented the model in
# Wang, Xiaolong, et al. "Non-local neural networks."
# Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
# Distributed under terms of the MIT license.

import torch
from torch import nn


class NonLocalLayer(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=None, t_kernel_size=1, t_stride=1, t_padding=None,
                 t_dilation=1, bias=True, residual=True):
        super().__init__()
        if t_padding is None:
            t_padding = (t_kernel_size - 1) // 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = hidden_dim if hidden_dim is not None else (input_dim + output_dim) // 2
        self.hidden_dim = hidden_dim
        self.theta = nn.Linear(hidden_dim, hidden_dim)
        self.phi = nn.Linear(hidden_dim, hidden_dim)
        self.g = nn.Linear(hidden_dim, hidden_dim)
        self.f = nn.Linear(hidden_dim, output_dim)
        self.conv = nn.Conv2d(
            input_dim,
            hidden_dim,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        if not residual:
            self.residual = lambda x: 0
        elif (input_dim == output_dim) and (t_stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=1,
                    stride=(t_stride, 1)),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: [batch, length, n_agents, input_dim]
        return output[batch, length, n_agents, output_dim]
        """
        batch, length, n_agents, _ = x.size()
        x = x.permute(0, 3, 1, 2)
        res = self.residual(x)
        if not isinstance(res, int):
            res = res.permute(0, 2, 3, 1)
        x = self.conv(x).permute(0, 2, 3, 1)

        length = x.size(1)

        x = x.reshape(batch, length * n_agents, self.hidden_dim)
        theta = self.theta(x.reshape(-1, self.hidden_dim)).reshape(batch, length * n_agents, -1)
        phi = self.phi(x.reshape(-1, self.hidden_dim)).reshape(batch, length * n_agents, -1).permute(0, 2, 1)
        g = self.g(x.reshape(-1, self.hidden_dim)).reshape(batch, length * n_agents, -1)
        y = (torch.bmm(theta, phi) / (theta.size(-1) ** .5)).softmax(dim=2)
        z = torch.bmm(y, g) / (y.size(-1) ** .5)
        z = self.f(z.view(batch, length * n_agents, -1)).reshape(batch, length, n_agents, self.output_dim)
        return z + res


class NonLocal(nn.Module):
    """
    Non Local Neural Networks
    """

    def __init__(self, input_dim, output_dim, t_kernel_size=1, small_model=False, tiny_model=False):
        super().__init__()
        layer_kwargs = {
            't_kernel_size': t_kernel_size,
        }
        self.non_local_layers = nn.Sequential(
            NonLocalLayer(input_dim, 64, t_stride=1, residual=False, **layer_kwargs),
            NonLocalLayer(64, 64, t_stride=1, **layer_kwargs),
            NonLocalLayer(64, 64, t_stride=1, **layer_kwargs),
            NonLocalLayer(64, 64, t_stride=1, **layer_kwargs),
            NonLocalLayer(64, 128, t_stride=2, **layer_kwargs),
            NonLocalLayer(128, 128, t_stride=1, **layer_kwargs),
            NonLocalLayer(128, 128, t_stride=1, **layer_kwargs),
            NonLocalLayer(128, 256, t_stride=2, **layer_kwargs),
            NonLocalLayer(256, 256, t_stride=1, **layer_kwargs),
            NonLocalLayer(256, output_dim, t_stride=1, **layer_kwargs),
        ) if not small_model and not tiny_model else (nn.Sequential(
            NonLocalLayer(input_dim, 32, t_stride=1, residual=False, **layer_kwargs),
            NonLocalLayer(32, 32, t_stride=2, **layer_kwargs),
            NonLocalLayer(32, 64, t_stride=2, **layer_kwargs),
            NonLocalLayer(64, output_dim, t_stride=1, **layer_kwargs),
        ) if not tiny_model else nn.Sequential(
            NonLocalLayer(input_dim, 32, t_stride=1, residual=False, **layer_kwargs),
            NonLocalLayer(32, 32, t_stride=2, **layer_kwargs),
            NonLocalLayer(32, output_dim, t_stride=2, **layer_kwargs),
        ))

    def forward(self, x):
        return self.non_local_layers(x)
