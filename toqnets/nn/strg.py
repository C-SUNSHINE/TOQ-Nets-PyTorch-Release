#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : strg.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch, and implemented the model in
# Wang, Xiaolong, and Abhinav Gupta. "Videos as space-time region graphs."
# Proceedings of the European conference on computer vision (ECCV). 2018.
# Distributed under terms of the MIT license.

import torch
from torch import nn


class SpaceTimeRegionalConv(nn.Module):
    """
    Space Time Region Graph
    """

    def __init__(self, input_dim, output_dim, t_kernel_size=1, t_stride=1, t_padding=None,
                 t_dilation=1, bias=True, residual=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if t_padding is None:
            t_padding = (t_kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
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

    def _get_graph(self, length, n_agents, device):
        g = torch.zeros(length, n_agents, length, n_agents, device=device)
        g = torch.max(g, torch.eye(length, length, device=device).view(length, 1, length, 1))
        g = torch.max(g, torch.eye(n_agents, n_agents, device=device).view(1, n_agents, 1, n_agents))
        return g.view(length * n_agents, length * n_agents)

    def forward(self, x):
        """
        :param x: [batch, length, n_agents, input_dim]
        return output: [batch, new_length, n_agents, output_dim]
        """
        batch, length, n_agents, _ = x.size()
        x = x.permute(0, 3, 1, 2)
        res = self.residual(x)
        x = self.conv(x)
        new_length = x.size(2)
        g = self._get_graph(new_length, n_agents, x.device)
        x = x.view(batch, self.output_dim, new_length * n_agents)
        x = torch.einsum('ncv,vw->ncw', (x, g)).view(batch, self.output_dim, new_length, n_agents) / (
                (length + n_agents - 2) ** .5)
        x = x + res
        return self.relu(x.permute(0, 2, 3, 1))


class STRG(nn.Module):

    def __init__(self, input_dim, output_dim, t_kernel_size=1, small_model=False, tiny_model=False):
        super().__init__()
        layer_kwargs = {
            't_kernel_size': t_kernel_size,
        }
        self.strg_layers = nn.Sequential(
            SpaceTimeRegionalConv(input_dim, 64, t_stride=1, residual=False, **layer_kwargs),
            SpaceTimeRegionalConv(64, 64, t_stride=1, **layer_kwargs),
            SpaceTimeRegionalConv(64, 64, t_stride=1, **layer_kwargs),
            SpaceTimeRegionalConv(64, 64, t_stride=1, **layer_kwargs),
            SpaceTimeRegionalConv(64, 128, t_stride=2, **layer_kwargs),
            SpaceTimeRegionalConv(128, 128, t_stride=1, **layer_kwargs),
            SpaceTimeRegionalConv(128, 128, t_stride=1, **layer_kwargs),
            SpaceTimeRegionalConv(128, 256, t_stride=2, **layer_kwargs),
            SpaceTimeRegionalConv(256, 256, t_stride=1, **layer_kwargs),
            SpaceTimeRegionalConv(256, output_dim, t_stride=1, **layer_kwargs),
        ) if not small_model and not tiny_model else (nn.Sequential(
            SpaceTimeRegionalConv(input_dim, 16, t_stride=1, residual=False, **layer_kwargs),
            SpaceTimeRegionalConv(16, 16, t_stride=1, **layer_kwargs),
            SpaceTimeRegionalConv(16, 32, t_stride=2, **layer_kwargs),
            SpaceTimeRegionalConv(32, 64, t_stride=2, **layer_kwargs),
            SpaceTimeRegionalConv(64, output_dim, t_stride=1, **layer_kwargs),
        ) if not tiny_model else nn.Sequential(
            SpaceTimeRegionalConv(input_dim, 32, t_stride=1, residual=False, **layer_kwargs),
            SpaceTimeRegionalConv(32, 64, t_stride=2, **layer_kwargs),
            SpaceTimeRegionalConv(64, output_dim, t_stride=2, **layer_kwargs),
        ))

    def forward(self, x):
        return self.strg_layers(x)
