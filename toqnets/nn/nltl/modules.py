#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : modules.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/15/2020
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import jactorch
import torch
import torch.nn as nn

from toqnets.nn.nlm import MLPLogic
from .functional import TemporalPoolingImplementation, backward_pooling_1d1d, interval_pooling, temporal_pooling_1d, temporal_pooling_2d

__all__ = ['TemporalPooling1D', 'TemporalPooling2D', 'TemporalLogicLayer', 'TemporalLogicMachine',
           'TemporalLogicMachinePartial', 'TemporalLogicMachineDP']


class TemporalPoolingBase(nn.Module):
    def __init__(self, implementation='forloop', residual=True):
        super().__init__()
        self.implementation = TemporalPoolingImplementation.from_string(implementation)
        self.residual = residual

    def get_output_dim(self, input_dim):
        return input_dim * 2 if not self.residual else input_dim * 3

    def forward(self, input):
        output = self.forward_pool(input)
        if self.residual:
            return torch.cat((input, output), dim=-1)
        else:
            return output

    def forward_pool(self, input):
        raise NotImplementedError()


class TemporalPooling1D(TemporalPoolingBase):
    def forward_pool(self, input):
        return temporal_pooling_1d(input, self.implementation)


class TemporalPooling2D(TemporalPoolingBase):
    def forward_pool(self, input):
        return temporal_pooling_2d(input, self.implementation)


class TemporalLogicLayer(nn.Module):
    def __init__(
        self,
        input_dim, output_dim, logic_hidden_dim=None,
        pooling_dim=1, pooling_implementation='forloop', pooling_residual=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.logic_hidden_dim = logic_hidden_dim if logic_hidden_dim is not None else []

        self.pooling_dim = pooling_dim
        self.pooling_residual = pooling_residual
        self.pooling_implementation = pooling_implementation

        if self.pooling_dim == 1:
            self.temporal_pooling = TemporalPooling1D(self.pooling_implementation, residual=self.pooling_residual)
        elif self.pooling_dim == 2:
            self.temporal_pooling = TemporalPooling2D(self.pooling_implementation, residual=self.pooling_residual)

        current_dim = self.temporal_pooling.get_output_dim(input_dim)
        self.logic_layer = MLPLogic(current_dim, self.output_dim, self.logic_hidden_dim)

    def forward(self, input):
        return self.logic_layer(self.temporal_pooling(input))


class TemporalLogicMachinePartial(nn.Module):
    def __init__(self, input_dim, output_dim, nr_layers, hidden_dims=None, residual_input=True, pooling_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_layers = nr_layers
        self.hidden_dims = hidden_dims if hidden_dims is not None else [input_dim for _ in range(nr_layers - 1)]
        self.residual_input = residual_input
        self.pooling_dim = pooling_dim
        assert len(self.hidden_dims) == nr_layers - 1

        self.layers = nn.ModuleList()
        current_dim = self.input_dim
        for i in range(nr_layers - 1):
            self.layers.append(TemporalLogicLayer(current_dim, self.hidden_dims[i], pooling_dim=self.pooling_dim))
            current_dim = self.hidden_dims[i] + (self.input_dim if self.residual_input else 0)
        self.layers.append(TemporalLogicLayer(current_dim, self.output_dim, pooling_dim=self.pooling_dim))

    def forward(self, input):
        f = input
        for i, module in enumerate(self.layers.children()):
            if i != 0 and self.residual_input:
                f = torch.cat((f, input), dim=-1)
            f = module(f)
        return f


class TemporalLogicMachineDP(nn.Module):
    def __init__(self, input_dim, output_dim, nr_steps, hidden_dims=None, residual_input=False, until=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_steps = nr_steps
        self.hidden_dims = hidden_dims if hidden_dims is not None else [input_dim for _ in range(nr_steps - 1)]
        self.residual_input = residual_input
        self.until = until

        self.step_linears = nn.ModuleList()
        relation_dim = self.input_dim * (2 if until else 1)
        current_dim = relation_dim + (self.input_dim if self.residual_input else 0)
        for i in range(self.nr_steps - 1):
            self.step_linears.append(MLPLogic(current_dim + relation_dim, self.hidden_dims[i], []))
            current_dim = self.hidden_dims[i] + (self.input_dim if self.residual_input else 0)
        self.step_linears.append(MLPLogic(current_dim + relation_dim, self.output_dim, []))

    def forward(self, f):
        t = f.size(1)
        if self.until:
            r = torch.cat((
                interval_pooling(f, reduction='max'),
                interval_pooling(f, reduction='min')
            ), dim=-1)
        else:
            r = interval_pooling(f, reduction='max')

        current = r[:, :, t - 1]
        for i in range(self.nr_steps):
            if self.residual_input:
                current = torch.cat((current, r[:, :, t - 1]), dim=-1)
            linear_input = torch.cat((jactorch.add_dim(current, 1, t), r), dim=-1)
            # current = self.step_linears[i](linear_input).max(dim=2)[0]
            current = backward_pooling_1d1d(self.step_linears[i](linear_input), 'broadcast')
        return current


class TemporalLogicMachineDPV2(TemporalLogicMachineDP):
    def __init__(self, input_dim, output_dim, nr_steps, hidden_dims=None, residual_input=True,
                 pooling_range=(0.25, 0.5), soft_pooling=False, soft_transform_pooling=False):
        super().__init__(input_dim, output_dim, nr_steps, hidden_dims=hidden_dims, residual_input=residual_input)
        self.pooling_range = pooling_range
        self.soft_pooling = soft_pooling
        self.soft_transform_pooling = soft_transform_pooling

    def _gen_pooling_range_mask(self, tensor):
        low, high = self.pooling_range[0], self.pooling_range[1]
        nr_time_steps = tensor.size(1)

        low_size, high_size = int(nr_time_steps * low), int(nr_time_steps * high + 0.9999)
        a = torch.arange(nr_time_steps, dtype=torch.int64, device=tensor.device)
        a, b = jactorch.meshgrid(a, dim=0)
        mask = (a + low_size <= b) * (b <= a + high_size)

        return mask.float().unsqueeze(0).unsqueeze(-1)

    def forward(self, f, beta=0):
        t = f.size(1)
        if self.soft_pooling:
            r = torch.cat((
                interval_pooling(f, reduction='softmax', beta=beta),
                interval_pooling(f, reduction='softmin', beta=beta)
            ), dim=-1)
        else:
            r = torch.cat((
                interval_pooling(f, reduction='max'),
                interval_pooling(f, reduction='min')
            ), dim=-1)

        if self.pooling_range is not None:
            mask = self._gen_pooling_range_mask(r)
            if len(mask.size()) < len(r.size()):
                mask = mask.view(mask.size() + (1,) * (len(r.size()) - len(mask.size())))
        else:
            mask = 1
        current = (r * mask).max(dim=2)[0]
        for i in range(self.nr_steps):
            if self.residual_input:
                current = torch.cat((current, f), dim=-1)
            linear_input = torch.cat((jactorch.add_dim(current, 1, t), r), dim=-1)
            current = self.step_linears[i](linear_input)
            current = current * mask
            if self.soft_transform_pooling:
                from math import exp
                scale = exp(beta)
                current = (current * torch.nn.functional.softmax(current / scale, dim=2)).sum(dim=2)
            else:
                current = current.max(dim=2)[0]
        return current
