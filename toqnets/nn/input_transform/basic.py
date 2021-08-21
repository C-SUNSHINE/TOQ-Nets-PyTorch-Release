#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : basic.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import math

import torch
from torch import nn


class AppendDim(nn.Module):
    """
    Append a new dim to states with size out_dim
    """

    def __init__(self, out_dim=1):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, states, **kwargs):
        x = states.unsqueeze(len(states.size()))
        x = x.repeat(*([1] * len(states.size()) + [self.out_dim]))
        return x

    def reset_parameters(self, *args, **kwargs):
        pass


class SoftCmp(nn.Module):
    """
    Sigmoid((x - y) / e^beta)
    """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, beta):
        return self.sigmoid((x - y) / math.exp(beta))


class Normalize(nn.Module):
    def __init__(self, distribution=None, **kwargs):
        super().__init__()
        self.distribution = distribution
        self.data_ = []
        if distribution is None:
            pass
        elif distribution == 'normal':
            mean = kwargs['mean'] if 'mean' in kwargs else 0
            std = kwargs['std'] if 'std' in kwargs else 1
            self.param = nn.Parameter(torch.Tensor([mean, std]), False)
        elif distribution == 'uniform':
            vmin = kwargs['minv'] if 'minv' in kwargs else 0
            vmax = kwargs['maxv'] if 'maxv' in kwargs else 1
            self.param = nn.Parameter(torch.Tensor([vmin, vmax]), False)
        else:
            raise NotImplementedError()

    def forward(self, x, keep_data=False):
        if keep_data:
            self.data_.append(x.detach().cpu().view(-1))
            return x
        if self.distribution is None:
            return x
        elif self.distribution == 'normal':
            mean = self.param[0]
            std = self.param[1]
            return (x - mean) / std
        elif self.distribution == 'uniform':
            vmin = self.param[0]
            vmax = self.param[1]
            return (x - vmin) / (vmax - vmin + 1e-5)
        else:
            raise NotImplementedError()

    def reset_parameters(self, name=None):
        assert len(self.data_) > 0
        data = torch.cat(self.data_)
        self.data_ = []
        if self.distribution is None:
            pass
        elif self.distribution == 'normal':
            with torch.no_grad():
                self.param[0] = data.mean().item()
                self.param[1] = data.std().item()
            if name is not None:
                print("Reset %s normal: mean=%f std=%f" % (name, float(self.param[0]), float(self.param[1])))
        elif self.distribution == 'uniform':
            with torch.no_grad():
                self.param[0] = data.min().item()
                self.param[1] = data.max().item()
            if name is not None:
                print("Reset %s uniform: min=%f max=%f" % (name, float(self.param[0]), float(self.param[1])))
        else:
            raise NotImplementedError()

    def recover_threshold(self, x):
        if self.distribution is None:
            return x
        elif self.distribution == 'normal':
            return x * float(self.param[1]) + float(self.param[0])
        elif self.distribution == 'uniform':
            return x * float(self.param[1] - self.param[0] + 1e-5) + float(self.param[0])
        else:
            raise NotImplementedError()

    def init_thresholds(self, x):
        if self.distribution is None:
            nn.init.normal_(x, 0, 1)
        elif self.distribution == 'normal':
            nn.init.normal_(x, 0, 1)
        elif self.distribution == 'uniform':
            nn.init.uniform_(x, 0, 1)
        else:
            raise NotImplementedError()


class Inequality(nn.Module):
    def __init__(self, out_dim=1, distribution=None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.thresholds = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        self.distribution = distribution
        self.normalize = Normalize(distribution)
        self.cmp = SoftCmp()
        self.normalize.init_thresholds(self.thresholds)

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, ... ]
        """
        states_expand = states.view(*(states.size() + (1,)))
        estimate_parameters = 'estimate_parameters' in kwargs and kwargs['estimate_parameters']
        states_expand = self.normalize(states_expand, keep_data=estimate_parameters)

        return self.cmp(states_expand, self.thresholds.view(*([1] * len(states.size()) + [self.out_dim])), beta)

    def reset_parameters(self, parameter_name, name=None):
        if parameter_name == 'primitive_inequality':
            self.normalize.reset_parameters(name=name)
            self.normalize.init_thresholds(self.thresholds)

    def get_descriptions(self, name='Inequality'):
        theta = self.thresholds.detach().cpu().view(self.out_dim)
        descroptions = []
        for k in range(theta.size(0)):
            t = self.normalize.recover_threshold(theta[k])
            if 'speed' in name:
                t = t * 8
            if 'acc' in name:
                t = t * 64
            descroptions.append("%s > %.2lf" % (name, t))
        return descroptions
