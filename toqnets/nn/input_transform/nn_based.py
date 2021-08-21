#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : nn_based.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import torch
from torch import nn

from toqnets.nn.utils import apply_last_dim


class PredicateNN(nn.Module):
    def __init__(self, in_dim, h_dims=None, out_dim=1, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.h_dims = h_dims if h_dims is not None else []
        self.out_dim = out_dim
        layers = []
        cur_dim = in_dim
        for i in range(len(self.h_dims)):
            layers.append(nn.Linear(cur_dim, self.h_dims[i]))
            layers.append(nn.ReLU())
            cur_dim = self.h_dims[i]
        layers.append(nn.Linear(cur_dim, out_dim))
        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, states, **kwargs):
        """
        :param states: [batch, length, n_agents, ... , in_dim]
        """
        x = apply_last_dim(self.layers, states)
        x = self.sigmoid(x)
        return x

    def reset_parameters(self, *args, **kwargs):
        pass


def expand_input(states, m):
    """
    :param states: [batch, length, n_agents, state_dim]
    return [batch, length, n_agents, ..., n_agents, out_dim]
    """
    if m == 0:
        return None
    inputs = []
    size = states.size()
    n_agents = states.size(2)
    for i in range(m):
        view_size = size[:2] + tuple([(n_agents if j == i else 1) for j in range(m)]) + size[3:]
        repeat_size = (1, 1) + tuple([(1 if j == i else n_agents) for j in range(m)]) + (1,)
        inputs.append(states.view(*view_size).repeat(*repeat_size))

    return torch.cat(inputs, dim=-1)


class N_aryPrimitivesNN(nn.Module):
    def __init__(self, member, in_dim, h_dims, out_dim, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.h_dims = h_dims
        self.out_dim = out_dim
        self.member = member
        if in_dim == 0:
            self.out_dim = 0
            self.predicates = lambda x: None
        else:
            self.predicates = PredicateNN(in_dim, h_dims, out_dim)

    def forward(self, states, beta=0, **kwargs):
        return self.predicates(expand_input(states, self.member))

    def reset_parameters(self, parameter_name):
        pass
        # for k in self.predicates:
        # self.predicates[k].reset_parameters(parameter_name, name=k)


class NullaryPrimitivesNN(N_aryPrimitivesNN):
    def __init__(self, state_dim=0, h_dims=None, out_dim=10, **kwargs):
        super().__init__(0, 0, h_dims, out_dim, **kwargs)

    def forward(self, states, beta=0, **kwargs):
        return self.predicates(states)


class UnaryPrimitivesNN(N_aryPrimitivesNN):
    def __init__(self, state_dim=0, h_dims=None, out_dim=10, **kwargs):
        super().__init__(1, state_dim, h_dims, out_dim, **kwargs)


class BinaryPrimitivesNN(N_aryPrimitivesNN):
    def __init__(self, state_dim=0, h_dims=None, out_dim=10, **kwargs):
        super().__init__(2, state_dim * 2, h_dims, out_dim, **kwargs)


class TrinaryPrimitiveNN(N_aryPrimitivesNN):
    def __init__(self, state_dim=0, h_dims=None, out_dim=10, **kwargs):
        super().__init__(3, state_dim * 3, h_dims, out_dim, **kwargs)
