#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : propnet.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# Modified from https://github.com/YunzhuLi/PropNet/blob/master/models.py
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

from torch import nn

from .utils import apply_last_dim


class AgentEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, h_dim),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, out_dim),
                                     nn.ReLU())

    def forward(self, x):
        return apply_last_dim(self.encoder, x)


class RelationEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, h_dim),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, out_dim),
                                     nn.ReLU())

    def forward(self, x):
        return apply_last_dim(self.encoder, x)


class Propagator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, out_dim),
                                     nn.ReLU())

    def forward(self, x):
        return apply_last_dim(self.encoder, x)
