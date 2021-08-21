#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : something_else.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import torch
from torch import nn

from toqnets.nn.input_transform.primitives import AlignDifferential, Distance
from .basic import AppendDim, Inequality


class N_aryPrimitivesSomethingElse(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 0
        self.ineqs = nn.ModuleDict({})

    def reset_parameters(self, parameter_name):
        for k in self.ineqs:
            self.ineqs[k].reset_parameters(parameter_name, name=k)


class NullaryPrimitivesSomethingElse(N_aryPrimitivesSomethingElse):
    def __init__(self, cmp_dim=10):
        super().__init__()

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, out_dim]
        """
        return None


class UnaryPrimitivesSomethingElse(N_aryPrimitivesSomethingElse):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.differential = AlignDifferential()
        self.ineqs.update({
            'appear': AppendDim(dim=1),
            'pos_x': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'pos_y': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'width': Inequality(out_dim=cmp_dim, distribution='normal'),
            'height': Inequality(out_dim=cmp_dim, distribution='normal'),
            'vel_x': Inequality(out_dim=cmp_dim, distribution='normal'),
            'vel_y': Inequality(out_dim=cmp_dim, distribution='normal'),
            'speed': Inequality(out_dim=cmp_dim, distribution='normal'),
            'acc_x': Inequality(out_dim=cmp_dim, distribution='normal'),
            'acc_y': Inequality(out_dim=cmp_dim, distribution='normal'),
            'acc': Inequality(out_dim=cmp_dim, distribution='normal'),
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.ineqs])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, out_dim]
        """
        velocity = self.differential(states)
        acc = self.differential(velocity)
        ineqs_inputs = {
            'appear': states[:, :, :, 4],
            'pos_x': states[:, :, :, 0],
            'pos_y': states[:, :, :, 1],
            'width': states[:, :, :, 2],
            'height': states[:, :, :, 3],
            'vel_x': velocity[:, :, :, 0],
            'vel_y': velocity[:, :, :, 1],
            'speed': torch.norm(velocity[:, :, :, :2], p=2, dim=3),
            'acc_x': acc[:, :, :, 0],
            'acc_y': acc[:, :, :, 1],
            'acc': torch.norm(acc[:, :, :, :2], p=2, dim=3),
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta=beta, name=k, **kwargs) for k in ineqs_inputs.keys()],
            dim=-1
        )
        return output


class BinaryPrimitivesSomethingElse(N_aryPrimitivesSomethingElse):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.distance = Distance()
        self.differential = AlignDifferential()
        self.ineqs.update({
            'dist': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'app_vel': Inequality(out_dim=cmp_dim, distribution='normal'),
            'overlap_x': Inequality(out_dim=1, distribution=None),
            'overlap_y': Inequality(out_dim=1, distribution=None),
            'contain_x': Inequality(out_dim=1, distribution=None),
            'contain_y': Inequality(out_dim=1, distribution=None),
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.ineqs])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, n_agents, out_dim]
        """
        n_agents = states.size(2)
        p1 = states.unsqueeze(2).repeat(1, 1, n_agents, 1, 1)
        p2 = states.unsqueeze(3).repeat(1, 1, 1, n_agents, 1)
        distances = self.distance(p1, p2, dim_index=(0, 1))
        app_velocity = self.differential(distances)
        distances_x = self.distance(p1, p2, dim_index=(0,))
        distances_y = self.distance(p1, p2, dim_index=(1,))
        ineqs_inputs = {
            'dist': distances.squeeze(4),
            'app_vel': app_velocity.squeeze(4),
            'overlap_x': distances_x.squeeze(4) - (p1[:, :, :, :, 2] + p2[:, :, :, :, 2]) / 2,
            'overlap_y': distances_y.squeeze(4) - (p1[:, :, :, :, 3] + p2[:, :, :, :, 3]) / 2,
            'contain_x': distances_x.squeeze(4) - (p1[:, :, :, :, 2] - p2[:, :, :, :, 2]) / 2,
            'contain_y': distances_y.squeeze(4) - (p1[:, :, :, :, 3] - p2[:, :, :, :, 3]) / 2,
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta=beta, **kwargs) for k in ineqs_inputs.keys()],
            dim=-1
        )
        return output
