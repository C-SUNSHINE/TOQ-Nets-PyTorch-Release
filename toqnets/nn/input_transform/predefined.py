#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : predefined.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import torch
from torch import nn

from .basic import Inequality
from toqnets.nn.input_transform.primitives import AlignDifferential, Distance


class N_aryPrimitivesPredefined(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 0
        self.primitive_list = []
        self.ineqs = nn.ModuleDict({})

    def reset_parameters(self, parameter_name):
        for k in self.primitive_list:
            self.ineqs[k].reset_parameters(parameter_name, name=k)

    def get_descriptions(self):
        descriptions = []
        for k in self.primitive_list:
            descriptions += self.ineqs[k].get_descriptions(name=k)
        return descriptions


class NullaryPrimitivesPredefined(N_aryPrimitivesPredefined):
    def __init__(self, cmp_dim=10):
        super().__init__()

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, out_dim]
        """
        return None


class UnaryPrimitivesPredefined(N_aryPrimitivesPredefined):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.differential = AlignDifferential()
        self.primitive_list = ['acc', 'pos_z', 'speed_xy', 'speed_z']
        self.ineqs.update({
            'pos_x': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'pos_y': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'pos_z': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'speed_xy': Inequality(out_dim=cmp_dim, distribution='normal'),
            'speed_z': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'acc': Inequality(out_dim=cmp_dim, distribution='normal')
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, out_dim]
        """
        velocity = self.differential(states)
        acc = self.differential(velocity)
        ineqs_inputs = {
            'pos_x': states[:, :, :, 0],
            'pos_y': states[:, :, :, 1],
            'pos_z': states[:, :, :, 2],
            'speed_xy': torch.norm(velocity[:, :, :, :2], p=2, dim=3),
            'speed_z': torch.norm(velocity[:, :, :, 2:], p=1, dim=3),
            'acc': torch.norm(acc, p=2, dim=3),
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output


class BinaryPrimitivesPredefined(N_aryPrimitivesPredefined):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.distance = Distance()
        self.primitive_list = ['dist_xy']
        self.ineqs.update({
            'dist_xy': Inequality(out_dim=cmp_dim, distribution='normal')
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, n_agents, out_dim]
        """
        n_agents = states.size(2)
        p1 = states.unsqueeze(2).repeat(1, 1, n_agents, 1, 1)
        p2 = states.unsqueeze(3).repeat(1, 1, 1, n_agents, 1)
        ineqs_inputs = {
            'dist_xy': self.distance(p1, p2, dim_index=(0, 1)).squeeze(4),
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output


class NullaryPrimitivesPredefined_v2(N_aryPrimitivesPredefined):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.differential = AlignDifferential()
        self.primitive_list = ['ball_acc', 'ball_pos_z', 'ball_speed']
        self.ineqs.update({
            'ball_acc': Inequality(out_dim=cmp_dim, distribution='normal'),
            'ball_pos_z': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'ball_speed': Inequality(out_dim=cmp_dim, distribution='normal'),
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, out_dim]
        """
        velocity = self.differential(states)
        acc = self.differential(velocity)
        ineqs_inputs = {
            'ball_acc': torch.norm(acc[:, :, 0, :], p=2, dim=2),
            'ball_pos_z': states[:, :, 0, 2],
            'ball_speed': torch.norm(states[:, :, 0, :], p=2, dim=2),
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output


class UnaryPrimitivesPredefined_v2(N_aryPrimitivesPredefined):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.differential = AlignDifferential()
        self.primitive_list = ['acc', 'pos_z', 'speed', 'dist_to_ball']
        self.distance = Distance()
        self.ineqs.update({
            'acc': Inequality(out_dim=cmp_dim, distribution='normal'),
            'pos_z': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'speed': Inequality(out_dim=cmp_dim, distribution='normal'),
            'dist_to_ball': Inequality(out_dim=cmp_dim, distribution='normal'),
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, out_dim]
        """
        velocity = self.differential(states)
        acc = self.differential(velocity)
        n_agents = states.size(2)
        p1 = states.unsqueeze(2).repeat(1, 1, n_agents, 1, 1)
        p2 = states.unsqueeze(3).repeat(1, 1, 1, n_agents, 1)
        dist = self.distance(p1, p2).squeeze(4)
        ineqs_inputs = {
            'pos_z': states[:, :, 1:, 2],
            'speed': torch.norm(velocity[:, :, 1:, :], p=2, dim=3),
            'acc': torch.norm(acc[:, :, 1:, :], p=2, dim=3),
            'dist_to_ball': dist[:, :, 0, 1:],
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output


class BinaryPrimitivesPredefined_v2(N_aryPrimitivesPredefined):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.distance = Distance()
        self.primitive_list = ['dist']
        self.ineqs.update({
            'dist': Inequality(out_dim=cmp_dim, distribution='normal')
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, n_agents, out_dim]
        """
        n_agents = states.size(2)
        p1 = states.unsqueeze(2).repeat(1, 1, n_agents, 1, 1)
        p2 = states.unsqueeze(3).repeat(1, 1, 1, n_agents, 1)
        ineqs_inputs = {
            'dist': self.distance(p1, p2, dim_index=(0, 1)).squeeze(4)[:, :, 1:, 1:],
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output
