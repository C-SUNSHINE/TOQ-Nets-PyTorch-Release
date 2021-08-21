#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : toyota.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import torch
from torch import nn

from toqnets.nn.input_transform.primitives import AlignDifferential, Distance, Angle
from toqnets.nn.nlm import exclude_mask
from .basic import Inequality, AppendDim


class N_aryPrimitivesToyotaJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 0
        self.primitive_list = []
        self.ineqs = nn.ModuleDict({})

    def reset_parameters(self, parameter_name):
        for k in self.primitive_list:
            self.ineqs[k].reset_parameters(parameter_name, name=k)


class NullaryPrimitivesToyotaJoint(N_aryPrimitivesToyotaJoint):
    def __init__(self, cmp_dim=10):
        super().__init__()

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_joints, state_dim]
        return [batch, length, out_dim]
        """
        return None


class UnaryPrimitivesToyotaJoint(N_aryPrimitivesToyotaJoint):
    def __init__(self, cmp_dim=10):
        super().__init__()
        self.differential = AlignDifferential()
        self.primitive_list = ['pos_x', 'pos_y', 'pos_z']
        self.ineqs.update({
            'pos_x': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'pos_y': Inequality(out_dim=cmp_dim, distribution='uniform'),
            'pos_z': Inequality(out_dim=cmp_dim, distribution='uniform'),
            # 'speed_x': Inequality(out_dim=cmp_dim, distribution='normal'),
            # 'speed_y': Inequality(out_dim=cmp_dim, distribution='normal'),
            # 'speed_z': Inequality(out_dim=cmp_dim, distribution='normal'),
            # 'speed': Inequality(out_dim=cmp_dim, distribution='normal'),
            # 'acc_x': Inequality(out_dim=cmp_dim, distribution='normal'),
            # 'acc_y': Inequality(out_dim=cmp_dim, distribution='normal'),
            # 'acc_z': Inequality(out_dim=cmp_dim, distribution='normal'),
            # 'acc': Inequality(out_dim=cmp_dim, distribution='normal'),
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_joints, state_dim]
        return [batch, length, n_joints, out_dim]
        """
        velocity = self.differential(states)
        acc = self.differential(velocity)
        ineqs_inputs = {
            'pos_x': states[:, :, :, 0],
            'pos_y': states[:, :, :, 1],
            'pos_z': states[:, :, :, 2],
            # 'speed_x': torch.norm(velocity[:, :, :, 0:1], p=2, dim=3),
            # 'speed_y': torch.norm(velocity[:, :, :, 1:2], p=2, dim=3),
            # 'speed_z': torch.norm(velocity[:, :, :, 2:3], p=2, dim=3),
            'speed': torch.norm(velocity, p=2, dim=3),
            # 'acc_x': torch.norm(acc[:, :, :, 0:1], p=2, dim=3),
            # 'acc_y': torch.norm(acc[:, :, :, 1:2], p=2, dim=3),
            # 'acc_z': torch.norm(acc[:, :, :, 2:3], p=2, dim=3),
            'acc': torch.norm(acc, p=2, dim=3),
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta, name=k, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output


class BinaryPrimitivesToyotaJoint(N_aryPrimitivesToyotaJoint):
    n_joints = 13
    edges = [(0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11), (4, 5), (4, 10), (5, 11), (10, 11),
             (11, 12), (10, 12)]

    def __init__(self, cmp_dim=10):
        super().__init__()
        self.distance = Distance()
        self.differential = AlignDifferential()
        self.primitive_list = ['dist', 'edge']
        self.ineqs.update({
            'dist': Inequality(out_dim=cmp_dim, distribution='uniform'),
            # 'app_vel': Inequality(out_dim=cmp_dim, distribution='normal'),
            'edge': AppendDim(out_dim=1),
        })
        self.adj = torch.zeros(self.n_joints, self.n_joints)
        for e in self.edges:
            self.adj[e[0], e[1]] = 1
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_joints, state_dim]
        return [batch, length, n_joints, n_joints, out_dim]
        """
        batch, length, n_agents, state_dim = states.size()
        p1 = states.unsqueeze(2).repeat(1, 1, n_agents, 1, 1)
        p2 = states.unsqueeze(3).repeat(1, 1, 1, n_agents, 1)
        distances = self.distance(p1, p2, dim_index=(0, 1, 2))
        app_velocity = self.differential(distances)
        edge_input = self.adj.to(states.device).view(1, 1, n_agents, n_agents).repeat(batch, length, 1, 1)
        ineqs_inputs = {
            'dist': distances.squeeze(4),
            'app_vel': app_velocity.squeeze(4),
            'edge': edge_input,
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta=beta, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output


class TrinaryPtimitivesToyotaJoint(N_aryPrimitivesToyotaJoint):
    n_joints = 13
    edges = [(0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11), (4, 5), (4, 10), (5, 11), (10, 11),
             (11, 12), (10, 12)]

    def __init__(self, cmp_dim=10):
        super().__init__()
        self.angle = Angle()
        self.differential = AlignDifferential()
        self.primitive_list = ['angle']
        self.ineqs.update({
            'angle': Inequality(out_dim=cmp_dim, distribution='uniform'),
            # 'angle_vel': Inequality(out_dim=cmp_dim, distribution='normal')
        })
        self.out_dim = sum([self.ineqs[k].out_dim for k in self.primitive_list])
        self.adj = torch.zeros(self.n_joints, self.n_joints)
        for e in self.edges:
            self.adj[e[0], e[1]] = 1
        self.triplet = self.adj.unsqueeze(0) * self.adj.unsqueeze(2)

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_joints, state_dim]
        return [batch, length, n_joints, n_joints, n_joints, out_dim]
        """
        batch, length, n_agents, state_dim = states.size()
        assert n_agents == self.n_joints
        n_agents = states.size(2)
        p1 = states.view(batch, length, n_agents, 1, 1, state_dim).repeat(1, 1, 1, n_agents, n_agents, 1)
        p2 = states.view(batch, length, 1, n_agents, 1, state_dim).repeat(1, 1, n_agents, 1, n_agents, 1)
        p3 = states.view(batch, length, 1, 1, n_agents, state_dim).repeat(1, 1, n_agents, n_agents, 1, 1)
        e_mask = exclude_mask(p1, cnt=3, dim=2)
        p1 *= e_mask.float()
        p2 *= e_mask.float()
        p3 *= e_mask.float()
        angle = self.angle(p1, p2, p3)
        # angle = angle * e_mask.float()[:, :, :, :, :, :1]
        angle = angle * self.triplet.to(angle.device).view(1, 1, n_agents, n_agents, n_agents, 1)
        angle_vel = self.differential(angle)
        ineqs_inputs = {
            'angle': angle.squeeze(5),
            'angle_vel': angle_vel.squeeze(5),
        }
        output = torch.cat(
            [self.ineqs[k](ineqs_inputs[k], beta, **kwargs) for k in self.primitive_list],
            dim=-1
        )
        return output


class NullaryPrimitivesToyotaSkeleton(N_aryPrimitivesToyotaJoint):
    n_joints = 13

    def __init__(self, cmp_dim=10):
        super().__init__()
        self.unary = UnaryPrimitivesToyotaJoint(cmp_dim=cmp_dim)
        self.binary = BinaryPrimitivesToyotaJoint(cmp_dim=cmp_dim)
        self.trinary = TrinaryPtimitivesToyotaJoint(cmp_dim=cmp_dim)
        self.out_dim = self.unary.out_dim * self.n_joints + \
                       self.binary.out_dim * self.n_joints * self.n_joints + \
                       self.trinary.out_dim * self.n_joints * self.n_joints * self.n_joints

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_joints, state_dim]
        return [batch, length, out_dim]
        """
        batch, length, n_agents, state_dim = states.size()
        unary = self.unary(states, beta, **kwargs).view(batch, length, -1)
        binary = self.binary(states, beta, **kwargs).view(batch, length, -1)
        trinary = self.trinary(states, beta, **kwargs).view(batch, length, -1)
        return torch.cat([unary, binary, trinary], dim=2)


class UnaryPrimitivesToyotaSkeleton(N_aryPrimitivesToyotaJoint):
    def __init__(self, cmp_dim=10):
        super().__init__()

    def forward(self, states, beta=0, **kwargs):
        """
        :param states: [batch, length, n_joints, state_dim]
        return [batch, length, out_dim]
        """
        return None
