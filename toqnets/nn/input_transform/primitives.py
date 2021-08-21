#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : primitives.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import math

import torch
from torch import nn

from toqnets.nn.utils import apply_last_dim, init_weights, Ternary


class Position(nn.Module):
    def __init__(self, position_extractor=lambda x: x):
        super().__init__()
        self.position_extractor = position_extractor

    def forward(self, states):
        """
        :param states: [batch, length, n_agents, state_dim]
        """
        return apply_last_dim(self.position_extractor, states)

    def show(self, name='Position', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = x\'s first three dims' % name)


class Differential(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def new_length(self, length):
        new_length = (length + self.padding * 2 - self.kernel_size + 1) // self.stride
        return new_length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """

        batch, length, n_agents, state_dim = states.size()
        padding = self.padding
        kernel_size = self.kernel_size
        stride = self.stride
        if padding != 0:
            if padding > 0:
                states = torch.cat([
                    states[:, :1].repeat(1, padding, 1, 1),
                    states,
                    states[:, -1:].repeat(1, padding, 1, 1),
                ], dim=1)
            else:
                states = states[:, -padding:padding]
        new_length = (length + padding * 2 - kernel_size + 1) // stride
        differentials = states[:, 0: new_length * stride:stride
                        ] - states[:, kernel_size - 1:kernel_size - 1 + new_length * stride:stride]
        return differentials

    def show(self, name='Differential', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = Differential(ks=%d, stride=%d, padding=%d)' % (
            name, self.kernel_size, self.stride, self.padding))


class AlignDifferential(nn.Module):
    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        padded_states = torch.cat([states[:, 0:1] * 2 - states[:, 1:2], states, states[:, -1:] * 2 - states[:, -2:-1]],
                                  dim=1)
        return (padded_states[:, 2:] - padded_states[:, :-2]) / 2

    def show(self, name='AlignDifferential', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = AlignDifferential()' % (name,))


class Angle(nn.Module):
    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, A, O, B):
        assert O.size() == A.size() and O.size() == B.size() and O.size()[-1] == 3
        a = torch.norm(A - O, p=2, dim=-1, keepdim=True)
        b = torch.norm(B - O, p=2, dim=-1, keepdim=True)
        c = torch.norm(A - B, p=2, dim=-1, keepdim=True)
        cos = (a * a + b * b - c * c) / (a * b * 2 + 1e-5)
        return torch.acos(cos)

    def show(self, name='Angle', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = Angle()' % (name,))


class MaxPoolTrinary(nn.Module):
    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        assert states.size(1) >= 3
        side_length = (states.size(1) + 1) // 3
        return torch.cat([
            torch.max(states[:, :side_length], dim=1, keepdim=True)[0],
            torch.max(states[:, side_length:-side_length], dim=1, keepdim=True)[0],
            torch.max(states[:, -side_length:], dim=1, keepdim=True)[0]
        ], dim=1)

    def show(self, name='MaxPoolTrinary', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = MaxPoolTrinary()' % (name,))


class MinPoolTrinary(nn.Module):
    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        assert states.size(1) >= 3
        side_length = (states.size(1) + 1) // 3
        return torch.cat([
            torch.min(states[:, :side_length], dim=1, keepdim=True)[0],
            torch.min(states[:, side_length:-side_length], dim=1, keepdim=True)[0],
            torch.min(states[:, -side_length:], dim=1, keepdim=True)[0]
        ], dim=1)

    def show(self, name='MinPoolTrinary', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = MinPoolTrinary()' % (name,))


def get_int_dim_index(name):
    if isinstance(name, int):
        return name
    name_list = 'axyz'
    assert name in name_list
    return [i for i in range(len(name_list)) if name_list[i] == name][0] - 1


class Length(nn.Module):
    def __init__(self, dim_index=-1):
        super().__init__()
        self.dim_index = dim_index

    def forward(self, states, dim_index=None):
        if dim_index is None:
            dim_index = self.dim_index
        if isinstance(dim_index, int):
            dim_index = [dim_index]
        else:
            dim_index = [get_int_dim_index(x) for x in dim_index]

        if -1 in dim_index:
            extractor = lambda x: torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        else:
            extractor = lambda x: torch.sqrt(torch.sum(x[:, dim_index].pow(2), dim=1, keepdim=True))
        return apply_last_dim(extractor, states)

    def show(self, name='Length', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = |x\'s dim %s|' % (name, str(self.dim_index)))


class Component(nn.Module):
    def __init__(self, dim_index=-1):
        super().__init__()
        self.dim_index = dim_index

    def forward(self, states, dim_index=None):
        if dim_index is None:
            dim_index = self.dim_index
        if isinstance(dim_index, int):
            dim_index = [dim_index]
        else:
            dim_index = [get_int_dim_index(x) for x in dim_index]

        if -1 in dim_index:
            extractor = lambda x: x
        else:
            extractor = lambda x: x[:, dim_index]
        return apply_last_dim(extractor, states)

    def show(self, name='Component', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = x\'s dim %s' % (name, str(self.dim_index)))


class Distance(nn.Module):
    def __init__(self, dim_index=-1):
        super().__init__()
        self.dim_index = dim_index
        self.length = Length(dim_index)

    def forward(self, states1, states2, dim_index=None):
        return self.length(states1 - states2, dim_index)

    def show(self, name='Distance', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x1, x2) = |x1 - x2|' % name)


class SoftCompare(nn.Module):
    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * (0 if alpha is None else alpha), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1) * (0 if beta is None else beta), requires_grad=True)
        if alpha is None:
            nn.init.normal_(self.alpha.data, 0, 1)
        else:
            self.alpha.requires_grad_(False)
        if beta is not None:
            self.beta.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raise NotImplementedError


class SoftSmall(SoftCompare):
    """
    Sigmoid((alpha - x) / e^beta)
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__(alpha, beta)

    def forward(self, x, beta=None):
        alpha = self.alpha
        if beta is None:
            beta = self.beta
        return self.sigmoid((alpha - x) / torch.exp(beta))

    def show(self, name='SoftSmall', indent=0, log=print, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        beta = kwargs['beta'] if 'beta' in kwargs else self.beta
        log(' ' * indent + '- %s(x) = Sigmoid((%lf - x) / %lf)' % (name, alpha, math.exp(beta)))


class SoftLarge(SoftCompare):
    """
    Sigmoid((x - alpha) / e^beta)
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__(alpha, beta)

    def forward(self, x, beta=None):
        alpha = self.alpha
        if beta is None:
            beta = self.beta
        return self.sigmoid((x - alpha) / torch.exp(beta))

    def show(self, name='SoftLarge', indent=0, log=print, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        beta = kwargs['beta'] if 'beta' in kwargs else self.beta
        log(' ' * indent + '- %s(x) = Sigmoid((x - %lf) / %lf)' % (name, alpha, math.exp(beta)))


class WithBall(nn.Module):
    def __init__(self, alpha=None, beta=None, trinary=False):
        super().__init__()
        self.position = Position()
        self.trinary = trinary
        self.distance = Distance()
        if trinary:
            self.min_pool_trinary = MinPoolTrinary()
        self.small = SoftSmall(alpha, beta)

    def new_length(self, length):
        return 3 if self.trinary else length

    def get_descriptions(self, n_agents, length):
        n_players = n_agents // 2
        agent_name = ['ball'] + ['A' + str(i) for i in range(1, n_players + 1)] + ['B' + str(i) for i in
                                                                                   range(1, n_players + 1)]
        res = []
        if self.trinary:
            trinary_name = ['pre', 'act', 'eff']
            for i in range(3):
                for p in range(1, n_agents):
                    res.append('WithBall(%s, %s)' % (agent_name[p], trinary_name[i]))
        else:
            new_length = self.new_length(length)
            for i in range(0, new_length):
                for p in range(1, n_agents):
                    res.append('WithBall(%s, %.2f)' % (agent_name[p], (i + .5) / new_length))
        return res

    def forward(self, states, beta=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents - 1, 1]
        """
        batch, length, n_agents, state_dim = states.size()
        ball_pos = self.position(states[:, :, :1])
        player_pos = self.position(states[:, :, 1:])
        dists = self.distance(player_pos, ball_pos)
        if self.trinary:
            dists = self.min_pool_trinary(dists)
        small = self.small(dists, beta=beta)
        return small

    def show(self, name='WithBall', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(p) = small(distance(p, ball))' % name)
        self.distance.show('distance', indent + 2, **kwargs)
        self.small.show('small', indent + 2, log=log, **kwargs)


class Moving(nn.Module):
    def __init__(self, alpha_ball=None, beta_ball=None, alpha_player=None, beta_player=None, trinary=False):
        super().__init__()
        self.position = Position()
        self.trinary = trinary
        if trinary:
            self.speed = nn.Sequential(AlignDifferential(), Length('xy'), MaxPoolTrinary())
        else:
            self.speed = nn.Sequential(Differential(kernel_size=4, stride=3, padding=-1), Length('xy'))
        self.ball_large = SoftLarge(alpha_ball, beta_ball)
        self.player_large = SoftLarge(alpha_player, beta_player)

    def new_length(self, length):
        return 3 if self.trinary else self.speed[0].new_length(length)

    def get_descriptions(self, n_agents, length):
        n_players = n_agents // 2
        agent_name = ['ball'] + ['A' + str(i) for i in range(1, n_players + 1)] + ['B' + str(i) for i in
                                                                                   range(1, n_players + 1)]
        res = []
        if self.trinary:
            trinary_name = ['pre', 'act', 'eff']
            for i in range(3):
                for p in range(n_agents):
                    res.append('Moving(%s, %s)' % (agent_name[p], trinary_name[i]))
        else:
            new_length = self.new_length(length)
            for i in range(0, new_length):
                for p in range(n_agents):
                    res.append('Moving(%s, %.2f)' % (agent_name[p], (i + .5) / new_length))
        return res

    def forward(self, states, beta=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, new_length, n_agents, 1]
        """
        batch, length, n_agents, state_dim = states.size()
        ball_pos = self.position(states[:, :, :1])
        player_pos = self.position(states[:, :, 1:])
        ball_speed_large = self.ball_large(self.speed(ball_pos), beta=beta)
        player_speed_large = self.player_large(self.speed(player_pos), beta=beta)
        return torch.cat([
            ball_speed_large, player_speed_large
        ], dim=2)

    def show(self, name='Moving', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(b/p) = ball_large/player_large(length(delta(position(b/p))))' % name)
        self.position.show('position', indent + 2, log=log, **kwargs)
        self.speed[0].show('delta', indent + 2, log=log, **kwargs)
        self.speed[1].show('length', indent + 2, log=log, **kwargs)
        self.ball_large.show('ball_large', indent + 2, log=log, **kwargs)
        self.player_large.show('player_large', indent + 2, log=log, **kwargs)


class BallTouched(nn.Module):
    def __init__(self, alpha=None, beta=None, trinary=False):
        super().__init__()
        self.position = Position()
        self.trinary = trinary
        if trinary:
            self.acceleration = nn.Sequential(
                AlignDifferential(),
                AlignDifferential(),
                Length(),
                MaxPoolTrinary()
            )
        else:
            self.acceleration = nn.Sequential(
                Differential(kernel_size=4, stride=3, padding=-1),
                Differential(kernel_size=3, stride=1),
                Length()
            )
        self.large = SoftLarge(alpha, beta)

    def new_length(self, length):
        return 3 if self.trinary else self.acceleration[1].new_length(self.acceleration[0].new_length(length))

    def get_descriptions(self, n_agents, length):
        n_players = n_agents // 2
        agent_name = ['ball'] + ['A' + str(i) for i in range(1, n_players + 1)] + ['B' + str(i) for i in
                                                                                   range(1, n_players + 1)]
        res = []
        if self.trinary:
            trinary_name = ['pre', 'act', 'eff']
            for i in range(3):
                res.append('BallTouched(%s)' % (trinary_name[i],))
        else:
            new_length = self.new_length(length)
            for i in range(0, new_length):
                res.append('BallTouched(%.2f)' % ((i + .5) / new_length,))
        return res

    def forward(self, states, beta=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, new_length, 1, 1]
        """
        batch, length, n_agents, state_dim = states.size()
        ball_pos = self.position(states[:, :, :1])
        ball_acc = self.acceleration(ball_pos)
        return self.large(ball_acc, beta=beta)

    def show(self, name='BallTouched', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s() = large(length(delta2(delta1(position(ball)))))' % name)
        self.position.show('position', indent + 2, **kwargs)
        self.acceleration[0].show('delta1', indent + 2, **kwargs)
        self.acceleration[1].show('delta2', indent + 2, **kwargs)
        self.acceleration[2].show('length', indent + 2, **kwargs)
        self.large.show('large', indent + 2, log=log, **kwargs)


class Close(nn.Module):
    def __init__(self, alpha=None, beta=None, trinary=False):
        super().__init__()
        self.position = Position()
        self.trinary = trinary
        self.distance = Distance()
        if trinary:
            self.min_pool_trinary = MinPoolTrinary()
        self.small = SoftSmall(alpha, beta)

    def new_length(self, length):
        return 3 if self.trinary else length

    def get_descriptions(self, n_agents, length):
        n_players = n_agents // 2
        agent_name = ['ball'] + ['A' + str(i) for i in range(1, n_players + 1)] + ['B' + str(i) for i in
                                                                                   range(1, n_players + 1)]
        res = []
        if self.trinary:
            trinary_name = ['pre', 'act', 'eff']
            for i in range(3):
                for p1 in range(1, n_agents - 1):
                    for p2 in range(p1 + 1, n_agents):
                        res.append('Close(%s, %s, %s)' % (agent_name[p1], agent_name[p2], trinary_name[i]))
        else:
            new_length = self.new_length(length)
            for i in range(0, new_length):
                for p1 in range(1, n_agents - 1):
                    for p2 in range(p1 + 1, n_agents):
                        res.append('Close(%s, %s, %.2f)' % (agent_name[p1], agent_name[p2], (i + .5) / new_length))
        return res

    def forward(self, states, alpha=None, beta=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        return [batch, length, n_agents, n_agents, 1]
        """
        batch, length, n_agents, state_dim = states.size()
        pos_1 = self.position(states.repeat(1, 1, 1, n_agents).view(batch, length, n_agents * n_agents, state_dim))
        pos_2 = self.position(states.repeat(1, 1, n_agents, 1))
        dist = self.distance(pos_1, pos_2)
        if self.trinary:
            dist = self.min_pool_trinary(dist)
        new_length = dist.size(1)
        small = self.small(dist.view(batch, new_length, n_agents, n_agents, 1), beta=beta)
        assert small.size() == torch.Size((batch, new_length, n_agents, n_agents, 1))
        indices = []
        for p1 in range(1, n_agents - 1):
            indices += [p1 * n_agents + p2 for p2 in range(p1 + 1, n_agents)]
        small = torch.index_select(small.view(batch, new_length, n_agents * n_agents), 2,
                                   torch.LongTensor(indices).to(states.device))
        return small

    def show(self, name='Close', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(p1, p2) = small(distance(p1, p2))' % name)
        self.distance.show('distance', indent + 2, **kwargs)
        self.small.show('small', indent + 2, log=log, **kwargs)


class Primitives(nn.Module):

    def __init__(self, length=17, n_agents=13, trinary=False, ternarize=False, ternarize_left=-0.01,
                 ternarize_right=0.01, **kwargs):
        super().__init__()
        self.n_agents = n_agents
        self.length = length
        self.trinary = trinary
        self.predicates = nn.ModuleList([
            BallTouched(trinary=trinary),
            Moving(beta_ball=math.log(0.3), beta_player=math.log(0.1), trinary=trinary),
            WithBall(beta=math.log(0.1), trinary=trinary),
            Close(beta=math.log(0.1), trinary=trinary),
        ])

        self.predicate_descriptions = []

        for predicate in self.predicates:
            self.predicate_descriptions += predicate.get_descriptions(n_agents, length)
        if ternarize:
            self.ternarize = Ternary(ternarize_left, ternarize_right)
        else:
            self.ternarize = None

        self.feature_length = len(self.predicate_descriptions)
        init_weights(self)

    def forward(self, states, hp=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        return results: [batch,length]
        """
        batch, length, n_agents, state_dim = states.size()
        assert self.length == length
        assert self.n_agents == n_agents
        if hp is None:
            hp = {}
        beta = None if 'beta' not in hp or hp['beta'] is None else torch.Tensor([hp['beta']]).to(states.device)

        pred_results = [predicate(states, beta=beta).view(batch, -1) for predicate in self.predicates]

        output = torch.cat(pred_results, dim=1)

        if self.ternarize is not None:
            output = self.ternarize(output)

        assert output.size(1) == self.feature_length

        return output

    def show(self, actions, weights, indent=0, log=print, threshold=0, **kwargs):
        self.predicates[0].show(indent=indent, log=log, **kwargs)
        self.predicates[1].show(indent=indent, log=log, **kwargs)
        self.predicates[2].show(indent=indent, log=log, **kwargs)
        self.predicates[3].show(indent=indent, log=log, **kwargs)
        # print(actions, weights.size(0))
        # exit()
        for i in range(weights.size(0)):
            action = actions[i]
            factors = [(float(weights[i, j]), self.predicate_descriptions[j]) for j in range(weights.size(1))]
            factors = list(reversed(sorted(factors, key=lambda x: x[0])))
            print(action)
            for x in factors:
                print(x[0], x[1])
            factors = [x for x in factors if abs(x[0]) > threshold]
            literals = ['not_' + x[1] if x[0] < 0 else x[1] for x in factors]
            conjunction = ' ^ '.join(literals)
            log("Action %s :  %s" % (action, conjunction))
