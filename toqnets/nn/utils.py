#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/19/2019
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total

def apply_last_dim(model, x):
    size = list(x.size())
    y = model(x.contiguous().view(-1, size[-1]))
    size[-1] = y.size(-1)
    y = y.view(torch.Size(size))
    return y


def average_sample_before_softmax(x):
    batch, n_samples, n_actions = x.size()
    x = nn.Softmax(dim=1)(x.view(batch * n_samples, n_actions))
    x = x.view(batch, n_samples, n_actions).mean(dim=1)
    x = torch.log(x)
    return x


def move_player_first(trajectories, playerid):
    batch, length, n_agents, state_dim = trajectories.size()
    n_players = n_agents // 2
    is_left = torch.le(playerid, n_players).type(torch.long)
    states = torch.zeros(trajectories.size()).to(trajectories.device)
    states[:, :, 0] = trajectories[:, :, 0]
    states[:, :, 1:n_players + 1] = \
        trajectories[:, :, 1:n_players + 1] * is_left.view(batch, 1, 1, 1).float() + \
        trajectories[:, :, n_players + 1:n_players * 2 + 1] * (1. - is_left.view(batch, 1, 1, 1).float())
    states[:, :, n_players + 1:n_players * 2 + 1] = \
        trajectories[:, :, 1:n_players + 1] * (1. - is_left.view(batch, 1, 1, 1).float()) + \
        trajectories[:, :, n_players + 1:n_players * 2 + 1] * is_left.view(batch, 1, 1, 1).float()
    new_playerid = playerid - n_players + n_players * is_left
    new_id = torch.arange(n_agents).to(trajectories.device).repeat(batch).view(batch, -1)
    minus1 = torch.le(
        new_id,
        new_playerid.view(batch, 1)
    ).type(torch.long)
    new_id[:, 1:] -= minus1[:, 1:]
    new_id[:, 1] = new_playerid
    states = states.gather(2, new_id.view(batch, 1, n_agents, 1).repeat(1, length, 1, state_dim))

    # print(playerid, new_id)
    # exit()

    # states_ = torch.zeros(trajectories.size()).to(trajectories.device)
    # for i in range(batch):
    #     switch_i = playerid[i] > n_players
    #     for j in range(length):
    #         index_map = [0 for i in range(n_agents)]
    #         for k in range(n_agents):
    #             new_k = 0
    #             if k > n_players:
    #                 if switch_i:
    #                     if k == playerid[i]:
    #                         new_k = 1
    #                     elif k < playerid[i]:
    #                         new_k = k - n_players + 1
    #                     else:
    #                         new_k = k - n_players
    #                 else:
    #                     new_k = k
    #             elif k > 0:
    #                 if switch_i:
    #                     new_k = k + n_players
    #                 else:
    #                     if k == playerid[i]:
    #                         new_k = 1
    #                     elif k < playerid[i]:
    #                         new_k = k + 1
    #                     else:
    #                         new_k = k
    #             index_map[new_k] = k
    #             states_[i, j, new_k] = trajectories[i, j, k]
    #         # print(new_id[i], index_map, playerid[i])
    #         # for k in range(n_agents):
    #         #     if ((states_[i, j, k] - states[i, j, k]) ** 2).sum() > 1e-5:
    #         #         print(i, j, k)
    #         #         exit()
    # assert ((states_ - states) ** 2).sum() < 1e-5
    return states


def init_weights(net, init_type='normal', init_param=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_param)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_param)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orth':
                init.orthogonal_(m.weight.data, gain=init_param)
            elif init_type == 'pdb':
                init.constant_(m.weight.data, 1.0)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_param)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class Binary(nn.Module):
    '''
    Binarize the input activations to 0, 1.
    '''

    def __init__(self, middle=0.5):
        super().__init__()
        self.middle = 0.5

    def forward(self, input):
        if input is None:
            return None
        input = input.clone()
        zero_index = input.lt(self.middle)
        one_index = ~zero_index
        input[zero_index] = 0
        input[one_index] = 1
        return input

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Ternary(nn.Module):
    '''
    Ternarize the input activations to -1, 0, 1.
    '''

    def __init__(self, left=-0.25, right=0.25):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, input):
        input = input.clone()
        left_index = input.lt(self.left)
        right_index = input.ge(self.right)
        input[left_index] = -1
        input[right_index] = 1
        input[~(left_index | right_index)] = 0
        return input

    def backward(self, grad_output):
        # input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input.ge(1)] = 0
        # grad_input[input.le(-1)] = 0
        return grad_input


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, ternarize_left=-0.25, ternarize_right=0.25, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.ternarize = Ternary(ternarize_left, ternarize_right)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.normal_(self.weight.data, 0, 1)

    def forward(self, input):
        return F.linear(input, self.ternarize(self.weight), self.bias)
