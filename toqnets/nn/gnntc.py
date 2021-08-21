#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gnntc.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import torch
from torch import nn

from toqnets.nn.propnet import AgentEncoder, RelationEncoder, Propagator


class GNNTC(nn.Module):
    """
    Graph Neural Network + Temporal Conv
    """

    def __init__(self, n_agents, state_dim=3, type_dim=3, h_dim=256, n_features=256,
                 layers=[(32, 9, 3), (32, 7, 2), (32, 5, 2)], dropout=0.5):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.type_dim = type_dim
        self.h_dim = h_dim
        self.n_features = n_features

        self.agent_encoder = AgentEncoder(type_dim, h_dim, h_dim)
        self.state_encoder = AgentEncoder(state_dim, h_dim, h_dim)
        self.relation_encoder = RelationEncoder(h_dim + h_dim, h_dim, h_dim)
        self.relation_propagator = Propagator(h_dim + h_dim + h_dim, h_dim)

        last_channel = h_dim

        conv_layers = []
        for (channel, kernel_size, stride) in layers:
            conv_layers.append(
                nn.Conv1d(last_channel, channel, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            last_channel = channel
        conv_layers.append(nn.Conv1d(last_channel, n_features, kernel_size=1, padding=0, stride=1))
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, states, types, playerid):
        """
        :param states: [batch, length, n_agents, state_dim]
        :param types: [batch, n_agents, type_dim]
        :param playerid: [batch]
        """
        batch, length, n_agents, state_dim = states.size()
        type_dim = types.size(2)
        assert n_agents == self.n_agents
        assert state_dim == self.state_dim
        assert type_dim == self.type_dim
        assert types.size() == torch.Size((batch, n_agents, type_dim))
        assert playerid.size() == torch.Size((batch,))
        h_dim = self.h_dim
        n_features = self.n_features

        agent_encode = self.agent_encoder(types)
        agent_encode_r = agent_encode.repeat(1, n_agents, 1)
        agent_encode_s = agent_encode.repeat(1, 1, n_agents).view(batch, n_agents * n_agents, h_dim)
        relation_encode = self.relation_encoder(torch.cat([agent_encode_r, agent_encode_s], dim=2))

        state_encode = self.state_encoder(states)
        state_encode_r = state_encode.repeat(1, 1, n_agents, 1)
        state_encode_s = state_encode.repeat(1, 1, 1, n_agents).view(batch, length, n_agents * n_agents, h_dim)
        relation_effect = self.relation_propagator(
            torch.cat([state_encode_s, state_encode_r, relation_encode.unsqueeze(1).repeat(1, length, 1, 1)], dim=3))

        agg_effect = relation_effect.view(batch, length, n_agents, n_agents, h_dim).sum(dim=3)
        # agg_effect:[batch, length, n_agents, h_dim]
        assert agg_effect.size() == torch.Size([batch, length, n_agents, h_dim])

        agg_effect = agg_effect.gather(2, playerid.view(-1, 1, 1, 1).repeat(1, length, 1, h_dim))[:, :, 0, :]
        # agg_effect:[batch, length, h_dim]
        assert agg_effect.size() == torch.Size([batch, length, h_dim])

        # NB(Jiayuan Mao @ 04/14): add contiguous() to avoid back-propagation error in PyTorch 1.4.
        output = self.conv(agg_effect.transpose(1, 2).contiguous())
        # print(output.size(), torch.Size([batch * n_agents, n_features, length]))
        assert output.size(0) == batch
        assert output.size(1) == n_features

        return output[:, :, output.size(2) // 2]
