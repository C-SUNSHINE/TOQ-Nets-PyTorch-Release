#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : STGCN_LSTM_RL.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

from copy import deepcopy

import torch
from torch import nn

from toqnets.nn.stgcn import STGCN
from toqnets.nn.utils import apply_last_dim
from toqnets.config_update import ConfigUpdate
from toqnets.config_update import update_config

class STGCN_LSTM_RL(nn.Module):
    default_config = {
        'name': 'STGCN_LSTM_RL',
        'n_agents': 45,
        'state_dim': [14, 14],
        'object_name_dim': 194,
        'n_actions': 2,
        'h_dim': 128,
        'n_features': 256,
        'kernel_size': (3, 7),  # (temporal_kernel_size, spatial_kernel_size)
        'edge_importance_weighting': True,
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        assert isinstance(config_update, ConfigUpdate)
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        for k in cls.default_config:
            if k not in config:
                config[k] = deepcopy(cls.default_config[k])
        return config

    def __init__(self, config):
        super().__init__()
        self.config = config

        n_agents = self.config['n_agents']
        state_dim = self.config['state_dim']
        object_name_dim = self.config['object_name_dim']
        n_actions = self.config['n_actions']
        n_features = self.config['n_features']
        kernel_size = self.config['kernel_size']
        edge_importance_weighting = self.config['edge_importance_weighting']

        self.stgcn = STGCN(n_agents + 1, state_dim[0] + state_dim[1] + object_name_dim, n_features, kernel_size,
                           edge_importance_weighting)

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_features, num_layers=2, batch_first=True)

        self.decoder = nn.Linear(n_features, n_actions)

    def forward(self, data=None, hp=None):
        """
        :param data:
            'nullary_states': [batch, length, state_dim[0]]
            'unary_states': [batch, length, n_agents, state_dim[1]]
            'actions': [batch]
        :param hp: hyper parameters
        """
        nullary_states = data['nullary_states']
        unary_states = data['unary_states']
        actions = data['actions']
        lengths = data['lengths']
        batch, length, n_agents, _ = unary_states.size()
        state_dim = self.config['state_dim']
        object_name_dim = self.config['object_name_dim']
        assert n_agents == self.config['n_agents']
        assert nullary_states.size() == torch.Size((batch, length, state_dim[0]))
        assert unary_states.size(-1) == state_dim[1] + object_name_dim
        n_features = self.config['n_features']

        nullary_expand = torch.cat([
            nullary_states.view(batch, length, 1, state_dim[0]),
            torch.zeros(batch, length, 1, state_dim[1] + object_name_dim).to(nullary_states.device),
        ], dim=3)
        unary_expand = torch.cat([
            torch.zeros(batch, length, n_agents, state_dim[0]).to(unary_states.device),
            unary_states,
        ], dim=3)
        inputs = torch.cat([
            nullary_expand,
            unary_expand,
        ], dim=2)
        x = self.stgcn(inputs)

        x = x.permute(0, 2, 3, 1).mean(2)

        _, (h_n, c_n) = self.lstm(x)

        features = h_n[-1]

        assert features.size() == torch.Size((batch, n_features))

        outputs = apply_last_dim(self.decoder, features.contiguous())

        return {'output': outputs,
                'target': actions,
                'loss': torch.zeros(1, device=nullary_states.device)}
