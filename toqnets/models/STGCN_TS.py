#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : STGCN_TS.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

from copy import deepcopy

import torch
from torch import nn

from toqnets.config_update import update_config, ConfigUpdate
from toqnets.nn.stgcn import STGCN
from toqnets.nn.utils import average_sample_before_softmax


class STGCN_TS(nn.Module):
    default_config = {
        'name': 'STGCN_TS',
        'n_agents': 13,
        'length': 30,
        'n_frames': 8,
        'state_dim': 3,
        'n_actions': 19,
        'h_dim': 128,
        'n_features': 500,
        'kernel_size': (9, 7),  # (temporal_kernel_size, spatial_kernel_size)
        'edge_importance_weighting': False,
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
        n_actions = self.config['n_actions']
        n_features = self.config['n_features']
        kernel_size = self.config['kernel_size']
        edge_importance_weighting = self.config['edge_importance_weighting']

        self.stgcn = STGCN(n_agents, state_dim, n_features, kernel_size, edge_importance_weighting,
                           graph_option='toyota')

        self.decoder = nn.Linear(n_features + (n_features if self.config['with_video'] else 0), n_actions)

    def forward(self, data=None, hp=None):
        """
        :param data:
            'trajectories': [batch, length, n_agents, state_dim]
            'actions': [batch]
            *'video': [batch, length, C, W, H]
        :param hp: hyper parameters
        """
        trajectories = data['trajectories']
        actions = data['actions']
        device = trajectories.device
        batch, length, n_agents, state_dim = trajectories.size()
        assert n_agents == self.config['n_agents']
        assert state_dim == self.config['state_dim']
        assert actions.size() == torch.Size((batch,))
        n_features = self.config['n_features']
        n_actions = self.config['n_actions']
        x = self.stgcn(trajectories)
        x = torch.mean(x, dim=(2, 3))

        assert x.size() == torch.Size((batch, n_features))

        x = self.decoder(x)

        return {'output': x,
                'target': actions,
                'loss': torch.zeros(1, device=device)}
