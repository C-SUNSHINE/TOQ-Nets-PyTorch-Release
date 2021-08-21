#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : STGCN_LSTM_SA.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

from copy import deepcopy

import torch
from torch import nn

from toqnets.config_update import ConfigUpdate, update_config
from toqnets.models.utils import get_temporal_indicator
from toqnets.nn.input_transform.primitives import AlignDifferential
from toqnets.nn.stgcn import STGCN_Layer, Graph
from toqnets.nn.utils import apply_last_dim, move_player_first
from toqnets.nn.utils import init_weights


class STGCN_LSTM_SA(nn.Module):
    default_config = {
        'name': 'STGCN_LSTM_SA',
        'n_agents': 13,
        'state_dim': 3,
        'image_dim': None,
        'type_dim': 4,
        'n_actions': 9,
        'h_dim': 128,
        'kernel_size': (3, 7),  # (temporal_kernel_size, spatial_kernel_size)
        'edge_importance_weighting': True,
        'dropout': 0.5,
        'input_feature': None,
        'temporal_indicator': None,
        'small_model': False,
        'tiny_model': False,
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
        type_dim = self.config['type_dim']
        h_dim = self.config['h_dim']
        n_actions = self.config['n_actions']
        kernel_size = self.config['kernel_size']
        edge_importance_weighting = self.config['edge_importance_weighting']
        input_feature = self.config['input_feature']
        dropout = self.config['dropout']
        temporal_indicator = self.config['temporal_indicator']
        small_model = self.config['small_model']
        tiny_model = self.config['tiny_model']
        if small_model:
            h_dim = self.config['h_dim'] = 84
        if tiny_model:
            h_dim = self.config['h_dim'] = 32

        tim_dim = 0 if temporal_indicator is None else 1
        if input_feature == 'physical':
            self.differential = AlignDifferential()
            add_state_dim = 3
        elif input_feature is None:
            add_state_dim = 0
        else:
            raise ValueError()

        self.graph = Graph(layout='complete', n_nodes=n_agents)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.A = A.repeat(kernel_size[1], 1, 1)

        self.stgcn_layers = nn.ModuleList([
            STGCN_Layer(state_dim + add_state_dim + type_dim + tim_dim, 64, kernel_size, 1, residual=False),
            STGCN_Layer(64, 64, kernel_size, 1, dropout=dropout),
            STGCN_Layer(64, 64, kernel_size, 1, dropout=dropout),
            STGCN_Layer(64, 128, kernel_size, 1, dropout=dropout),
            STGCN_Layer(128, 128, kernel_size, 1, dropout=dropout),
            STGCN_Layer(128, 256, kernel_size, 1, dropout=dropout),
            STGCN_Layer(256, 256, kernel_size, 1, dropout=dropout),
            STGCN_Layer(256, h_dim, kernel_size, 1),
        ]) if not small_model and not tiny_model else (nn.ModuleList([
            STGCN_Layer(state_dim + add_state_dim + type_dim + tim_dim, 16, kernel_size, 1, residual=False),
            STGCN_Layer(16, 16, kernel_size, 1, dropout=dropout),
            STGCN_Layer(16, 32, kernel_size, 2, dropout=dropout),
            STGCN_Layer(32, 32, kernel_size, 1, dropout=dropout),
            STGCN_Layer(32, 64, kernel_size, 2, dropout=dropout),
            STGCN_Layer(64, h_dim, kernel_size, 1)
        ]) if not tiny_model else nn.ModuleList([
            STGCN_Layer(state_dim + add_state_dim + type_dim + tim_dim, 12, kernel_size, 1, residual=False),
            STGCN_Layer(12, 12, kernel_size, 2, dropout=dropout),
            STGCN_Layer(12, 12, kernel_size, 2, dropout=dropout),
            STGCN_Layer(12, h_dim, kernel_size, 1)
        ]))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(kernel_size[1], n_agents, n_agents), requires_grad=True)
                for i in self.stgcn_layers
            ])
        else:
            self.edge_importance = [1] * len(self.stgcn_layers)

        self.lstm = nn.LSTM(input_size=h_dim, hidden_size=h_dim, num_layers=2, batch_first=True)

        self.decoder = nn.Linear(h_dim, n_actions)

    def forward(self, data=None, hp=None):
        """
        :param data:
            'trajectories': [batch, length, n_agents, state_dim]
            'playerid': [batch]
            * 'images': [batch, length, n_agents, W, H, C]
            'actions': [batch]
            'types': [batch, n_agents, type_dim]
        :param hp: hyper parameters
        """
        types = data['types'].type(torch.float)
        trajectories = data['trajectories']
        playerid = data['playerid'] if 'playerid' in data else None
        actions = data['actions']
        middle = data['middle'] if 'middle' in data else None
        device = trajectories.device
        batch, length, n_agents, state_dim = trajectories.size()
        assert n_agents == self.config['n_agents']
        assert state_dim == self.config['state_dim']
        assert actions.size() == torch.Size((batch,))
        temporal_indicator = self.config['temporal_indicator']

        if playerid is not None and playerid.max() > -0.5:
            states = move_player_first(trajectories, playerid)
            types = torch.zeros(states.size(0), states.size(2), 4, device=states.device)
            n_players = n_agents // 2
            for k in range(n_agents):
                tp = (0 if k == 0 else 1) if k <= 1 else (2 if k <= n_players else 3)
                types[:, k, tp] = 1
        else:
            states = trajectories

        if self.config['input_feature'] == 'physical':
            velocity = self.differential(states)
            acc = self.differential(velocity)
            states = torch.cat([
                states,
                torch.norm(velocity[:, :, :, :2], p=2, dim=3, keepdim=True),
                torch.norm(velocity[:, :, :, 2:], p=1, dim=3, keepdim=True),
                torch.norm(acc, p=2, dim=3, keepdim=True)
            ], dim=3)
        elif self.config['input_feature'] is None:
            pass
        else:
            raise ValueError()

        inputs = torch.cat([states, types.unsqueeze(1).repeat(1, length, 1, 1)], dim=3)
        if temporal_indicator is not None:
            if middle is None:
                tim = get_temporal_indicator(temporal_indicator, length, device).view(1, length, 1, 1).repeat(
                    batch, 1, n_agents, 1)
            else:
                tim = get_temporal_indicator(temporal_indicator, length, batch_middle=middle, device=device).view(
                    batch, length, 1, 1).repeat(1, 1, n_agents, 1)
            inputs = torch.cat([inputs, tim], dim=3)

        x = inputs.permute(0, 3, 1, 2).contiguous()
        A = self.A.to(device)
        for gcn, importance in zip(self.stgcn_layers, self.edge_importance):
            x, _ = gcn(x, A * importance)
        x = x.permute(0, 2, 3, 1).mean(2)

        _, (h_n, c_n) = self.lstm(x)

        features = h_n[-1]

        assert features.size(1) == self.config['h_dim']

        outputs = apply_last_dim(self.decoder, features.contiguous())
        return {'output': outputs,
                'target': actions,
                'loss': torch.zeros(1, device=trajectories.device)}

    def reset_parameters(self):
        init_weights(self.stgcn_layers[0])

    def set_grad(self, option):
        if option == 'all':
            for param in self.parameters():
                param.requires_grad_(True)
        elif option == 'none':
            for param in self.parameters():
                param.requires_grad_(False)
        elif option == 'gfootball_finetune':
            for param in self.parameters():
                param.requires_grad_(False)
            for param in self.stgcn.st_gcn_layers[0].parameters():
                param.requires_grad_(True)
        else:
            raise ValueError()
