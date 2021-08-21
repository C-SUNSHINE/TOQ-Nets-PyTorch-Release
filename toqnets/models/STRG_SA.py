#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : STRG_SA.py
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
from toqnets.models.utils import get_temporal_indicator
from toqnets.nn.input_transform.primitives import AlignDifferential
from toqnets.nn.strg import STRG
from toqnets.nn.utils import init_weights
from toqnets.nn.utils import move_player_first


class STRG_SA(nn.Module):
    default_config = {
        'name': 'STRG_SA',
        'n_agents': 13,
        'state_dim': 3,
        'type_dim': 4,
        'n_actions': 9,
        'h_dim': 128,
        'n_features': 256,
        'input_feature': None,
        't_kernel_size': 7,
        'small_model': False,
        'tiny_model': False,
        'temporal_indicator': None,
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
        n_actions = self.config['n_actions']
        n_features = self.config['n_features']
        input_feature = self.config['input_feature']
        t_kernel_size = self.config['t_kernel_size']
        temporal_indicator = self.config['temporal_indicator']
        small_model = self.config['small_model']
        tiny_model = self.config['tiny_model']
        if small_model:
            n_features = self.config['n_features'] = 128
        if tiny_model:
            n_features = self.config['n_features'] = 64
        tim_dim = 0 if temporal_indicator is None else 1
        if input_feature == 'physical':
            self.differential = AlignDifferential()
            add_state_dim = 3
        elif input_feature is None:
            add_state_dim = 0
        else:
            raise ValueError()
        self.strg = STRG(state_dim + add_state_dim + type_dim + tim_dim, n_features, t_kernel_size=t_kernel_size,
                         small_model=small_model, tiny_model=tiny_model)

        self.decoder = nn.Linear(n_features, n_actions)
        init_weights(self)

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
        # print(types.size())
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
        x = self.strg(inputs)

        n_features = x.size(3)
        x = x.mean(dim=(1, 2))

        assert x.size() == torch.Size((batch, n_features))

        x = self.decoder(x.contiguous())
        return {'output': x,
                'target': actions,
                'loss': torch.zeros(1, device=trajectories.device)}

    def reset_parameters(self):
        init_weights(self.strg.strg_layers[0])

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
