#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : STGCN_VB.py
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
from toqnets.nn.input_transform.input_transform import InputTransformPredefined
from toqnets.nn.stgcn import STGCN
from toqnets.nn.utils import apply_last_dim, move_player_first, init_weights


class STGCN_VB(nn.Module):
    default_config = {
        'name': 'STGCN_VB',
        'n_agents': 13,
        'state_dim': 14,
        'n_actions': 8,
        'h_dim': 128,
        'n_features': 256,
        'kernel_size': (7, 7),  # (temporal_kernel_size, spatial_kernel_size)
        'edge_importance_weighting': False,
        'input_feature': None,
        'small_model': True,
        'tiny_model': False,
        'max_gcn': False,
        'temporal_indicator': None,
        'cmp_dim': 5
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
        input_feature = self.config['input_feature']
        temporal_indicator = self.config['temporal_indicator']
        small_model = self.config['small_model']
        tiny_model = self.config['tiny_model']
        cmp_dim = self.config['cmp_dim']
        if small_model:
            n_features = self.config['n_features'] = 128
        if tiny_model:
            n_features = self.config['n_features'] = 64
        tim_dim = 0 if temporal_indicator is None else 1
        if input_feature == 'physical':
            self.input_transform = InputTransformPredefined(
                type_dim=0, cmp_dim=cmp_dim,
                time_reduction='none'
            )
            unary_input_dim = self.input_transform.out_dims[1]
            binary_input_dim = self.input_transform.out_dims[2]
            binary_output_dim = binary_input_dim
        elif input_feature is None:
            unary_input_dim = 0
            binary_input_dim = 0
            binary_output_dim = 0
        else:
            raise ValueError()
        self.state_encoder = lambda x: x  # AgentEncoder(state_dim + type_dim, h_dim, h_dim)
        self.stgcn = STGCN(n_agents, state_dim + unary_input_dim + tim_dim, n_features, kernel_size,
                           edge_importance_weighting, small_model=small_model,
                           tiny_model=tiny_model, binary_input_dim=binary_input_dim,
                           binary_output_dim=binary_output_dim,
                           max_gcn=self.config['max_gcn'])

        self.decoder = nn.Linear(n_features, n_actions)

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
        states = data['states']
        actions = data['actions']
        middle = data['middle'] if 'middle' in data else None
        device = states.device
        batch, length, n_agents, state_dim = states.size()
        beta = hp['beta']
        assert n_agents == self.config['n_agents']
        assert state_dim == self.config['state_dim']
        assert actions.size() == torch.Size((batch,))
        temporal_indicator = self.config['temporal_indicator']

        binary_input = None
        if self.config['input_feature'] == 'physical':
            nlm_inputs = self.input_transform(states, beta=beta)
            binary_input = nlm_inputs[2]
            inputs = torch.cat([states, nlm_inputs[1]], dim=3)
        elif self.config['input_feature'] is None:
            inputs = torch.cat([states], dim=3)
        else:
            raise ValueError()
        if temporal_indicator is not None:
            if middle is None:
                tim = get_temporal_indicator(temporal_indicator, length, device).view(1, length, 1, 1).repeat(
                    batch, 1, n_agents, 1)
            else:
                tim = get_temporal_indicator(temporal_indicator, length, batch_middle=middle, device=device).view(
                    batch, length, 1, 1).repeat(1, 1, n_agents, 1)
            inputs = torch.cat([inputs, tim], dim=3)

        x = self.stgcn(inputs, binary_input=binary_input)

        n_features = x.size(1)
        x = x.mean(dim=(2, 3))

        assert x.size() == torch.Size((batch, n_features))

        x = apply_last_dim(self.decoder, x.contiguous())
        return {'output': x,
                'target': actions,
                'loss': torch.zeros(1, device=states.device)}

    def reset_parameters(self):
        init_weights(self.stgcn.st_gcn_layers[0])

    def set_grad(self, option):
        pass
