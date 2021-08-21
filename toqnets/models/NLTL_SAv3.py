#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : NLTL_SAv3.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

from copy import deepcopy

import torch
import torch.nn as nn

from toqnets.config_update import ConfigUpdate, update_config
from toqnets.models.utils import get_temporal_indicator
from toqnets.nn.input_transform.input_transform import InputTransformPredefined
from toqnets.nn.nlm import LogicMachine
from toqnets.nn.nltl.modules import TemporalLogicMachineDP
from toqnets.nn.utils import init_weights, move_player_first


class NLTL_SAv3(nn.Module):
    default_config = {
        'name': 'NLTL_SAv3',
        'state_dim': 3,
        'type_dim': 4,
        'image_dim': None,
        'n_actions': 9,
        'input_transformer': 'predefined',
        'depth': 3,
        't_depth': 3,
        'default_temporal_integrator': 'max',
        'logic_output_dims': [16, 16, 16],
        'logic_hidden_dim': [],
        'cmp_dim': 5,
        'logic_layer_permute': True,
        'both_quantify': False,
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

        type_dim = self.config['type_dim']
        n_actions = self.config['n_actions']
        depth = self.config['depth']
        t_depth = self.config['t_depth']
        logic_output_dims = self.config['logic_output_dims']
        cmp_dim = self.config['cmp_dim']
        input_transformer = self.config['input_transformer']
        logic_layer_permute = self.config['logic_layer_permute']
        both_quantify = self.config['both_quantify']
        temporal_indicator = self.config['temporal_indicator']
        assert n_actions >= 2

        if input_transformer == 'predefined':
            self.input_transform = InputTransformPredefined(
                type_dim=type_dim, cmp_dim=cmp_dim,
                time_reduction='none'
            )

        input_dims = self.input_transform.out_dims
        input_dims[0] += 1 if temporal_indicator is not None else 0
        if depth > 0:
            self.logic_machine = LogicMachine(
                depth=depth, breadth=2, input_dims=input_dims, output_dims=logic_output_dims,
                logic_hidden_dim=self.config['logic_hidden_dim'], exclude_self=True, residual=True,
                permute=logic_layer_permute,
                forall=both_quantify
            )
            output_dims = self.logic_machine.output_dims
            self.feature_dim = output_dims[0]
        else:
            self.feature_dim = 128
            self.logic_machine0 = nn.Sequential(
                nn.Linear(input_dims[1], self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU())
        if t_depth >= 0:
            self.temporal_machine = TemporalLogicMachineDP(
                self.feature_dim, self.feature_dim, t_depth,
                until=both_quantify)

        self.fc = nn.Linear(self.feature_dim, n_actions)
        self.l1loss = nn.L1Loss()

    def add_temporal_indicator(self, indicator, inputs, batch, length, middle=None, device='cpu'):
        res = get_temporal_indicator(indicator, length, batch_middle=middle, device=device)
        if middle is None:
            res = res.view(1, length, 1).repeat(batch, 1, 1)
        else:
            res = res.view(batch, length, 1)
        if inputs[0] is None:
            inputs[0] = res
        else:
            inputs[0] = torch.cat((inputs[0], res), dim=-1)
        return inputs

    def forward(self, data=None, hp=None):
        """
        :param data:
            'trajectories': [batch, length, n_agents, state_dim]
            'playerid': [batch]
            'actions': [batch]
            'types': [batch, n_agents, type_dim]
        :param hp: hyper parameters
        """
        types = data['types']
        trajectories = data['trajectories']
        playerid = data['playerid'] if 'playerid' in data else None
        actions = data['actions']
        middle = data['middle'] if 'middle' in data else None
        batch, length, n_agents, state_dim = trajectories.size()
        assert state_dim == self.config['state_dim']
        assert actions.size() == torch.Size((batch,))
        assert types.size(1) == n_agents
        temporal_indicator = self.config['temporal_indicator']

        types = types.float()

        if playerid is not None and playerid.max() > -0.5:
            states = move_player_first(trajectories, playerid)
            types = torch.zeros(states.size(0), states.size(2), 4, device=states.device)
            n_players = n_agents // 2
            for k in range(n_agents):
                tp = (0 if k == 0 else 1) if k <= 1 else (2 if k <= n_players else 3)
                types[:, k, tp] = 1
        else:
            states = trajectories

        beta = hp['beta']
        inputs_to_transform = states
        if 'estimate_parameters' in hp and hp['estimate_parameters']:
            self.input_transform.estimate_parameters(inputs_to_transform)

        nlm_inputs = self.input_transform(inputs_to_transform, add_unary_tensor=types, beta=beta)
        if self.config['depth'] > 0:
            if temporal_indicator is not None:
                nlm_inputs = self.add_temporal_indicator(temporal_indicator, nlm_inputs, batch, nlm_inputs[1].size(1),
                                                         middle=middle,
                                                         device=states.device)
            nlm_outputs = self.logic_machine(
                nlm_inputs,
            )
            nltl_input = nlm_outputs[0]
        else:
            nltl_input = self.logic_machine0(nlm_inputs[1][:, :, 1])
        if self.config['t_depth'] >= 0:
            nltl_output = self.temporal_machine(nltl_input)
            features = nltl_output.max(dim=1)[0]
        else:
            default_temporal_integrator = self.config['default_temporal_integrator']
            if default_temporal_integrator == 'middle':
                features = nltl_input[:, nltl_input.size(1) // 2]
            elif default_temporal_integrator == 'avg':
                features = nltl_input.mean(1)
            elif default_temporal_integrator == 'max':
                features = nltl_input.max(1)[0]
            else:
                raise ValueError()

        assert len(features.size()) == 2 and features.size(0) == trajectories.size(0) and features.size(
            1) == self.feature_dim

        output = self.fc(features)

        model_loss = self.l1loss(
            self.fc.weight,
            torch.zeros(self.fc.weight.size()).to(self.fc.weight.device)
        ).repeat(batch).view(batch, -1).mean(1) * 0.1

        return {
            'output': output,
            'target': actions,
            'loss': model_loss
        }

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
            self.primitives_require_grad(True)
        else:
            raise ValueError()

    def reset_parameters(self):
        self.input_transform.reset_parameters('primitive_inequality')
        if self.config['depth'] > 0:
            init_weights(self.logic_machine.layers[0])

    def primitives_require_grad(self, mode):
        self.input_transform.require_grad(mode)

    def get_path(self, k):
        pass
