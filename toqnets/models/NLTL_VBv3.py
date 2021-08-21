#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : NLTL_VBv3.py
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
from toqnets.nn.input_transform.input_transform import InputTransformPredefined, InputTransformNN, InputTransformTCN
from toqnets.nn.nlm import LogicMachine
from toqnets.nn.nltl.modules import TemporalLogicMachineDP
from toqnets.nn.stgcn import Graph, STGCN_Layer
from toqnets.nn.utils import move_player_first, init_weights


class NLTL_VBv3(nn.Module):
    default_config = {
        'name': 'NLTL_VBv3',
        'state_dim': 14,
        'n_actions': 8,
        'input_transformer': 'predefined',
        'depth': 3,
        't_depth': 3,
        'logic_output_dims': [48, 48, 48],
        'logic_hidden_dim': [],
        'cmp_dim': 10,
        'logic_layer_permute': True,
        'both_quantify': False,
        'temporal_indicator': None
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

        state_dim = self.config['state_dim']
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
                cmp_dim=cmp_dim,
                time_reduction='none'
            )
        elif input_transformer == 'nn':
            # input('use nn')
            self.input_transform = InputTransformNN(
                state_dim=state_dim, out_dim=64, h_dims=[64], time_reduction='none'
            )
        elif input_transformer == 'stgcn':
            h_dim = 64
            dropout = 0.5
            kernel_size = (7, 7)
            self.stgcn_layers = nn.ModuleList([
                STGCN_Layer(state_dim, h_dim, kernel_size, 1, residual=False),
                STGCN_Layer(h_dim, h_dim, kernel_size, 1, dropout=dropout),
                STGCN_Layer(h_dim, h_dim, kernel_size, 1),
            ])
            self.input_transform = InputTransformNN(state_dim=h_dim, h_dims=[], out_dim=h_dim,
                                                    time_reduction='none')
        elif input_transformer == 'tcn':
            self.input_transform = InputTransformTCN(state_dim=state_dim,
                                                     h_dims=[96],
                                                     out_dim=48,
                                                     time_reduction='none')
        elif input_transformer == 'tconv':
            self.input_transform = InputTransformTCN(state_dim=state_dim,
                                                     out_dim=96,
                                                     time_reduction='none',
                                                     single_tconv=True)
        else:
            raise ValueError()

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
            self.logic_machine0 = nn.Sequential(nn.Linear(input_dims[1], self.feature_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.feature_dim, self.feature_dim),
                                                nn.ReLU())
        if t_depth >= 0:
            self.logic_machine2 = TemporalLogicMachineDP(self.feature_dim, self.feature_dim, t_depth,
                                                         until=both_quantify)

        self.decoder = nn.Linear(self.feature_dim, 1) if n_actions == 2 else nn.Linear(self.feature_dim, n_actions)
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
        states = data['states']
        actions = data['actions']
        middle = data['middle'] if 'middle' in data else None
        batch, length, n_agents, state_dim = states.size()
        assert state_dim == self.config['state_dim']
        assert actions.size() == torch.Size((batch,))
        temporal_indicator = self.config['temporal_indicator']

        beta = hp['beta']
        is_eval = False if 'eval' not in hp else hp['eval']
        is_prerun = False if 'prerun' not in hp else hp['prerun']
        if 'estimate_parameters' in hp and hp['estimate_parameters']:
            self.input_transform.estimate_parameters(states)
            return None

        binary_logic_layer = True if 'binary_logic_layer' in hp and hp['binary_logic_layer'] else False

        inputs = states

        if self.config['input_transformer'] == 'stgcn':
            x = states.permute(0, 3, 1, 2).contiguous()
            A = torch.tensor(Graph(layout='complete', n_nodes=n_agents).A, dtype=torch.float32, requires_grad=False).to(
                states.device)
            # x[batch, in_channels, length, n_agents]
            for gcn in self.stgcn_layers:
                x, _ = gcn(x, A)
            inputs_to_transform = x.permute(0, 2, 3, 1)
            assert inputs_to_transform.size()[:3] == torch.Size((batch, length, n_agents))
        else:
            inputs_to_transform = states

        nlm_inputs = self.input_transform(inputs_to_transform, beta=beta)
        if self.config['depth'] > 0:
            if temporal_indicator is not None:
                nlm_inputs = self.add_temporal_indicator(temporal_indicator, nlm_inputs, batch, nlm_inputs[1].size(1),
                                                         middle=middle,
                                                         device=states.device)
            nlm_outputs = self.logic_machine(nlm_inputs, binary_layer=binary_logic_layer, keep_data=is_prerun)
            nltl_input = nlm_outputs[0]
        else:
            nltl_input = self.logic_machine0(nlm_inputs[1][:, :, 1])
        if self.config['t_depth'] >= 0:
            nltl_output = self.logic_machine2(nltl_input)
            features = nltl_output.max(dim=1)[0]
        else:
            # TODO make this according to temporal indicator
            features = nltl_input[:, nltl_input.size(1) // 2]

        assert len(features.size()) == 2 and features.size(0) == states.size(0) and features.size(
            1) == self.feature_dim

        output = self.decoder(features)
        if self.config['n_actions'] == 2:
            output = torch.cat([-output, output], dim=1)

        model_loss = self.l1loss(
            self.decoder.weight,
            torch.zeros(self.decoder.weight.size()).to(self.decoder.weight.device)
        ).repeat(batch).view(batch, -1).mean(1) * 1.0

        return {'output': output,
                'target': actions,
                'loss': model_loss}

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

    def show(self, action_list=None, log=print, save_dir=None, save_name=None, **kwargs):
        pass