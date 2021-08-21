#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : NLTL_RLv3.py
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
from toqnets.nn.nlm import LogicMachine
from toqnets.nn.input_transform.input_transform import InputTransformNN
from toqnets.nn.nltl.modules import TemporalLogicMachineDPV2


class NLTL_RLv3(nn.Module):
    default_config = {
        'name': 'NLTL_RLv3',
        'length': 1000,
        'state_dim': [14, 14],
        'object_name_dim': 194,
        'n_actions': 45,  # 23,
        'input_transform': 'nn',
        'input_transform_h_dim': [],
        'depth': 3,
        'logic_output_dims': [48, 48, 48],
        'logic_hidden_dim': [96],
        'logic_layer_permute': True
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
        object_name_dim = self.config['object_name_dim']
        depth = self.config['depth']
        logic_output_dims = self.config['logic_output_dims']
        input_transform = self.config['input_transform']
        input_transform_h_dim = tuple(self.config['input_transform_h_dim'])
        logic_layer_permute = self.config['logic_layer_permute']
        assert n_actions >= 2

        if input_transform == 'nn':
            self.input_transform = InputTransformNN(state_dim=state_dim[1] + object_name_dim,
                                                    h_dims=input_transform_h_dim,
                                                    out_dim=64, time_reduction='none')

        input_dims = self.input_transform.out_dims
        self.logic_machine = LogicMachine(
            depth=depth, breadth=2, input_dims=input_dims, output_dims=logic_output_dims,
            logic_hidden_dim=[], exclude_self=True, residual=True, permute=logic_layer_permute,
        )
        output_dims = self.logic_machine.output_dims
        self.feature_dim = output_dims[0]

        self.logic_machine2 = TemporalLogicMachineDPV2(self.feature_dim, self.feature_dim, 3, pooling_range=(0, 1))

        self.decoder = nn.Linear(self.feature_dim, 1) if n_actions == 2 else nn.Linear(self.feature_dim, n_actions)
        self.l1loss = nn.L1Loss()

    def forward(self, data=None, hp=None):
        """
        :param data:
            'nullary_states': [batch, length, state_dim[0]]
            'unary_states': [batch, length, n_agents, state_dim[1] + object_name_dim]
            'actions': [batch]
        :param hp: hyper parameters
        """
        nullary_states = data['nullary_states']
        unary_states = data['unary_states']
        actions = data['actions']
        batch, length, n_agents, _ = unary_states.size()
        state_dim = self.config['state_dim']
        assert nullary_states.size() == torch.Size((batch, length, state_dim[0]))
        assert unary_states.size(-1) == state_dim[1]

        beta = hp['beta']
        is_prerun = False if 'prerun' not in hp else hp['prerun']

        binary_logic_layer = True if 'binary_logic_layer' in hp and hp['binary_logic_layer'] else False

        nlm_inputs = self.input_transform(unary_states, special_nullary_states=nullary_states, beta=beta)
        nlm_outputs = self.logic_machine(nlm_inputs, binary_layer=binary_logic_layer, keep_data=is_prerun)

        nltl_input = nlm_outputs[0]
        nltl_output = self.logic_machine2(nltl_input)
        features = nltl_output.max(dim=1)[0]

        output = self.decoder(features)
        if self.config['n_actions'] == 2:
            output = torch.cat([-output, output], dim=1)

        model_loss = self.l1loss(
            self.decoder.weight,
            torch.zeros(self.decoder.weight.size()).to(self.decoder.weight.device)
        ).repeat(batch).view(batch, -1).mean(1) * 0.0

        return {'output': output,
                'target': actions,
                'loss': model_loss}

    def reset_parameters(self):
        self.input_transform.reset_parameters('primitive_inequality')

    def primitives_require_grad(self, mode):
        self.input_transform.require_grad(mode)

    def show(self, log=print, save_dir=None, **kwargs):
        pass
