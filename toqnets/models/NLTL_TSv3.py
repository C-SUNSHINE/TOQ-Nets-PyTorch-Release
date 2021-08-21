#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : NLTL_TSv3.py
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
from toqnets.nn.input_transform.input_transform import InputTransformToyotaSkeleton, InputTransformToyotaJoint, \
    InputTransformNN
from toqnets.nn.nlm import LogicMachine
from toqnets.nn.nltl.modules import TemporalLogicMachineDPV2
from toqnets.nn.stgcn import STGCN_Layer, Graph
from toqnets.nn.utils import average_sample_before_softmax


class NLTL_TSv3(nn.Module):
    default_config = {
        'name': 'NLTL_TSv3',
        'length': 30,
        'n_frames': 8,
        'state_dim': 3,
        'type_dim': 13,
        'h_dim': 128,
        'image_dim': None,
        'n_actions': 19,
        'input_transformer': 'joint',
        'depth': 3,
        'logic_output_dims': 32,
        'logic_hidden_dim': [64],
        'image_feature_dim': 500,
        'cmp_dim': 5,
        'logic_layer_permute': True,
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
        type_dim = self.config['type_dim']
        n_actions = self.config['n_actions']
        depth = self.config['depth']
        logic_output_dims = self.config['logic_output_dims']
        cmp_dim = self.config['cmp_dim']
        input_transformer = self.config['input_transformer']
        logic_layer_permute = self.config['logic_layer_permute']
        logic_hidden_dim = self.config['logic_hidden_dim']
        image_feature_dim = self.config['image_feature_dim']
        assert n_actions >= 2

        self.types = torch.eye(type_dim)

        if input_transformer == 'joint':
            self.input_transform = InputTransformToyotaJoint(type_dim=type_dim, cmp_dim=cmp_dim, time_reduction='none')
            breadth = 3
        elif input_transformer == 'skeleton':
            self.input_transform = InputTransformToyotaSkeleton(type_dim=type_dim, cmp_dim=cmp_dim,
                                                                time_reduction='none')
            breadth = 1
        elif input_transformer in ['stgcn', 'stgcn+joint']:
            h_dim = self.config['h_dim']
            n_joints = 13
            kernel_size = (7, 7)
            self.skeleton_graph = Graph(layout='toyota', n_nodes=n_joints)
            A = torch.tensor(self.skeleton_graph.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A', A)
            self.A = self.A.repeat(kernel_size[1], 1, 1)
            self.stgcn_layers = nn.ModuleList([
                STGCN_Layer(state_dim, h_dim, kernel_size, 1, residual=False),
            ])
            self.input_transform = InputTransformNN(state_dim=h_dim, h_dims=[h_dim], out_dim=h_dim,
                                                    type_dim=type_dim, time_reduction='none')
            breadth = 2
            if input_transformer == 'stgcn+joint':
                self.input_transform_joint = InputTransformToyotaJoint(cmp_dim=cmp_dim, time_reduction='none')
                breadth = 3
        elif input_transformer == 'nn':
            h_dim = 128
            self.input_transform = InputTransformNN(breadth=3, state_dim=state_dim, h_dims=[h_dim + h_dim],
                                                    out_dim=h_dim,
                                                    type_dim=type_dim, time_reduction='none')
            breadth = 3
        else:
            raise NotImplementedError()
        if input_transformer == 'stgcn+joint':
            stgcn_dims = self.input_transform.out_dims + [0]
            joint_dims = self.input_transform_joint.out_dims
            input_dims = [x[0] + x[1] for x in zip(stgcn_dims, joint_dims)]
        else:
            input_dims = self.input_transform.out_dims
        self.nlm_input_dims = tuple(input_dims)

        self.logic_machine = LogicMachine(
            depth=depth, breadth=breadth, input_dims=input_dims, output_dims=logic_output_dims,
            logic_hidden_dim=logic_hidden_dim, exclude_self=True, residual=True, permute=logic_layer_permute,
        )
        output_dims = self.logic_machine.output_dims
        self.feature_dim = output_dims[0]
        self.logic_machine2 = TemporalLogicMachineDPV2(self.feature_dim, self.feature_dim, 3)

        total_feature_dim = self.feature_dim
        self.decoder = nn.Linear(total_feature_dim, 1) if n_actions == 2 else nn.Linear(total_feature_dim, n_actions)
        self.l1loss = nn.L1Loss()

    def stabilize(self, states):
        siz = states.size()
        w = 5
        padded = nn.ReplicationPad2d(padding=(0, 0, w // 2, w // 2))(states.permute(0, 3, 1, 2)).permute((0, 2, 3, 1))
        # print(states.size(), padded.size())
        length = padded.size(1)
        res = padded[:, :length + 1 - w] / w
        for k in range(1, w):
            res += padded[:, k:length + 1 - w + k] / w
        assert res.size() == siz
        return res

    def forward(self, data=None, hp=None):
        """
        :param data:
            'trajectories': [batch, length, n_agents, state_dim]
            'playerid': [batch]
            'actions': [batch]
            'types': [batch, n_agents, type_dim]
        :param hp: hyper parameters
        """
        trajectories = data['trajectories']
        actions = data['actions']
        device = trajectories.device
        batch, length, n_agents, state_dim = trajectories.size()
        assert state_dim == self.config['state_dim']
        assert actions.size() == torch.Size((batch,))
        assert n_agents == self.config['type_dim']

        types = self.types.to(device)
        states = trajectories

        states = self.stabilize(states)

        beta = hp['beta']
        keep_data = False if 'prerun' not in hp else hp['prerun']
        if 'estimate_parameters' in hp and hp['estimate_parameters']:
            if self.config['input_transformer'] == 'stgcn+joint':
                self.input_transform_joint.estimate_parameters(states)
            else:
                self.input_transform.estimate_parameters(states)
            return None

        binary_logic_layer = True if 'binary_logic_layer' in hp and hp['binary_logic_layer'] else False

        types = types.view(1, n_agents, n_agents).repeat(batch, 1, 1)

        if self.config['input_transformer'] in ['stgcn', 'stgcn+joint']:
            x = states.permute(0, 3, 1, 2).contiguous()
            A = self.A
            for gcn in self.stgcn_layers:
                x, _ = gcn(x, A)
            inputs_to_transform = x.permute(0, 2, 3, 1)
            assert inputs_to_transform.size()[:3] == torch.Size((batch, length, n_agents))
        else:
            inputs_to_transform = states

        nlm_inputs = self.input_transform(inputs_to_transform, add_unary_tensor=types, beta=beta)

        if self.config['input_transformer'] == 'stgcn+joint':
            nlm_inputs_joints = self.input_transform_joint(states, beta=beta)
            from toqnets.nn.nlm import merge as merge_predicates
            nlm_inputs = [merge_predicates(x[0], x[1]) for x in zip(nlm_inputs + [None], nlm_inputs_joints)]

        nlm_outputs = self.logic_machine(nlm_inputs, binary_layer=binary_logic_layer, keep_data=keep_data)

        nltl_input = nlm_outputs[0]
        nltl_output = self.logic_machine2(nltl_input)
        features = nltl_output.max(dim=1)[0]

        assert len(features.size()) == 2 and features.size(0) == trajectories.size(0) and features.size(
            1) == self.feature_dim
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
        if self.config['input_transformer'] == 'stgcn+joint':
            self.input_transform_joint.reset_parameters('primitive_inequality')
        else:
            self.input_transform.reset_parameters('primitive_inequality')

    def primitives_require_grad(self, mode):
        self.input_transform.require_grad(mode)

    def show(self, log=print, save_dir=None, **kwargs):
        pass
