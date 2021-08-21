#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : TrajectoryWithActionNvN.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import os

import numpy as np
import torch

from . import cfg


def get_path(config, mode):
    name = 'TrajectoryWithAction%dv%d' % (config['n_players'], config['n_players'])
    return os.path.join('data', name, name + '_' + mode)


class TrajectoryWithActionNvN:
    default_config = {
        'name': 'TrajectoryWithActionNvN',
        'n_data': {
            'train': 12288,
            'val': 2048,
            'test': 2048
        },
        'length': 50,
        'n_players': 6,
        'state_dim': 3,
        'actions': ['none', 'movement', 'ball_control', 'trap', 'short_pass', 'long_pass', 'high_pass', 'shot',
                    'deflect', 'interfere', 'trip', 'sliding'],
        'weight': [0, 0.09654126716226998, 2.0232633573855643, 24.57845784578458, 14.329639232362966,
                   172.96891495601173, 79.96355827422398, 547.1465677179963, 733.3200716132883, 452.87469287469287,
                   76.78480300773805, 3301.7465293327364]
    }

    def complete_config(self):
        self.config['n_agents'] = self.config['n_players'] * 2 + 1
        self.config['n_actions'] = len(self.config['actions'])

    def __init__(self, config, mode=None):
        """
        :param config: config for dataset, in addition to the default self.config
        :param mode: 'train', 'val' or 'test'
        """
        self.config = config
        self.complete_config()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.trajectories, self.actions = self.fetch()
        self.trajectories, self.actions = torch.Tensor(self.trajectories), torch.LongTensor(self.actions)
        n_players = self.config['n_players']
        n_agents = self.config['n_agents']
        self.types = torch.zeros(n_agents, 3)
        for k in range(n_agents):
            self.types[k, 0 if k == 0 else (1 if k <= n_players else 2)] = 1

    def fetch(self):
        raw = np.load(get_path(self.config, self.mode) + '.npz')

        # tmp = raw['actions']
        # print(tmp.shape)
        # ccc = [0 for i in range(15)]
        # for i in range(tmp.shape[0]):
        #     for j in range(tmp.shape[1]):
        #         for k in range(tmp.shape[2]):
        #             ccc[tmp[i, j, k]] += 1
        # print(ccc)
        # exit()

        action_indices = [cfg.ACTION_TO_INDEX[action] for action in self.config['actions']]
        map_index = -np.ones(max(action_indices) + 1, dtype=np.int)
        for i, index in enumerate(action_indices):
            map_index[index] = i

        trajectories = raw['trajectories']
        # print(map_index, raw['actions'], raw['actions'].max())
        # exit()
        actions = map_index[raw['actions']]

        # print(map_index)

        assert actions.min() >= 0

        assert trajectories.shape[1] >= self.config['length']
        trajectories = trajectories[:, :self.config['length']]
        assert trajectories.shape == (
            trajectories.shape[0], trajectories.shape[1], self.config['n_agents'], self.config['state_dim'])
        assert actions.shape[1] >= self.config['length']
        actions = actions[:, :self.config['length']]
        assert actions.shape == (trajectories.shape[0], trajectories.shape[1], self.config['n_agents'])
        return trajectories, actions

    def __getitem__(self, item):
        return {
            'trajectories': self.trajectories[item],
            'actions': self.actions[item],
            'types': self.types
        }

    def __len__(self):
        return self.trajectories.shape[0]
