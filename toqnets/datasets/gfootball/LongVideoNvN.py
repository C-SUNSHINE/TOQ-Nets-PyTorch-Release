#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : LongVideoNvN.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import os
import random
from copy import deepcopy

import numpy as np
import torch

from toqnets.config_update import ConfigUpdate, update_config
from toqnets.datasets.utils import sample_from_list


def get_folder(config):
    name = config['folder'] % (config['n_players'], config['n_players'])
    return os.path.join('data', name)


def get_raw_action(config, action):
    folder = get_folder(config)
    file_list = os.listdir(folder)
    file_list = [x for x in file_list if x.startswith(action) and x.endswith('.npz')]
    np_datas = []
    for name in file_list:
        np_datas.append(np.load(os.path.join(folder, name))['data'])
    np_datas = np.concatenate(np_datas, axis=0)
    print(action, np_datas.shape[0])
    return np_datas


def move_player_to_first(trajectory, playerid, n_left_players, n_right_players):
    playerid = int(playerid)
    left_indices = [i for i in range(1, n_left_players + 1)]
    right_indices = [i for i in range(n_left_players + 1, n_left_players + n_right_players + 1)]
    if playerid > n_left_players:
        k = playerid - n_left_players - 1
        right_indices = right_indices[k:k + 1] + right_indices[:k] + right_indices[k + 1:]
        indices = [0] + right_indices + left_indices
    else:
        k = playerid - 1
        left_indices = left_indices[k:k + 1] + left_indices[:k] + left_indices[k + 1:]
        indices = [0] + left_indices + right_indices
    return trajectory[:, indices]


class LongVideoNvN:
    default_config = {
        'name': 'LongVideoNvN',
        'length': 61,
        'n_players': 6,
        'temporal': 'exact',
        'state_dim': 3,
        'folder': 'LongVideo%dv%d',
        'actions': [
            'trap',
            'short_pass',
            'long_pass',
            'high_pass',
            'interfere',
            'trip',
            'shot',
            'deflect',
            'sliding'
        ],
        'n_train': None,
        'finetune_n_train': 5,
        'finetune_scale': 2.0,
        'finetune_subsample': 2,
        'exact_length': 25,
    }

    def set_equal_sample(self, mode, val):
        self.equal_sample[mode] = val

    def set_only_regular(self, mode, val):
        pass

    def set_gfootball_finetune(self, val):
        self.gfootball_finetune = val

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        config['n_agents'] = config['n_players'] * 2 + 1
        return config

    def _get_actions_action2index(self):
        actions = self.config['actions']
        action2index = {a: i for i, a in enumerate(actions)}
        return actions, action2index

    def __init__(self, config):
        """
        :param config: config for dataset, in addition to the default self.config
        """
        self.config = config
        self.actions, self.action2index = self._get_actions_action2index()
        self.library, self.indices = self.fetch()
        n_players = self.config['n_players']
        n_agents = self.config['n_agents']
        self.types = torch.zeros(n_agents, 3)
        for k in range(n_agents):
            self.types[k, 0 if k == 0 else (1 if k <= n_players else 2)] = 1
        self.equal_sample = {'train': False, 'val': False, 'test': False}
        self.gfootball_finetune = False

    def fetch(self):
        n_agents = self.config['n_agents']
        length = self.config['length']
        state_dim = self.config['state_dim']

        library = {'train': [], 'val': [], 'test': []}
        indices = {'train': [], 'val': [], 'test': []}

        for aid, action in enumerate(self.config['actions']):
            raw_data = get_raw_action(self.config, action)
            rng = np.random.default_rng(2333)
            rng.shuffle(raw_data)
            print("Loading action %s" % action)
            num = raw_data.shape[0]
            if self.config['n_train'] is None:
                end_train = num * 8 // 13
                end_val = num * 10 // 13
                n_rep = 1
                n_rep_val = 1
            elif self.config['n_train'] >= 1:
                end_train = self.config['n_train']
                end_val = end_train * 2
                n_rep = max(1, int(num * 8 / 13 / end_train))
                n_rep_val = max(1, 1024 // end_train)
                assert end_val < num
            else:
                assert 0 < self.config['n_train'] < 1
                end_train = int(self.config['n_train'] * num)
                end_val = end_train * 2
                n_rep = 1
                n_rep_val = 1
                assert end_train < end_val < num

            block_data = {
                'train': raw_data[:end_train],
                'val': raw_data[end_train: end_val],
                'test': raw_data[end_val:]
            }
            block_data['train'] = np.repeat(block_data['train'], n_rep, axis=0)
            block_data['val'] = np.repeat(block_data['val'], n_rep_val, axis=0)

            for mode in ['train', 'val', 'test']:
                raw = block_data[mode]
                tra = np.zeros((raw.shape[0], length, n_agents, state_dim))
                pid = np.zeros(raw.shape[0]).astype(np.int)
                act = np.zeros(raw.shape[0]).astype(np.int)

                for k in range(n_agents):
                    if k == 0:
                        tra[:, :, k, :] = raw[:, :, 0:3]
                    else:
                        tra[:, :, k, :2] = raw[:, :, (k - 1) * 13 + 3:(k - 1) * 13 + 5]
                    if k > 0:
                        pid_k = raw[:, length // 2, -n_agents + k].astype(np.int) * k
                        # print('raw[:, length, -n_agents + k] = ', raw[:, (length + 1) // 2, -n_agents + k])
                        # print("pid_k", pid_k)
                        pid = np.maximum(pid, pid_k)
                        # pid += pid_k
                tra[:, :, :, :2] *= 60
                tra[:, :, :, 2:] *= 0.3
                act += aid
                n_data = raw.shape[0]
                library[mode].append((torch.Tensor(tra), torch.LongTensor(pid)))
                index = np.zeros((n_data, 2), dtype=np.int)
                index[:, 0] = aid
                index[:, 1] = np.arange(n_data)
                indices[mode].append(index)
                print(action, mode, library[mode][-1][0].size(0))

        for key in indices:
            indices[key] = np.concatenate(indices[key], axis=0)
            np.random.seed(233)
            np.random.shuffle(indices[key])

        return library, indices

    def get_action_list(self):
        return tuple(self.actions)

    def get_item_aid_sid(self, aid, sid, mode, library=None):
        if library is None:
            library = self.library
        trajectories = library[mode][aid][0][sid].clone()
        playerid = library[mode][aid][1][sid].clone()
        if self.gfootball_finetune:
            subsample = self.config['finetune_subsample']
            scale = self.config['finetune_scale']
            if scale != 1.0:
                trajectories *= scale
            if subsample != 1:
                offset = (trajectories.size(0) % subsample) // 2
                trajectories = trajectories[offset:trajectories.size(0):subsample]
        length = trajectories.size(0)
        if self.config['temporal'] == 'all':
            start_time = random.randint(12, 26)
            duration = 25
        elif self.config['temporal'] == 'left':
            start_time = random.randint(12, 19)
            duration = 25
        elif self.config['temporal'] == 'right':
            start_time = random.randint(19, 26)
            duration = 25
        elif self.config['temporal'] == 'exact':
            duration = self.config['exact_length']
            start_time = length // 2 - (duration - 1) // 2
        else:
            raise ValueError()
        trajectories = trajectories[start_time:start_time + duration]
        middle = self.config['length'] // 2 - start_time
        return {
            'trajectories': trajectories,
            'playerid': playerid,
            'actions': aid,
            'types': self.types,
            'sample_ids': sid,
            'middle': middle,
        }

    def getitem_withmode(self, item, mode):
        if self.equal_sample[mode]:
            aid = sample_from_list([i for i in range(len(self.actions))])
            sid = random.randint(0, self.library[mode][aid][0].shape[0] - 1)
            if self.gfootball_finetune:
                sid = random.randint(0, self.config['finetune_n_train'])
            weight = 1.0
        else:
            aid = self.indices[mode][item][0]
            sid = self.indices[mode][item][1]
            total = self.len_withmode(mode)
            n_actions = len(self.actions)
            fre = self.library[mode][aid][0].shape[0]
            weight = total / n_actions / fre

        dat = self.get_item_aid_sid(aid, sid, mode)
        dat['weight'] = weight
        return dat

    def len_withmode(self, mode):
        return self.indices[mode].shape[0]


class LongVideoNvN_Wrapper_FewShot_Softmax:
    default_config = {
        'name': 'LongVideoNvN_Wrapper_FewShot_Softmax',
        'equal_sample': {'train': False, 'val': False, 'test': False},
        'only_regular': {'train': False, 'val': False, 'test': False},
        'new_actions': [
            'trap', 'sliding'
        ],
        'n_few_shot': 50,
        'exact_length': None,
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        return config

    def complete_sub_config(self, config):
        sub_args = {}
        if config['exact_length'] is not None:
            sub_args['exact_length'] = config['exact_length']
        sub_config = deepcopy(LongVideoNvN.complete_config(ConfigUpdate(sub_args)))
        return sub_config

    def set_equal_sample(self, mode, val):
        self.config['equal_sample'][mode] = val

    def set_only_regular(self, mode, val):
        self.config['only_regular'][mode] = val

    def fetch_from_sub(self, library):
        indices = {'train': [], 'val': [], 'test': []}
        n_few_shot = self.config['n_few_shot']
        new_library = {'train': [], 'val': [], 'test': []}
        for aid, action in enumerate(self.get_action_list()):
            if action in self.config['new_actions']:
                tra = torch.cat((library['train'][aid][0], library['val'][aid][0], library['test'][aid][0]),
                                dim=0)
                pid = torch.cat((library['train'][aid][1], library['val'][aid][1], library['test'][aid][1]),
                                dim=0)
                new_library['train'].append((tra[:n_few_shot], pid[:n_few_shot]))
                new_library['val'].append((tra[n_few_shot:n_few_shot * 2], pid[n_few_shot:n_few_shot * 2]))
                new_library['test'].append((tra[n_few_shot * 2:], pid[n_few_shot * 2:]))
            else:
                for split in ('train', 'val', 'test'):
                    new_library[split].append(library[split][aid])
            for split in ('train', 'val', 'test'):
                n_data = new_library[split][aid][0].shape[0]
                index = np.zeros((n_data, 2), dtype=np.int)
                index[:, 0] = aid
                index[:, 1] = np.arange(n_data)
                indices[split].append(index)
                print('%s %s %d' % (action, split, n_data))
        for split in ('train', 'val', 'test'):
            indices[split] = np.concatenate(indices[split], axis=0)
        return new_library, indices

    def __init__(self, config):
        self.config = config
        sub_config = self.complete_sub_config(config)
        self.dataset = LongVideoNvN(sub_config)
        self.library, self.indices = self.fetch_from_sub(self.dataset.library)

        self.reg_indices = {}
        actions = self.dataset.config['actions']
        self.reg_action_onehot = [False if actions[i] in self.config['new_actions'] else True
                                  for i in range(len(actions))]
        self.reg_action_indices = [i for i in range(len(actions)) if self.reg_action_onehot[i]]
        self.all_action_indices = [i for i in range(len(actions))]

        for mode in ['train', 'val', 'test']:
            self.reg_indices[mode] = np.zeros(self.indices[mode].shape, dtype=np.int)
            j = 0

            for i in range(self.indices[mode].shape[0]):
                if self.reg_action_onehot[self.indices[mode][i][0]]:
                    self.reg_indices[mode][j] = self.indices[mode][i]
                    j += 1
            self.reg_indices[mode] = self.reg_indices[mode][:j]

    def get_action_list(self):
        return self.dataset.get_action_list()

    def get_new_actions(self):
        return [x for x in deepcopy(self.config['new_actions'])]

    def getitem_withmode(self, item, mode):
        aid, sid = None, None
        if self.config['equal_sample'][mode]:
            if self.config['only_regular'][mode]:
                aid = sample_from_list(self.reg_action_indices)
            else:
                aid = sample_from_list(self.all_action_indices)
            sid = random.randint(0, self.library[mode][aid][0].shape[0] - 1)
        else:
            if self.config['only_regular'][mode]:
                aid = self.reg_indices[mode][item][0]
                sid = self.reg_indices[mode][item][1]
            else:
                aid = self.indices[mode][item][0]
                sid = self.indices[mode][item][1]
        return self.dataset.get_item_aid_sid(aid, sid, mode, self.library)

    def len_withmode(self, mode):
        if self.config['only_regular'][mode]:
            return self.reg_indices[mode].shape[0]
        return self.indices[mode].shape[0]


if __name__ == '__main__':
    config = LongVideoNvN.complete_config(ConfigUpdate({
        'n_players': 8,
        'n_train': 0.1,
    }))
    dataset = LongVideoNvN(config)
