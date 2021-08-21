#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : TrajectorySingleActionNvN.py
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
    return os.path.join('/data/vision/billf/scratch/lzz/TOQ-Nets-PyTorch/data', name)


def get_raw_action(config, action):
    folder = get_folder(config)
    file_list = os.listdir(folder)
    file_list = [x for x in file_list if x.startswith(action) and x.endswith('.npz')]
    np_datas = []
    for name in file_list:
        np_datas.append(np.load(os.path.join(folder, name))['data'])
    return np.concatenate(np_datas, axis=0)


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
    # print(playerid, indices)
    return trajectory[:, indices]


class TrajectorySingleActionNvN:
    default_config = {
        'name': 'TrajectorySingleActionNvN',
        'length': 17,
        'n_players': 6,
        'state_dim': 3,
        'folder': 'SingleAction%dv%d',
        'actions': [
            ('trap', (3500, 500, 1000)),
            ('short_pass', (3500, 500, 1000)),
            ('long_pass', (3500, 500, 1000)),
            ('high_pass', (3500, 500, 1000)),
            ('interfere', (3500, 500, 1000)),
            ('trip', (3500, 500, 1000)),
            ('shot', (3500, 500, 1000)),
            ('deflect', (3500, 500, 1000)),
            ('sliding', (2800, 400, 800))
        ],
        'n_train': None,
        'finetune_scale': 2.0,
        'finetune_subsample': 2,
        'finetune_actions': ['trap', 'short_pass', 'trip', 'shot', 'sliding'],
    }

    def set_equal_sample(self, mode, val):
        self.equal_sample[mode] = val

    def set_gfootball_finetune(self, val):
        self.gfootball_finetune = val

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        config['n_agents'] = config['n_players'] * 2 + 1
        return config

    def _get_actions_action2index(self):
        actions = [x[0] for x in self.config['actions']]
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

        for aid, (action, (n_train, n_val, n_test)) in enumerate(self.config['actions']):
            raw_data = get_raw_action(self.config, action)
            print("Loading action %s" % action)
            assert raw_data.shape[0] >= n_train + n_val + n_test

            block_data = {
                'train': raw_data[:n_train],
                'val': raw_data[n_train: n_train + n_val],
                'test': raw_data[n_train + n_val: n_train + n_val + n_test]
            }
            if self.config['n_train'] is not None:
                n_rep = round(block_data['train'].shape[0] * 8 / 13 / self.config['n_train'])
                block_data['train'] = np.repeat(block_data['train'][:self.config['n_train']], n_rep, axis=0)
                n_val_rep = max(1, round(600 / len(self.config['actions']) / self.config['n_train']))
                block_data['val'] = np.repeat(block_data['val'][:self.config['n_train']], n_val_rep, axis=0)

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

        for key in indices:
            indices[key] = np.concatenate(indices[key], axis=0)
            np.random.seed(233)
            np.random.shuffle(indices[key])

        # for i in range(10):
        #     tra = library['test'][0][0][i]
        #     print(tra[8, 1], tra[8, 7], tra[8, 0])
        # from .visualizer import display_any
        # display_any(library['test'][0][0][0:3], 6, 6, [0, 0, 0], ['a', 'b', 'c'], [1, 1, 1], 'temp')
        # exit()

        return library, indices

    def get_action_list(self):
        return tuple(self.actions)

    def get_item_aid_sid(self, aid, sid, mode):
        trajectories = self.library[mode][aid][0][sid].clone()
        playerid = self.library[mode][aid][1][sid].clone()
        if self.gfootball_finetune:
            subsample = self.config['finetune_subsample']
            scale = self.config['finetune_scale']
            if scale != 1.0:
                trajectories *= scale
            if subsample != 1:
                offset = (trajectories.size(0) % subsample) // 2
                trajectories = trajectories[offset:trajectories.size(0):subsample]
        return {
            'trajectories': trajectories,
            'playerid': playerid,
            'actions': aid,
            'types': self.types
        }

    def getitem_withmode(self, item, mode):
        if self.equal_sample[mode]:
            aid = sample_from_list([i for i in range(len(self.actions))])
            if self.gfootball_finetune:
                sid = random.randint(0, 29)
            else:
                sid = random.randint(0, self.library[mode][aid][0].shape[0] - 1)
        else:
            aid = self.indices[mode][item][0]
            sid = self.indices[mode][item][1]

        return self.get_item_aid_sid(aid, sid, mode)

    def len_withmode(self, mode):
        return self.indices[mode].shape[0]


class TrajectorySingleActionNvN_Wrapper_FewShot_Softmax:
    default_config = {
        'name': 'TrajectorySingleActionNvN_Wrapper_FewShot_Softmax',
        'equal_sample': {'train': False, 'val': False, 'test': False},
        'only_regular': {'train': False, 'val': False, 'test': False},
        'new_actions': [
            ('trap', (50, 25, 25)),
            ('sliding', (50, 25, 25)),
        ]
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        return config

    def complete_sub_config(self, config):
        sub_config = deepcopy(TrajectorySingleActionNvN.complete_config(ConfigUpdate()))
        new_action_dict = {x[0]: x for x in config['new_actions']}
        sub_config['actions'] = [x if x[0] not in new_action_dict else new_action_dict[x[0]] for x in
                                 sub_config['actions']]
        return sub_config

    def set_equal_sample(self, mode, val):
        self.config['equal_sample'][mode] = val

    def set_only_regular(self, mode, val):
        self.config['only_regular'][mode] = val

    def __init__(self, config):
        self.config = config
        sub_config = self.complete_sub_config(config)
        self.dataset = TrajectorySingleActionNvN(sub_config)
        self.library = self.dataset.library
        self.indices = self.dataset.indices

        self.reg_indices = {}
        actions = self.dataset.config['actions']
        self.reg_action_onehot = [False if actions[i][0] in self.config['new_actions'] else True
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
        return [x[0] for x in deepcopy(self.config['new_actions'])]

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
        return {
            'trajectories': self.library[mode][aid][0][sid],
            'playerid': self.library[mode][aid][1][sid],
            'actions': aid,
            'types': self.dataset.types
        }

    def len_withmode(self, mode):
        if self.config['only_regular']:
            return self.reg_indices[mode].shape[0]
        return self.indices[mode].shape[0]


class TrajectorySingleActionNvN_Wrapper_FewShot_Matching:
    default_config = {
        'name': 'TrajectorySingleActionNvN_Wrapper_FewShot_Matching',
        'n_supports': 10,
        'n_targets': 16,
        'only_regular': {'train': False, 'val': False, 'test': False},
        'new_actions': [('trap', (50, 25, 25)), ('sliding', (50, 25, 25))]
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        return config

    def complete_sub_config(self, config):
        sub_config = deepcopy(TrajectorySingleActionNvN.complete_config(ConfigUpdate()))
        new_action_dict = {x[0]: x for x in config['new_actions']}
        sub_config['actions'] = [x if x[0] not in new_action_dict else new_action_dict[x[0]] for x in
                                 sub_config['actions']]
        return sub_config

    def set_only_regular(self, mode, val):
        self.config['only_regular'][mode] = val

    def __init__(self, config):
        self.config = config
        sub_config = self.complete_sub_config(config)
        self.dataset = TrajectorySingleActionNvN(sub_config)
        self.library = self.dataset.library
        self.indices = self.dataset.indices

        self.reg_indices = {}
        actions = self.dataset.config['actions']
        self.reg_action_onehot = [False if actions[i][0] in self.config['new_actions'] else True
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
        # print([len(self.reg_indices[mode]) for mode in ['train', 'val', 'test']])
        # exit()

    def get_action_list(self):
        return self.dataset.get_action_list()

    def getitem_withmode(self, item, mode):
        class_choice = self.reg_action_indices if self.config['only_regular'][
            mode] else self.all_action_indices
        n_classes = len(class_choice)
        n_supports = self.config['n_supports']
        n_targets = self.config['n_targets']
        tra = []
        pid = []
        act = []

        tra_t = []
        pid_t = []
        act_t = []

        to_action_id = []
        for aid in range(n_classes):
            i = class_choice[aid]
            to_action_id.append(i)
            support_choice = np.sort(
                np.random.randint(self.library['train'][i][0].size(0) - n_supports + 1, size=n_supports)
            ) + np.arange(n_supports, dtype=np.int)

            target_choice = np.random.randint(self.library[mode][i][0].size(0), size=n_targets)

            tra.append(self.library['train'][i][0][support_choice])
            pid.append(self.library['train'][i][1][support_choice])
            act.append(torch.ones(n_supports, dtype=torch.long) * aid)
            tra_t.append(self.library[mode][i][0][target_choice])
            pid_t.append(self.library[mode][i][1][target_choice])
            act_t.append(torch.ones(n_targets, dtype=torch.long) * aid)

        tra_t = torch.cat(tra_t, dim=0)
        pid_t = torch.cat(pid_t, dim=0)
        act_t = torch.cat(act_t, dim=0)
        target_indices = torch.randperm(n_classes * n_targets)
        tra_t = tra_t[target_indices][:n_targets]
        pid_t = pid_t[target_indices][:n_targets]
        act_t = act_t[target_indices][:n_targets]

        tra = torch.cat(tra + [tra_t], dim=0)
        pid = torch.cat(pid + [pid_t], dim=0)
        act = torch.cat(act + [act_t], dim=0)

        act_onehot = torch.zeros((act.size(0), n_classes)).scatter(1, act.unsqueeze(1), 1.0)

        to_action_id = torch.LongTensor(to_action_id)

        return {
            'trajectories': tra,
            'playerid': pid,
            'types': self.dataset.types.unsqueeze(0).repeat(act.size(0), 1, 1),
            'support_label_onehot': act_onehot[:-n_targets],
            'target_label': act[-n_targets:],
            'to_action_id': to_action_id
        }

    def len_withmode(self, mode):
        if self.config['only_regular']:
            return self.reg_indices[mode].shape[0] // self.config['n_targets']
        return self.indices[mode].shape[0] // self.config['n_targets']


class TrajectotySingleActionNvN_Wrapper_Binary:
    default_config = {
        'name': 'TrajectotySingleActionNvN_Wrapper_Binary',
        'action': 'trap',
        'select': False,
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        return config

    def complete_sub_config(self, config):
        sub_config = deepcopy(TrajectorySingleActionNvN.complete_config(ConfigUpdate()))
        return sub_config

    def set_equal_sample(self, mode, val):
        pass

    def set_only_regular(self, mode, val):
        pass

    def __init__(self, config):
        self.config = config
        sub_config = self.complete_sub_config(config)
        self.dataset = TrajectorySingleActionNvN(sub_config)
        self.library = self.dataset.library
        self.indices = self.dataset.indices

        self.true_indices = {}
        self.false_indices = {}
        self.all_indices = {}
        actions = self.dataset.config['actions']
        action = self.config['action']
        self.true_onehot = [actions[i][0] == action for i in range(len(actions))]

        for mode in ['train', 'val', 'test']:
            self.true_indices[mode] = np.zeros(self.indices[mode].shape, dtype=np.int)
            self.false_indices[mode] = np.zeros(self.indices[mode].shape, dtype=np.int)
            true_cnt = 0
            false_cnt = 0

            for i in range(self.indices[mode].shape[0]):
                if self.true_onehot[self.indices[mode][i][0]]:
                    self.true_indices[mode][true_cnt] = self.indices[mode][i]
                    true_cnt += 1
                else:
                    self.false_indices[mode][false_cnt] = self.indices[mode][i]
                    false_cnt += 1
            real_cnt = min(true_cnt, false_cnt)
            np.random.seed(233)
            np.random.shuffle(self.true_indices[mode][:true_cnt])
            np.random.shuffle(self.false_indices[mode][:false_cnt])
            self.true_indices[mode] = self.true_indices[mode][:real_cnt]
            self.false_indices[mode] = self.false_indices[mode][:real_cnt]
            self.all_indices[mode] = np.concatenate([self.true_indices[mode], self.false_indices[mode]], axis=0)
            np.random.shuffle(self.all_indices[mode])
        # print(self.true_onehot, self.true_indices['test'].shape, self.false_indices['test'].shape)
        # print(self.len_withmode('test'))
        # for i in range(min(self.len_withmode('test'), 50)):
        #     print(self.getitem_withmode(i, 'test')['actions'], self.all_indices[mode][i][0])
        # exit()

    def get_action_list(self):
        return ["not_%s" % self.config['action'], self.config['action']]

    def getitem_withmode(self, item, mode):
        aid, sid = self.all_indices[mode][item]
        trajectory = self.library[mode][aid][0][sid].clone()
        playerid = self.library[mode][aid][1][sid]
        n_players = self.dataset.config['n_players']
        select = self.config['select']
        n_agents = n_players * 2 + 1
        if self.config['action'] == 'trap':
            n_select_agents = 2 if select else n_agents
            types = torch.zeros(n_select_agents, 4).type(torch.long)
            if self.true_onehot[aid] or sid % 2 < 2:
                states = move_player_to_first(trajectory, playerid, n_players, n_players)
                types[0, 0] = types[1, 1] = 1
            return {
                'trajectories': states[:, :n_select_agents],
                'playerid': -1,
                'actions': 1 if self.true_onehot[aid] else 0,
                'original_actions': aid,
                'types': types[:n_select_agents],
            }
        elif self.config['action'] == 'deflect':
            n_select_agents = 2 if select else n_agents
            types = torch.zeros(n_select_agents, 4).type(torch.long)
            if self.true_onehot[aid] or sid % 2 < 1:
                states = move_player_to_first(trajectory, playerid, n_players, n_players)
                types[0, 0] = types[1, 1] = 1
            else:
                dist_l = abs(trajectory[trajectory.size(0) // 2, 0, 0] - trajectory[trajectory.size(0) // 2, 1, 0])
                dist_r = abs(trajectory[trajectory.size(0) // 2, 0, 0] - trajectory[
                    trajectory.size(0) // 2, n_players + 1, 0])
                temp_pid = 1 if dist_l < dist_r else n_players + 1
                states = move_player_to_first(trajectory, temp_pid, n_players, n_players)
                types[0, 0] = types[1, 1] = 1
            return {
                'trajectories': states[:, :n_select_agents],
                'playerid': -1,
                'actions': 1 if self.true_onehot[aid] else 0,
                'original_actions': aid,
                'types': types[:n_select_agents],
            }
        elif self.config['action'] == 'shot':
            n_select_agents = 3 if select else n_agents
            types = torch.zeros(n_select_agents, 4).type(torch.long)
            if self.true_onehot[aid] or sid % 2 < 1:
                states = move_player_to_first(trajectory, playerid, n_players, n_players)
                if select:
                    states = states.index_select(dim=1, index=torch.LongTensor([0, 1, n_players + 1]))
                    types[0, 0] = types[1, 1] = types[2, 3] = 1
                else:
                    types[0, 0] = types[1, 1] = types[n_players + 1, 3] = 1
            else:
                dist_l = abs(trajectory[trajectory.size(0) // 2, 0, 0] - trajectory[trajectory.size(0) // 2, 1, 0])
                dist_r = abs(trajectory[trajectory.size(0) // 2, 0, 0] - trajectory[
                    trajectory.size(0) // 2, n_players + 1, 0])
                # print(item, dist_l, dist_r)
                # print('3:', trajectory[trajectory.size(0) // 2, 0, 0], trajectory[trajectory.size(0) // 2, 1, 0],
                #       trajectory[trajectory.size(0) // 2, 7, 0])
                avg_half_dist = ((trajectory[:trajectory.size(0) // 2, 0:1, :2] -
                                  trajectory[:trajectory.size(0) // 2, :, :2]) ** 2).sum(dim=(0, 2))
                if dist_l < dist_r:
                    temp_pid = avg_half_dist[n_players + 1:].argmax(0) + n_players + 1
                else:
                    temp_pid = avg_half_dist[1:n_players + 1].argmax(0) + 1

                states = move_player_to_first(trajectory, temp_pid, n_players, n_players)
                if select:
                    states = states.index_select(dim=1, index=torch.LongTensor([0, 1, n_players + 1]))
                    types[0, 0] = types[1, 1] = types[2, 3] = 1
                else:
                    types[0, 0] = types[1, 1] = types[n_players + 1, 3] = 1
            return {
                'trajectories': states[:, :n_select_agents],
                'playerid': -1,
                'actions': 1 if self.true_onehot[aid] else 0,
                'original_actions': aid,
                'types': types[:n_select_agents],
            }
        elif self.config['action'] in ['short_pass', 'long_pass', 'high_pass']:
            n_select_agents = 3 if select else n_agents
            types = torch.zeros(n_select_agents, 4).type(torch.long)
            if self.true_onehot[aid] or sid % 2 < 2:
                states = move_player_to_first(trajectory, playerid, n_players, n_players)
                min_second_half_dist = ((states[states.size(0) // 2:, 0:1, :2] -
                                         states[states.size(0) // 2:, :, :2]) ** 2).sum(dim=2).min(dim=0)[
                    0].view(n_agents)
                min_second_half_dist[0:2] = 1e9
                min_second_half_dist[n_players + 1:] = 1e9
                closest_teammate = min_second_half_dist.argmin(dim=0)
                if select:
                    states = states.index_select(dim=1, index=torch.LongTensor([0, 1, closest_teammate]))
                    types[0, 0] = types[1, 1] = types[2, 2 if closest_teammate <= n_players else 3] = 1
                else:
                    types[0, 0] = types[1, 1] = types[
                        closest_teammate, 2 if closest_teammate <= n_players else 3] = 1
            return {
                'trajectories': states[:, :n_select_agents],
                'playerid': -1,
                'actions': 1 if self.true_onehot[aid] else 0,
                'original_actions': aid,
                'types': types[:n_select_agents],
            }
        elif self.config['action'] == 'trip':
            n_select_agents = 3 if select else n_agents
            types = torch.zeros(n_select_agents, 4).type(torch.long)
            if self.true_onehot[aid] or sid % 2 < 2:
                n_players = self.dataset.config['n_players']
                states = move_player_to_first(trajectory, playerid, n_players, n_players)
                mid_time = states.size(0) // 2
                mid_left = mid_time - states.size(0) // 4
                mid_right = mid_time + states.size(0) // 4
                min_mid_dist = ((states[mid_left:mid_right + 1, 1:2, :2] -
                                 states[mid_left:mid_right + 1, :, :2]) ** 2).sum(dim=2).min(dim=0)[0].view(
                    n_agents)
                min_mid_dist[0:2] = 1e9
                closest_player = min_mid_dist.argmin(dim=0)
                if select:
                    states = states.index_select(dim=1, index=torch.LongTensor([0, 1, closest_player]))
                    types[0, 0] = types[1, 1] = types[2, 2 if closest_player <= n_players else 3] = 1
                else:
                    types[0, 0] = types[1, 1] = types[closest_player, 2 if closest_player <= n_players else 3] = 1
            return {
                'trajectories': states[:, :n_select_agents],
                'playerid': -1,
                'actions': 1 if self.true_onehot[aid] else 0,
                'original_actions': aid,
                'types': types[:n_select_agents],
            }
        elif self.config['action'] == 'interfere':
            raise NotImplementedError()
            # if self.true_onehot[aid] or sid %2<2:
            #     pass
            # return {
            #     'trajectories': states,
            #     'playerid': -1,
            #     'actions': 1 if self.true_onehot[aid] else 0,
            #     'types': -1,
            # }
        else:
            raise NotImplementedError()

    def len_withmode(self, mode):
        return self.all_indices[mode].shape[0]
