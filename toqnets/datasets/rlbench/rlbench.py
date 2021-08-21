#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rlbench.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import bz2
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from toqnets.config_update import ConfigUpdate, update_config
from toqnets.datasets.rlbench.raw_object_name_list import get_raw_object_name_list
from toqnets.datasets.utils import sample_from_list
from .rlbench_cfg import *


class RLBench:
    default_config = {
        'name': 'RLBench',
        'toy': False,
        'length': 100,
        'n_objs': 45,
        'state_dim': [14, 14],
        'object_name_dim': 194,
        'actions': 'all',
        'n_train': None,
        'version': 'v2',
        'temporal_scale': None,  # f2 s2 ...
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        for k in cls.default_config.keys():
            if k not in config:
                config[k] = deepcopy(cls.default_config[k])
        return config

    @classmethod
    def parse_filename(cls, filename):
        """
        return action_name, number of trajectories, id
        """
        words = filename.split('_')
        return words[0], int(words[1][1:]), int(words[2])

    @classmethod
    def _get_filelist(cls):
        filelist = os.listdir(os.path.join(DATA_PATH, 'data'))
        filelist = [x[:-4] for x in filelist if x.endswith('.pkl')]
        return filelist

    def _get_obj_name2index(self):
        names = deepcopy(get_raw_object_name_list())

        names = [x for x in names if 'waypoint' not in x]

        res = {}
        for i, name in enumerate(names):
            res[name] = i
        assert self.config['object_name_dim'] == 0 or len(res) <= self.config['object_name_dim']
        return res

    def set_equal_sample(self, mode, val):
        self.equal_sample[mode] = val

    def set_only_regular(self, mode, val):
        pass

    def set_temporal_scale(self, val):
        self.config['temporal_scale'] = val

    def _filelist_filter(self, filelist):
        filelist = list(sorted(filelist))
        filelist = [filename for filename in filelist if self.parse_filename(filename)[1] <= 100]
        filelist_of_action = {}
        for filename in filelist:
            action = self.parse_filename(filename)[0]
            if action not in filelist_of_action:
                filelist_of_action[action] = []
            filelist_of_action[action].append(filename)
        new_list = []
        for a, l in filelist_of_action.items():
            sample_cnt = 0
            for fn in l:
                n_samples = self.parse_filename(fn)[1]
                sample_cnt += n_samples
                new_list.append(fn)
                if sample_cnt >= N_SAMPLE_PER_ACTION:
                    break
        return new_list

    def _toy_filter(self, filelist):
        return [filename for filename in filelist if
                (self.parse_filename(filename)[0] in toy_actions or toy_actions == 'all')]

    def _get_obj_name_index(self, name, onehot=False):
        if name not in self.obj_name2index:
            index = -1
        else:
            index = self.obj_name2index[name]
        if onehot:
            res = np.zeros((len(self.config['object_name_dim']),))
            if index != -1:
                res[index] = 1
            return res
        return index

    def fetch(self, filelist):
        self.info = []
        self.data = {'nullary': [], 'unary': []}
        index = 0
        length = self.config['length']
        cnt = {}
        for filename in filelist:
            action, n_trajectories, _ = self.parse_filename(filename)
            if action not in self.actions:
                continue
            action_id = self.action2index[action]
            fullname = os.path.join(DATA_PATH, 'data', filename + '.pkl')
            with bz2.open(fullname, 'rb') as fin:
                print('loading ', fullname)
                trajectories = pickle.load(fin)['trajectory']
                assert len(trajectories) == n_trajectories
                for traj in tqdm(trajectories):
                    frames = traj[(len(traj) - length) // 2:(len(traj) - length) // 2 + length] if len(
                        traj) > length else traj[:]
                    inf = {'index': index, 'length': len(frames), 'action': action, 'action_id': action_id}
                    index += 1
                    nullary_dat = []
                    unary_dat = []
                    for t, frame in enumerate(frames):
                        obs = frame['observation']
                        nullary_state = np.concatenate((
                            obs['gripper_pose'],
                            [obs['gripper_open']],
                            obs['gripper_touch_forces']), axis=0)
                        assert nullary_state.shape[0] == self.config['state_dim'][0]
                        unary_state = []
                        for elem in obs['task_low_dim_state_json']:
                            name = elem['name']
                            color = elem['color'] if 'color' in elem else [0., 0., 0.]
                            exist = elem['exist']
                            pose = elem['pose']
                            if 'bbox' in elem:
                                bbox = elem['bbox']
                            else:
                                bbox = np.zeros(6)
                            if name in self.obj_name2index:
                                if self.config['version'] == 'v1':
                                    object_unary_state = np.concatenate((
                                        color,  # len=3
                                        [float(exist)],  # len=1
                                        pose,  # len=7
                                    ), axis=0)
                                elif self.config['version'] == 'v2':
                                    object_unary_state = np.concatenate((
                                        bbox,  # len=6
                                        [float(exist)],  # len=1
                                        pose,  # len=7
                                    ), axis=0)
                                else:
                                    raise ValueError()
                                if self.config['object_name_dim'] > 0:
                                    object_unary_state = np.concatenate([
                                        object_unary_state,
                                        self._get_obj_name_index(name, onehot=True)
                                    ], axis=0)
                                unary_state.append(object_unary_state)
                                assert unary_state[-1].shape[0] == self.config['state_dim'][1] + self.config[
                                    'object_name_dim']
                        unary_state = np.stack(unary_state, axis=0)
                        nullary_dat.append(nullary_state)
                        unary_dat.append(unary_state)
                    nullary_dat = torch.from_numpy(np.stack(nullary_dat, axis=0)).float()
                    unary_dat = torch.from_numpy(np.stack(unary_dat, axis=0)).float()
                    self.info.append(inf)
                    self.data['nullary'].append(nullary_dat)
                    self.data['unary'].append(unary_dat)

        self.split_indices = {'train': [], 'val': [], 'test': []}
        self.action_cnt = {'train': {}, 'val': {}, 'test': {}}
        self.action_split_indices = {}
        for index, inf in enumerate(self.info):
            assert inf['index'] == index
            action = inf['action']
            if action not in self.action_indices:
                self.action_indices[action] = []
            self.action_indices[action].append(index)

        for action in self.actions:
            self.action_split_indices[action] = {'train': [], 'val': [], 'test': []}
            self.action_indices[action] = np.array(self.action_indices[action])
            np.random.seed(233)
            np.random.shuffle(self.action_indices[action])
            num = self.action_indices[action].shape[0]
            end_train = num * 8 // 13
            end_val = num * 10 // 13
            if 'n_train' in self.config and self.config['n_train'] is not None:
                end_train = self.config['n_train']
            self.action_split_indices[action]['train'] = self.action_indices[action][:end_train]
            self.action_split_indices[action]['val'] = self.action_indices[action][end_train:end_val]
            self.action_split_indices[action]['test'] = self.action_indices[action][end_val:num]
            for mode in ['train', 'val', 'test']:
                self.split_indices[mode].append(self.action_split_indices[action][mode])
            self.action_cnt['train'][action] = end_train
            self.action_cnt['val'][action] = end_val - end_train
            self.action_cnt['test'][action] = num - end_val

        for mode in ['train', 'val', 'test']:
            self.split_indices[mode] = np.concatenate([np.array(x, dtype=np.int) for x in self.split_indices[mode]],
                                                      axis=0)
            np.random.seed(233)
            np.random.shuffle(self.split_indices[mode])

    def __init__(self, config, filelist=None):
        self.config = deepcopy(config)
        self.info = None
        self.data = None
        self.obj_name2index = self._get_obj_name2index()
        self.action_indices = {}
        self.action_cnt = {}
        self.split_indices = {}
        self.action_split_indices = {}
        if filelist is None:
            filelist = self._get_filelist()
            if self.config['version'] == 'v2' or self.config['toy']:
                filelist = self._filelist_filter(filelist)
            if self.config['toy']:
                filelist = self._toy_filter(filelist)
        all_actions = list(sorted(list(set(self.parse_filename(filename)[0] for filename in filelist))))
        if isinstance(self.config['actions'], str):
            self.actions = all_actions if self.config['actions'] == 'all' else deepcopy(
                default_action_lists[self.config['actions']])
        else:
            self.actions = self.config['actions']
        self.actions = list(sorted(list(set(self.actions).intersection(set(all_actions)))))
        self.action2index = {a: i for i, a in enumerate(self.actions)}
        self.fetch(filelist)
        self.equal_sample = {'train': False, 'val': False, 'test': False}

    def get_action_list(self):
        return deepcopy(self.actions)

    def _expand_nullary(self, nullary):
        length = self.config['length']
        if nullary.size(0) < length:
            nullary = torch.cat((nullary, nullary[-1:].repeat(length - nullary.size(0), 1)), dim=0)
        return nullary

    def _expand_unary(self, unary):
        length = self.config['length']
        n = self.config['n_objs']
        if unary.size(1) < n:
            unary = torch.cat((unary, torch.zeros(unary.size(0), n - unary.size(1), unary.size(2))), dim=1)
        if unary.size(0) < length:
            unary = torch.cat((unary, unary[-1:].repeat(length - unary.size(0), 1, 1)), dim=0)
        return unary

    def _time_scale(self, states):
        cut = False
        temporal_scale = self.config['temporal_scale']
        if temporal_scale is None:
            return states
        elif temporal_scale.startswith('f'):
            k = int(temporal_scale[1:])
            return states[::k]
        elif temporal_scale.startswith('s'):
            k = int(temporal_scale[1:])
            subs = []
            pre = states[:-1]
            nex = states[1:]
            for i in range(k):
                subs.append((pre * (k - i) + nex * i) / k)
            subs = torch.cat(subs, dim=1)
            subs = subs.view(subs.size(0) * k, *states.size()[1:])
            if cut:
                subs = subs[:states.size(1)]
                assert subs.size() == states.size()
            return subs
        else:
            raise ValueError()

    def getitem_from_index(self, index, mode):
        res = {
            'actions': self.info[index]['action_id'],
            'lengths': self.info[index]['length'],
            'nullary_states': self._time_scale(self._expand_nullary(self.data['nullary'][index].clone())),
            'unary_states': self._time_scale(self._expand_unary(self.data['unary'][index].clone()))
        }
        return res

    def getitem_withmode(self, item, mode):
        if self.equal_sample[mode]:
            action = sample_from_list(self.actions)
            index = sample_from_list(self.action_split_indices[action][mode])
        else:
            index = self.split_indices[mode][item]
        dat = self.getitem_from_index(index, mode)
        if self.equal_sample[mode]:
            weight = 1.0
        else:
            fre = self.action_cnt[mode][self.actions[dat['actions']]]
            tot = self.len_withmode(mode)
            n_actions = len(self.actions)
            weight = tot / n_actions / fre
        dat['weight'] = weight
        return dat

    def len_withmode(self, mode):
        return self.split_indices[mode].shape[0]


class RLBench_Fewshot:
    default_config = {
        'name': 'RLBench_Fewshot',
        'toy': False,
        'length': 100,
        'n_objs': 45,
        'state_dim': [14, 14],
        'object_name_dim': 194,
        'actions': 'Open',
        'labels': 'Open',
        'n_train': None,
        'n_new_train': 10,
        'seed': 233,
        'temporal_scale': None,
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        for k in cls.default_config.keys():
            if k not in config:
                config[k] = deepcopy(cls.default_config[k])
        return config

    def complete_sub_config(self, config):
        sub_config = deepcopy(RLBench.complete_config(ConfigUpdate({
            'toy': config['toy'],
            'length': config['length'],
            'n_objs': config['n_objs'],
            'state_dim': config['state_dim'],
            'object_name_dim': config['object_name_dim'],
            'actions': 'all',
            'n_train': None,
            'temporal_scale': config['temporal_scale']
        })))
        return sub_config

    def set_equal_sample(self, mode, val):
        self.equal_sample[mode] = val

    def set_only_regular(self, mode, val):
        self.only_regular[mode] = val

    def set_temporal_scale(self, val):
        self.dataset.set_temporal_scale(val)

    def _get_filelist(self, toy):
        filelist = os.listdir(os.path.join(DATA_PATH, 'data'))
        filelist = [tuple(RLBench.parse_filename(x[:-4]) + (x[:-4],)) for x in filelist if x.endswith('.pkl')]
        label_actions = {}
        max_cnt = 0
        action_filelist = {}
        multiplier = 100
        for action, num, pkg_id, filename in filelist:
            if toy and toy_actions != 'all' and action not in toy_actions:
                continue
            label = self.label_action(action)
            if label not in label_actions:
                label_actions[label] = set()
            label_actions[label].add(action)
            max_cnt = max(max_cnt, len(label_actions[label]))
            if action not in action_filelist:
                action_filelist[action] = []
            action_filelist[action].append(filename)
        new_filelist = []
        for action in action_filelist:
            label = self.label_action(action)
            cnt = len(label_actions[label])
            n_samples = round(max_cnt / cnt * multiplier)
            action_filelist[action] = list(reversed(sorted(
                action_filelist[action], key=lambda x: RLBench.parse_filename(x)[1]
            )))
            for filename in action_filelist[action]:
                new_filelist.append(filename)
                n_samples -= RLBench.parse_filename(filename)[1]
                if n_samples <= 0:
                    break

        return new_filelist

    def _split_data(self):
        split_indices = {'train': [], 'val': [], 'test': []}
        split_indices_reg = {'train': [], 'val': [], 'test': []}
        action_split_indices = {'train': {}, 'val': {}, 'test': {}}
        action_cnt = {'train': {}, 'val': {}, 'test': {}}
        label_cnt = {'train': {}, 'val': {}, 'test': {}}
        label_cnt_reg = {'train': {}, 'val': {}, 'test': {}}
        n_train = self.config['n_train']
        n_new_train = self.config['n_new_train']
        for action in self.sub_actions:
            num = self.action_indices[action].shape[0]
            if action in self.new_actions:
                end_train = n_new_train
                end_val = n_new_train * 2
                np.random.seed(self.config['seed'])
                np.random.shuffle(self.action_indices[action])
            elif n_train is not None:
                end_train = n_train
                end_val = n_train + max(5, n_train // 4)
            else:
                end_train = num * 8 // 13
                end_val = num * 10 // 13
            assert (n_new_train == 0 and action in self.new_actions) or 0 < end_train < end_val
            assert 0 <= end_train <= end_val < num

            split_indices['train'].append(self.action_indices[action][:end_train])
            split_indices['val'].append(self.action_indices[action][end_train:end_val])
            split_indices['test'].append(self.action_indices[action][end_val:num])
            if action not in self.new_actions:
                split_indices_reg['train'].append(self.action_indices[action][:end_train])
                split_indices_reg['val'].append(self.action_indices[action][end_train:end_val])
                split_indices_reg['test'].append(self.action_indices[action][end_val:num])
            action_split_indices['train'][action] = self.action_indices[action][:end_train]
            action_split_indices['val'][action] = self.action_indices[action][end_train:end_val]
            action_split_indices['test'][action] = self.action_indices[action][end_val:]
            split_num = {'train': end_train, 'val': end_val - end_train, 'test': num - end_val}

            label = self.label_action(action)
            for mode in ['train', 'val', 'test']:
                action_cnt[mode][action] = split_num[mode]
                if label not in label_cnt[mode]:
                    label_cnt[mode][label] = 0
                label_cnt[mode][label] += split_num[mode]
                if action not in self.new_actions:
                    if label not in label_cnt_reg[mode]:
                        label_cnt_reg[mode][label] = 0
                    label_cnt_reg[mode][label] += split_num[mode]

        for mode in ['train', 'val', 'test']:
            split_indices[mode] = np.concatenate([np.array(x, dtype=np.int) for x in split_indices[mode]], axis=0)
            split_indices_reg[mode] = np.concatenate([np.array(x, dtype=np.int) for x in split_indices_reg[mode]],
                                                     axis=0)
            np.random.seed(233)
            np.random.shuffle(split_indices[mode])
            np.random.shuffle(split_indices_reg[mode])
        return split_indices, split_indices_reg, action_split_indices, action_cnt, label_cnt, label_cnt_reg

    def _get_action_label(self, labels):
        action_label = {}
        n_labels = len(labels['label_names'])
        for i in range(1, n_labels):
            for a in labels[i]:
                action_label[a] = i
        return action_label

    def _get_label2actions(self, sub_actions, labels):
        label2actions = [[] for i in range(len(labels))]
        label2actions_reg = [[] for i in range(len(labels))]
        for action in sub_actions:
            label = self.label_action(action)
            label2actions[label].append(action)
            if action not in self.new_actions:
                label2actions_reg[label].append(action)
        return label2actions, label2actions_reg

    def label_action(self, action):
        if action in self.action_label:
            return self.action_label[action]
        return 0

    def __init__(self, config):
        config = deepcopy(config)
        if isinstance(config['actions'], str):
            config['actions'] = fewshot_default_split[config['actions']]
        if isinstance(config['labels'], str):
            config['labels'] = fewshot_default_labels[config['labels']]
        self.config = config
        sub_config = self.complete_sub_config(config)

        self.action_label = self._get_action_label(config['labels'])

        filelist = self._get_filelist(config['toy'])
        self.dataset = RLBench(sub_config, filelist=filelist)

        self.labels = config['labels']['label_names']
        self.new_actions = self.config['actions']['new']
        self.sub_actions = self.dataset.get_action_list()
        self.reg_actions = [a for a in self.sub_actions if a not in self.new_actions]
        self.label2actions, self.label2actions_reg = self._get_label2actions(self.sub_actions, self.labels)
        self.action2index = self.dataset.action2index
        self.action_indices = self.dataset.action_indices
        self.split_indices, self.split_indices_reg, self.action_split_indices, \
        self.action_cnt, self.label_cnt, self.label_cnt_reg = self._split_data()

        self.equal_sample = {'train': False, 'val': False, 'test': False}
        self.only_regular = {'train': False, 'val': False, 'test': False}

        for mode in ['train', 'val', 'test']:
            for index in self.split_indices_reg[mode]:
                assert self.sub_actions[self.dataset.getitem_from_index(index, mode)['actions']] not in self.new_actions

    def get_action_list(self):
        return tuple(self.labels)

    def get_original_action_list(self):
        return tuple(self.sub_actions)

    def get_new_actions(self):
        return tuple(self.new_actions)

    def label_action_in_data(self, dat):
        action_id = dat['actions']
        label = self.label_action(self.sub_actions[action_id])
        dat['actions'] = label
        dat['original_actions'] = action_id
        assert 0 <= label < len(self.labels)
        return dat

    def getitem_from_index(self, index, mode):
        dat = self.dataset.getitem_from_index(index, mode)
        dat = self.label_action_in_data(dat)
        return dat

    def getitem_withmode(self, item, mode):
        index = None
        if self.equal_sample[mode]:
            while index is None:
                if self.only_regular[mode]:
                    label = sample_from_list([i for i in range(len(self.label2actions))])
                    action = sample_from_list([a for a in self.label2actions[label] if a not in self.new_actions])
                else:
                    label = sample_from_list([i for i in range(len(self.label2actions))])
                    action = sample_from_list([a for a in self.label2actions[label]])
                index = sample_from_list(self.action_split_indices[mode][action])
                if index is None:
                    # zero shot learning
                    assert self.config['n_new_train'] == 0 and action in self.new_actions and mode in ['train', 'val']
        else:
            if self.only_regular[mode]:
                index = self.split_indices_reg[mode][item]
            else:
                index = self.split_indices[mode][item]
        assert index is not None
        dat = self.dataset.getitem_from_index(index, mode)
        dat = self.label_action_in_data(dat)
        if self.equal_sample[mode]:
            weight = float(1)
        else:
            label = dat['actions']
            action = self.sub_actions[dat['original_actions']]
            tot = self.len_withmode(mode)
            n_labels = len(self.label2actions)
            n_actions = len(self.label2actions_reg[label]) if self.only_regular[mode] else len(
                self.label2actions[label])
            fre = self.action_cnt[mode][action]
            weight = float(tot / n_labels / n_actions / fre)
        dat['weight'] = weight
        return dat

    def len_withmode(self, mode):
        if self.only_regular[mode]:
            return self.split_indices_reg[mode].shape[0]
        return self.split_indices[mode].shape[0]


if __name__ == '__main__':
    config = RLBench.complete_config(ConfigUpdate({'toy': False, 'actions': 'default10'}))
    dataset = RLBench(config)
    print(dataset.get_action_list())
