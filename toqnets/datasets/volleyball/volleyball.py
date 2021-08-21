#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : volleyball.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import os
import random
from copy import deepcopy

import torch
from tqdm import tqdm

from toqnets.config_update import ConfigUpdate, update_config
from toqnets.datasets.utils import sample_from_list


class VolleyBall:
    VOLLEYBALL_DATA_PATH = 'data/volleyball'
    VIDEO_DATA_PATH = 'data/volleyball/video'
    ANNOTATION_SUBPATH = 'volleyball_tracking_annotation'

    VIDEO_ID_SPLIT = {
        'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
        'val': [2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
        'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
    }

    INDIVIDUAL_ACTIONS = ['blocking',
                          'waiting',
                          'setting',
                          'moving',
                          'standing',
                          'spiking',
                          'digging']

    def get_video_frame_activity(self):
        activity_label = {}
        for mode in ('train', 'val', 'test'):
            for video_id in self.VIDEO_ID_SPLIT[mode]:
                activity_label[video_id] = {}
                annotation_fin = open(os.path.join(self.VIDEO_DATA_PATH, str(video_id), 'annotations.txt'))
                for line in annotation_fin:
                    words = line.strip().split(' ')
                    for i, w in enumerate(words):
                        if w.endswith('.jpg'):
                            frame_id = int(w.split('.')[0])
                            activity = words[i + 1]
                            activity = activity.replace('-', '_')
                            activity_label[video_id][frame_id] = activity
                annotation_fin.close()
        return activity_label

    def get_states_from_file(self, video_id, frame_id):
        filename = os.path.join(self.VOLLEYBALL_DATA_PATH, self.ANNOTATION_SUBPATH, str(video_id), str(frame_id),
                                '%d.txt' % frame_id)
        try:
            fin = open(filename, 'r')
        except Exception:
            return None
        records = []
        max_fid = -1e9
        min_fid = 1e9
        for line in fin:
            words = line.strip().split(' ')
            for i in range(9):
                words[i] = int(words[i])
            records.append(words)
            fid = words[5]
            min_fid = min(min_fid, fid)
            max_fid = max(max_fid, fid)
        length = max_fid - min_fid + 1
        assert length == 20
        states = torch.zeros(length, self.config['n_agents'], self.config['state_dim'])
        states[:, :, 4] = 1  # lost at the beginning
        for record in records:
            k = record[5] - min_fid
            pid = record[0]
            xmin, ymin, xmax, ymax = tuple(record[1:5])
            xctr, yctr = (xmin + xmax) / 2, (ymin + ymax) / 2
            xsiz, ysiz = (xmax - xmin) / 2, (ymax - ymin) / 2
            lost = record[6]
            grouping = record[7]
            ia = record[9]
            onehot_ia = [1 if ia == self.INDIVIDUAL_ACTIONS[i] else 0 for i in range(len(self.INDIVIDUAL_ACTIONS))]
            states[k, pid, :] = torch.FloatTensor([xctr, yctr, xsiz, ysiz, lost, grouping] + onehot_ia + [0])
        fin.close()
        return states

    def get_framd_ids(self, video_id):
        folder = os.path.join(self.VOLLEYBALL_DATA_PATH, self.ANNOTATION_SUBPATH, str(video_id))
        frame_ids = os.listdir(folder)
        frame_ids = list(filter(lambda x: x.isdigit(), frame_ids))
        frame_ids = list(map(lambda x: int(x), frame_ids))
        return frame_ids

    default_config = {
        'name': 'VolleyBall',
        'toy': False,
        'length': 20,
        'n_agents': 13,
        'state_dim': 14,
        'temporal': 'exact',
        'actions': [
            'r_set',
            'r_spike',
            'r_pass',
            'r_winpoint',
            'l_winpoint',
            'l_pass',
            'l_spike',
            'l_set',
        ],
        'subsample': 1,
    }

    def set_equal_sample(self, mode, val):
        self.equal_sample[mode] = val

    def set_only_regular(self, mode, val):
        pass

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
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
        self.equal_sample = {'train': False, 'val': False, 'test': False}

    def fetch(self):
        rng = random.Random(2333)

        library = {}
        indices = {}
        for mode in ['train', 'val', 'test']:
            library[mode] = [[] for a in self.actions]
            indices[mode] = []

        activity_label = self.get_video_frame_activity()

        progress_bar = tqdm(total=4678)

        for mode in ['train', 'val', 'test']:
            for video_id in self.VIDEO_ID_SPLIT[mode]:
                frame_ids = self.get_framd_ids(video_id)
                if self.config['toy']:
                    frame_ids = frame_ids[:10]
                for frame_id in frame_ids:
                    states = self.get_states_from_file(video_id, frame_id)
                    progress_bar.update(1)
                    if states is None:
                        continue
                    if frame_id not in activity_label[video_id]:
                        continue
                    aid = self.action2index[activity_label[video_id][frame_id]]
                    library[mode][aid].append(states)
                    indices[mode].append((aid, len(library[mode][aid]) - 1))
            rng.shuffle(indices[mode])
            for aid in range(len(self.actions)):
                library[mode][aid] = torch.stack(library[mode][aid], dim=0)
        return library, indices

    def get_action_list(self):
        return tuple(self.actions)

    def get_item_aid_sid(self, aid, sid, mode, library=None):
        if library is None:
            library = self.library
        states = library[mode][aid][sid]
        length = states.size(0)
        if self.config['temporal'] == 'all':
            duration = length * 2 // 3
            start_time = random.randint(0, length - duration)
            middle = length / 2 - start_time
            states = states[start_time:start_time + duration]
            length = duration
        elif self.config['temporal'] == 'exact':
            middle = length / 2
        else:
            raise ValueError()
        if self.config['subsample'] != 1:
            stride = self.config['subsample']
            new_length = (length - 1) // stride + 1
            offset = random.randint(0, length - (new_length - 1) * stride - 1)
            assert offset + stride * (new_length - 1) < length
            states = states[offset:offset + stride * new_length:stride]
        return {
            'states': states,
            'actions': aid,
            'sample_ids': sid,
            'middle': middle
        }

    def getitem_withmode(self, item, mode):
        if self.equal_sample[mode]:
            aid = sample_from_list([i for i in range(len(self.actions))])
            sid = random.randint(0, self.library[mode][aid].size(0) - 1)
            weight = 1.0
        else:
            aid = self.indices[mode][item][0]
            sid = self.indices[mode][item][1]
            total = self.len_withmode(mode)
            n_actions = len(self.actions)
            fre = self.library[mode][aid].size(0)
            weight = total / n_actions / fre

        dat = self.get_item_aid_sid(aid, sid, mode)
        dat['weight'] = weight
        return dat

    def len_withmode(self, mode):
        return len(self.indices[mode])


# class LongVideoNvN_Wrapper_FewShot_Softmax:
#     default_config = {
#         'name': 'LongVideoNvN_Wrapper_FewShot_Softmax',
#         'equal_sample': {'train': False, 'val': False, 'test': False},
#         'only_regular': {'train': False, 'val': False, 'test': False},
#         'new_actions': [
#             'trap', 'sliding'
#         ],
#         'n_few_shot': 50,
#         'exact_length': None,
#     }
#
#     @classmethod
#     def complete_config(cls, config_update, default_config=None):
#         config = deepcopy(cls.default_config) if default_config is None else default_config
#         update_config(config, config_update)
#         return config
#
#     def complete_sub_config(self, config):
#         sub_args = {}
#         if config['exact_length'] is not None:
#             sub_args['exact_length'] = config['exact_length']
#         sub_config = deepcopy(LongVideoNvN.complete_config(ConfigUpdate(sub_args)))
#         return sub_config
#
#     def set_equal_sample(self, mode, val):
#         self.config['equal_sample'][mode] = val
#
#     def set_only_regular(self, mode, val):
#         self.config['only_regular'][mode] = val
#
#     def fetch_from_sub(self, library):
#         indices = {'train': [], 'val': [], 'test': []}
#         n_few_shot = self.config['n_few_shot']
#         new_library = {'train': [], 'val': [], 'test': []}
#         for aid, action in enumerate(self.get_action_list()):
#             if action in self.config['new_actions']:
#                 tra = torch.cat((library['train'][aid][0], library['val'][aid][0], library['test'][aid][0]),
#                                 dim=0)
#                 pid = torch.cat((library['train'][aid][1], library['val'][aid][1], library['test'][aid][1]),
#                                 dim=0)
#                 new_library['train'].append((tra[:n_few_shot], pid[:n_few_shot]))
#                 new_library['val'].append((tra[n_few_shot:n_few_shot * 2], pid[n_few_shot:n_few_shot * 2]))
#                 new_library['test'].append((tra[n_few_shot * 2:], pid[n_few_shot * 2:]))
#             else:
#                 for split in ('train', 'val', 'test'):
#                     new_library[split].append(library[split][aid])
#             for split in ('train', 'val', 'test'):
#                 n_data = new_library[split][aid][0].shape[0]
#                 index = np.zeros((n_data, 2), dtype=np.int)
#                 index[:, 0] = aid
#                 index[:, 1] = np.arange(n_data)
#                 indices[split].append(index)
#                 print('%s %s %d' % (action, split, n_data))
#         for split in ('train', 'val', 'test'):
#             indices[split] = np.concatenate(indices[split], axis=0)
#         return new_library, indices
#
#     def __init__(self, config):
#         self.config = config
#         sub_config = self.complete_sub_config(config)
#         self.dataset = LongVideoNvN(sub_config)
#         self.library, self.indices = self.fetch_from_sub(self.dataset.library)
#
#         self.reg_indices = {}
#         actions = self.dataset.config['actions']
#         self.reg_action_onehot = [False if actions[i] in self.config['new_actions'] else True
#                                   for i in range(len(actions))]
#         self.reg_action_indices = [i for i in range(len(actions)) if self.reg_action_onehot[i]]
#         self.all_action_indices = [i for i in range(len(actions))]
#
#         for mode in ['train', 'val', 'test']:
#             self.reg_indices[mode] = np.zeros(self.indices[mode].shape, dtype=np.int)
#             j = 0
#
#             for i in range(self.indices[mode].shape[0]):
#                 if self.reg_action_onehot[self.indices[mode][i][0]]:
#                     self.reg_indices[mode][j] = self.indices[mode][i]
#                     j += 1
#             self.reg_indices[mode] = self.reg_indices[mode][:j]
#
#     def get_action_list(self):
#         return self.dataset.get_action_list()
#
#     def get_new_actions(self):
#         return [x for x in deepcopy(self.config['new_actions'])]
#
#     def getitem_withmode(self, item, mode):
#         aid, sid = None, None
#         if self.config['equal_sample'][mode]:
#             if self.config['only_regular'][mode]:
#                 aid = sample_from_list(self.reg_action_indices)
#             else:
#                 aid = sample_from_list(self.all_action_indices)
#             sid = random.randint(0, self.library[mode][aid][0].shape[0] - 1)
#             # while True:
#             #     it = random.randint(0, self.len_withmode(mode) - 1)
#             #     if self.indices[mode][it][0] == aid:
#             #         sid = self.indices[mode][it][1]
#             #         break
#         else:
#             if self.config['only_regular'][mode]:
#                 aid = self.reg_indices[mode][item][0]
#                 sid = self.reg_indices[mode][item][1]
#             else:
#                 aid = self.indices[mode][item][0]
#                 sid = self.indices[mode][item][1]
#         # print(mode, item, aid, sid, self.config['equal_sample'][mode])
#         return self.dataset.get_item_aid_sid(aid, sid, mode, self.library)
#         # return {
#         #     'trajectories': self.library[mode][aid][0][sid],
#         #     'playerid': self.library[mode][aid][1][sid],
#         #     'actions': aid,
#         #     'types': self.dataset.types
#         # }
#
#     def len_withmode(self, mode):
#         if self.config['only_regular'][mode]:
#             return self.reg_indices[mode].shape[0]
#         return self.indices[mode].shape[0]


if __name__ == '__main__':
    config = VolleyBall.complete_config(ConfigUpdate({}))
    dataset = VolleyBall(config)
