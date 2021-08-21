#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : toyota.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import json
import os
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
from tqdm import tqdm

from toqnets.config_update import ConfigUpdate, update_config
from toqnets.datasets.utils import read_video, sample_from_list

DATA_PATH = "data/toyota_dataset"
CACHE_PATH = 'data/toyota_dataset'
CACHE_PATH_CLIPS = 'data/toyota_dataset_10_clips'


class ToyotaSmartHome:
    default_config = {
        'name': 'ToyotaSmartHome',
        'toy': False,
        'length': 30,
        'n_frames': 8,
        'n_clips': 10,
        'njts': 13,
        'image_size': (224, 224),
        'with_video': False,
        'truncate': 'clip',
        'subsample': 1,
        'version': 1,
    }

    default_actions = ['Drink', 'Walk', 'Sitdown', 'Eat', 'Cook', 'Usetelephone', 'WatchTV', 'Takepills', 'Pour',
                       'Readbook', 'Leave', 'Enter', 'Laydown', 'Getup', 'Uselaptop', 'Makecoffee', 'Usetablet',
                       'Maketea', 'Cutbread']

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        for k in cls.default_config.keys():
            if k not in config:
                config[k] = deepcopy(cls.default_config[k])
        return config

    @classmethod
    def get_filelist(cls, toy=False):
        filelist = os.listdir(os.path.join(DATA_PATH, 'json'))
        filelist = [x[:-5] for x in filelist if x.endswith('.json')]
        if toy:
            random.seed(233)
            random.shuffle(filelist)
            filelist = filelist[:1024]
        return filelist

    def set_equal_sample(self, mode, val):
        self.equal_sample[mode] = val

    def set_only_regular(self, mode, val):
        pass

    def set_clip_sample(self, mode, val):
        self._clip_sample[mode] = val

    def __init__(self, config):
        self.config = config
        self.info = []
        self.split_indices = {}
        self.data = []
        self.actions, self.action2index = self._get_actions()
        self.action_cnt = {}
        self.action_indices = {}
        self.fetch()
        self._clip_sample = {'train': 'one', 'val': 'one', 'test': 'one'}
        self.equal_sample = {'train': False, 'val': False, 'test': False}

    def _get_actions(self):
        actions = tuple(self.default_actions)
        action2index = {}
        for i, action in enumerate(actions):
            action2index[action] = i
        return actions, action2index

    def _preprocess_dat(self, dat, dat_info):
        std_njts = self.config['njts']
        length = self.config['length']
        assert dat['njts'] == std_njts
        frames = []
        scores = []
        frame_cnt = 0
        for k in dat_info['frame_list']:
            frame = None
            score = 1
            for y in dat['frames'][k]:
                if frame is None or y['cumscore'] > score:
                    frame = y
                    score = frame['cumscore']
            frames.append(frame)
            if frame is not None:
                frame_cnt += 1
        if self.config['version'] >= 2 and frame_cnt < 0.5 * len(frames):
            return None
        for k in range(1, len(frames)):
            if frames[k] is None:
                frames[k] = frames[k - 1]
        for k in range(len(frames) - 1, 0, -1):
            if frames[k - 1] is None:
                frames[k - 1] = frames[k]
        if None in frames:
            for k in range(len(frames)):
                frames[k] = {'pose3d': [0.0] * 39}
        # Make length to be length
        if len(frames) >= length:
            if self.config['truncate'] == 'subsample':
                stride = len(frames) // length
                offset = random.randint(0, len(frames) - length * stride)
                frames = frames[offset:offset + stride * length:stride]
            elif self.config['truncate'] == 'clip':
                stride = 1
                offset = random.randint(0, len(frames) - length * stride)
                frames = frames[offset:offset + stride * length:stride]
            else:
                raise ValueError()
        else:
            frames += frames[-1:] * (length - len(frames))
        assert len(frames) == length
        if self.config['subsample'] != 1:
            stride = self.config['subsample']
            new_length = (len(frames) + stride - 1) // stride
            offset = random.randint(0, len(frames) - new_length * stride)
            frames = frames[offset:offset + stride * new_length:stride]
            assert len(frames) == new_length
        dat['frames'] = frames
        return dat

    def _sample_and_read_video_frames(self, filename, frame_lists, clip_sample='one'):
        if clip_sample == 'one':
            mp4_frames_dir = os.path.join(CACHE_PATH, 'mp4_frames', filename)
        else:
            mp4_frames_dir = os.path.join(CACHE_PATH_CLIPS, 'mp4_frames', filename)
        if clip_sample == 'all':
            sample_frame_lists = frame_lists
        elif clip_sample == 'one':
            sample_frame_lists = frame_lists[0:1]
        else:
            raise ValueError()
        n_frames = self.config['n_frames']
        frames = []
        for frame_list in sample_frame_lists:
            if len(frame_list) >= n_frames:
                stride = len(frame_list) // n_frames
                offset = random.randint(0, len(frame_list) - n_frames * stride)
                new_frame_list = frame_list[offset:offset + stride * n_frames:stride]
            else:
                new_frame_list = frame_list + frame_list[-1:] * (n_frames - len(frame_list))
            for k in new_frame_list:
                frames.append(cv2.imread(os.path.join(mp4_frames_dir, '%s_%d.png' % (filename, k))))
                # print(os.path.join(mp4_frames_dir, '%s_%d.png' % (filename, k)), (frames[-1] is None))
                if frames[-1] is None:
                    exit()
        return np.stack(frames)

    def _get_data_from_file(self, filename):
        skeleton_filename = os.path.join(DATA_PATH, 'json', filename + '.json')
        mp4_frames_dir = os.path.join(CACHE_PATH, 'mp4_frames', filename)
        with open(skeleton_filename, 'r') as fin:
            dat = json.load(fin)
        with open(os.path.join(mp4_frames_dir, 'info.json'), 'r') as fin:
            dat_info = json.load(fin)
        dat = self._preprocess_dat(dat, dat_info)
        if dat is None:
            return dat
        dat['frame_list'] = dat_info['frame_list']
        return dat

    def _get_clips_frame_lists(self, filename):
        mp4_frames_dir = os.path.join(CACHE_PATH_CLIPS, 'mp4_frames', filename)
        info_filename = os.path.join(mp4_frames_dir, 'info.json')
        while True:
            if os.path.isfile(info_filename):
                with open(info_filename) as fin:
                    dat_info = json.load(fin)
                if len(dat_info['frame_lists']) == self.config['n_clips']:
                    return dat_info['frame_lists']
            cache_video(filename, n_clips=self.config['n_clips'])

    def _get_coords_from_data(self, dat):
        coords = []
        for f in dat['frames']:
            coords += f['pose3d']
        coords = np.stack(coords)
        return torch.from_numpy(coords).float().view(len(dat['frames']), 3, self.config['njts']).permute(0, 2, 1)

    def get_action_list(self):
        return tuple(self.actions)

    def action_to_index(self, action):
        return self.action2index[action]

    def fetch(self):
        filelist = self.get_filelist(self.config['toy'])
        action_list = self.get_action_list()
        self.info = []
        self.data = []
        loading_bar = tqdm(filelist)
        loading_bar.set_description('Loading skeleton json\'s')
        self.coords = torch.zeros(len(loading_bar), (self.config['length'] - 1) // self.config['subsample'] + 1,
                                  self.config['njts'], 3)
        drop_cnt = 0
        for index, filename in enumerate(loading_bar):
            action = filename.split('_')[0].split('.')[0]
            if action in action_list:
                inf = {
                    'action': action,
                    'filename': filename,
                }
                self.info.append(inf)
                dat = self._get_data_from_file(filename)
                self.data.append(dat)
                if self.config['version'] == 1:
                    assert dat is not None
                if dat is not None:
                    self.coords[index] = self._get_coords_from_data(dat)
                else:
                    drop_cnt += 1
        print("Dropped %d/%d" % (drop_cnt, len(filelist)))
        action_all_indices = {}
        for index, inf in enumerate(self.info):
            if inf['action'] not in action_all_indices:
                action_all_indices[inf['action']] = []
            if self.data[index] is not None:
                action_all_indices[inf['action']].append(index)

        if self.config['version'] == 2:
            pass
            # for action in action_all_indices:
            #     print(action, len(action_all_indices[action]))
            # exit()

        self.action_indices = {
            'train': {}, 'val': {}, 'test': {}
        }
        self.split_indices = {
            'train': [], 'val': [], 'test': [],
        }
        self.action_cnt = {
            'train': {}, 'val': {}, 'test': {}
        }
        for action in action_all_indices:
            action_all_indices[action] = np.array(action_all_indices[action])
            np.random.seed(2333)
            np.random.shuffle(action_all_indices[action])
            num = action_all_indices[action].shape[0]
            end_train = num * 8 // 13
            end_val = num * 10 // 13
            self.action_indices['train'][action] = action_all_indices[action][:end_train]
            self.action_indices['val'][action] = action_all_indices[action][end_train:end_val]
            self.action_indices['test'][action] = action_all_indices[action][end_val:]
            for mode in ['train', 'val', 'test']:
                self.split_indices[mode].append(self.action_indices[mode][action])
            self.action_cnt['train'][action] = end_train
            self.action_cnt['val'][action] = end_val - end_train
            self.action_cnt['test'][action] = num - end_val

        for mode in ['train', 'val', 'test']:
            self.split_indices[mode] = np.concatenate([np.array(x, dtype=np.int) for x in self.split_indices[mode]],
                                                      axis=0)
            np.random.seed(233)
            np.random.shuffle(self.split_indices[mode])

    def getitem_from_index(self, index, mode, option=None):
        action = self.info[index]['action']
        filename = self.info[index]['filename']
        if option == 'filename':
            return filename
        frame_list = self.data[index]['frame_list']
        coords = self.coords[index]
        res = {
            'sample_ids': index,
            'index': index,
            'trajectories': coords,
            'actions': self.action2index[action]
        }
        if self.config['with_video']:
            if self._clip_sample[mode] == 'all':
                frame_lists = self._get_clips_frame_lists(filename)
            else:
                frame_lists = [frame_list]
            res['video'] = torch.from_numpy(
                self._sample_and_read_video_frames(filename, frame_lists, clip_sample=self._clip_sample[mode])
            ).float().permute(0, 3, 1, 2).div(255)
        return res

    def getitem_withmode(self, item, mode):
        if self.equal_sample[mode]:
            action = sample_from_list(self.actions)
            index = sample_from_list(self.action_indices[mode][action])
            dat = self.getitem_from_index(index, mode)
            weight = 1.0
        else:
            index = self.split_indices[mode][item]
            dat = self.getitem_from_index(index, mode)
            fre = self.action_cnt[mode][self.actions[dat['actions']]]
            tot = self.len_withmode(mode)
            n_actions = len(self.actions)
            weight = tot / n_actions / fre

        dat['weight'] = weight
        return dat

    def len_withmode(self, mode):
        return self.split_indices[mode].shape[0]

    def test(self):
        count = {}
        for i in range(self.len_withmode('train')):
            value = self.getitem_withmode(i, 'train')['actions']
            value = self.actions[value]
            if value not in count:
                count[value] = 0
            count[value] += 1
        keys = count.keys()
        keys = sorted(keys)
        for k in keys:
            print(k, " :", count[k])


class ToyotaSmartHome_Wrapper_FewShot_Softmax:
    default_config = {
        'name': 'ToyotaSmartHome_Wrapper_FewShot_Softmax',
        'equal_sample': {'train': False, 'val': False, 'test': False},
        'only_regular': {'train': False, 'val': False, 'test': False},
        'toy': False,
        'new_actions': [
            'Sitdown', 'Takepills', 'Uselaptop'
        ],
        'new_n_train': 25,
        'with_video': False,
        'n_frames': 8,
        'truncate': 'subsample',
        'version': 1,
    }

    @classmethod
    def complete_config(cls, config_update, default_config=None):
        config = deepcopy(cls.default_config) if default_config is None else default_config
        update_config(config, config_update)
        return config

    def complete_sub_config(self, config):
        sub_config = deepcopy(ToyotaSmartHome.complete_config(ConfigUpdate({
            'toy': config['toy'],
            'with_video': config['with_video'],
            'n_frames': config['n_frames'],
            'truncate': config['truncate'] if 'truncate' in config else 'subsample',
            'version': config['version']
        })))
        return sub_config

    def set_clip_sample(self, mode, val):
        self.dataset.set_clip_sample(mode, val)

    def set_equal_sample(self, mode, val):
        self.config['equal_sample'][mode] = val

    def set_only_regular(self, mode, val):
        self.config['only_regular'][mode] = val

    def __init__(self, config):
        self.config = config
        sub_config = self.complete_sub_config(config)
        self.dataset = ToyotaSmartHome(sub_config)
        self.info = self.dataset.info
        self.data = self.dataset.data
        self.actions = self.dataset.get_action_list()
        self.new_actions = self.config['new_actions']
        self.reg_actions = [x for x in self.actions if x not in self.new_actions]
        action_indices = {}
        for index, inf in enumerate(self.info):
            if inf['action'] not in action_indices:
                action_indices[inf['action']] = []
            if self.data[index] is not None:
                action_indices[inf['action']].append(index)
        self.split_indices = {
            'train': [],
            'val': [],
            'test': [],
        }
        self.split_indices_reg = {
            'train': [],
            'val': [],
            'test': [],
        }
        self.action_split_indices = {
            'train': {},
            'val': {},
            'test': {},
        }
        self.action_cnt = {
            'train': {},
            'val': {},
            'test': {}
        }
        for action in action_indices:
            action_indices[action] = np.array(action_indices[action])
            np.random.seed(233)
            np.random.shuffle(action_indices[action])
            num = action_indices[action].shape[0]
            if action in self.new_actions:
                end_train = self.config['new_n_train']
                end_val = end_train + max((num - end_train) * 1 // 4, 1)
            else:
                end_train = num * 8 // 13
                end_val = num * 10 // 13
            self.split_indices['train'].append(action_indices[action][:end_train])
            self.split_indices['val'].append(action_indices[action][end_train:end_val])
            self.split_indices['test'].append(action_indices[action][end_val:])
            if action not in self.new_actions:
                self.split_indices_reg['train'].append(action_indices[action][:end_train])
                self.split_indices_reg['val'].append(action_indices[action][end_train:end_val])
                self.split_indices_reg['test'].append(action_indices[action][end_val:])
            self.action_split_indices['train'][action] = action_indices[action][:end_train]
            self.action_split_indices['val'][action] = action_indices[action][end_train:end_val]
            self.action_split_indices['test'][action] = action_indices[action][end_val:]
            self.action_cnt['train'][action] = end_train
            self.action_cnt['val'][action] = end_val - end_train
            self.action_cnt['test'][action] = num - end_val

        for mode in ['train', 'val', 'test']:
            self.split_indices[mode] = np.concatenate([np.array(x, dtype=np.int) for x in self.split_indices[mode]],
                                                      axis=0)
            self.split_indices_reg[mode] = np.concatenate(
                [np.array(x, dtype=np.int) for x in self.split_indices_reg[mode]],
                axis=0
            )
            np.random.seed(233)
            np.random.shuffle(self.split_indices[mode])
            np.random.shuffle(self.split_indices_reg[mode])
        self._action_stat = {}
        self._index_stat = {}

    def get_action_list(self):
        return self.dataset.get_action_list()

    def get_new_actions(self):
        return tuple(self.config['new_actions'])

    def getitem_from_index(self, index, mode, option=None):
        return self.dataset.getitem_from_index(index, mode, option=option)

    def getitem_withmode(self, item, mode):
        index = None
        if self.config['equal_sample'][mode]:
            if self.config['only_regular'][mode]:
                action = sample_from_list(self.reg_actions)
            else:
                action = sample_from_list(self.actions)
            index = sample_from_list(self.action_split_indices[mode][action])
        else:
            if self.config['only_regular'][mode]:
                index = self.split_indices_reg[mode][item]
            else:
                index = self.split_indices[mode][item]
        assert index is not None
        dat = self.dataset.getitem_from_index(index, mode)
        if self.config['equal_sample'][mode]:
            weight = float(1)
        else:
            fre = self.action_cnt[mode][self.actions[dat['actions']]]
            tot = self.len_withmode(mode)
            n_actions = len(self.actions) - len(self.new_actions) if self.config['only_regular'][mode] else len(
                self.actions)
            weight = float(tot / n_actions / fre)
        dat['weight'] = weight
        return dat

    def len_withmode(self, mode):
        if self.config['only_regular'][mode]:
            return self.split_indices_reg[mode].shape[0]
        return self.split_indices[mode].shape[0]


def cache_video(filename, image_size=(224, 224), max_length=64, n_clips=None):
    mod = 999983
    rng = np.random.default_rng((hash(filename) % mod + mod) % mod)
    cache_path = CACHE_PATH if n_clips is None else CACHE_PATH_CLIPS
    video_filename = os.path.join(DATA_PATH, 'mp4', filename + '.mp4')
    mp4_frames_dir = os.path.join(cache_path, 'mp4_frames', filename)
    os.makedirs(mp4_frames_dir, exist_ok=True)
    frames = read_video(video_filename)
    frames = [cv2.resize(f, dsize=tuple(image_size)) for f in frames]
    frame_lists = []
    for k in range(n_clips if n_clips is not None else 1):
        if len(frames) > max_length:
            offset = rng.integers(0, len(frames) - max_length)
            offset = int(offset)
            frame_list = [offset + i for i in range(max_length)]
        else:
            frame_list = [i for i in range(len(frames))]
        frame_lists.append(frame_list)
    if n_clips is None:
        info = {
            'filename': filename,
            'frame_list': frame_lists[0],
            'image_size': image_size
        }
    else:
        info = {
            'filename': filename,
            'frame_lists': frame_lists,
            'image_size': image_size
        }
    dump_frames = set([y for x in frame_lists for y in x])
    print(filename, 'n_clips=%s' % str(n_clips), len(dump_frames))
    for k in dump_frames:
        cv2.imwrite(os.path.join(mp4_frames_dir, '%s_%d.png' % (filename, k)), frames[k])
    with open(os.path.join(mp4_frames_dir, 'info.json'), 'w') as fout:
        json.dump(info, fout)
    return len(dump_frames)


def unroll_videos(toy=False):
    filelist = ToyotaSmartHome.get_filelist(toy)
    tot = 0
    for k, filename in enumerate(filelist):
        tot += cache_video(filename, image_size=(224, 224), max_length=64)
        print("%d/%d: %.2lf MB used" % (k + 1, len(filelist), tot * 0.097))


if __name__ == '__main__':
    # unroll_videos(True)
    # exit()
    config_update = ConfigUpdate({
        'toy': False,
        'with_video': False
    })
    dataset = ToyotaSmartHome(ToyotaSmartHome.complete_config(config_update))
    print(dataset.get_action_list())
    exit()
    for mode in ['train', 'val', 'test']:
        print(mode, dataset.len_withmode(mode))
    dataset.test()
    from torch.utils.data.dataloader import DataLoader
    from toqnets.datasets.utils import DatasetModeWrapper

    datasetmode = DatasetModeWrapper(dataset, 'train')
    loader = DataLoader(datasetmode, batch_size=16, shuffle=True, num_workers=10)
    for data in tqdm(loader):
        print(data['video'].size())
