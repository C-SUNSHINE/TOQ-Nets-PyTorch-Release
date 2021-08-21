#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import random
from threading import Lock

import cv2

sample_from_list_lock = Lock()


def sample_from_list(lis):
    if len(lis) == 0:
        return None
    sample_from_list_lock.acquire(blocking=True)
    s = random.randint(0, len(lis) - 1)
    sample_from_list_lock.release()
    return lis[s]


def update_dict(d1, d2):
    assert isinstance(d1, dict) and isinstance(d2, dict)
    for k in d2.keys():
        if k not in d1 or not isinstance(d1[k], dict) or not isinstance(d2[k], dict):
            d1[k] = d2[k]
        else:
            update_dict(d1[k], d2[k])


class DatasetModeWrapper:
    def __init__(self, dataset, mode):
        assert hasattr(dataset, 'getitem_withmode')
        assert hasattr(dataset, 'len_withmode')
        assert mode in ['train', 'val', 'test']
        self._dataset = dataset
        self._mode = mode

    def __getitem__(self, item):
        return self._dataset.getitem_withmode(item, self._mode)

    def __len__(self):
        return self._dataset.len_withmode(self._mode)

    def __getattr__(self, item):
        return getattr(self._dataset, item)


def read_video(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise Exception('Cannot open {}'.format(path))
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    return video
