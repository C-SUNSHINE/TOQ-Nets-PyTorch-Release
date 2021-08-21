#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualizer.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import json
import os

import cv2

from toqnets.utils import make_video

DATA_PATH = "/data/vision/billf/scratch/lzz/ActionGrounding/toyota_dataset"


def read_video(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    fps = cap.get(5)
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
    return video, fps


def annotate_frame(frame, pose2d):
    assert len(pose2d) == 26
    edges = [(0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11), (4, 5), (4, 10), (5, 11), (10, 11),
             (11, 12), (10, 12)]
    for i in range(13):
        x = round(pose2d[i])
        y = round(pose2d[i + 13])
        cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
        # cv2.putText(frame, str(i), (x + 3, y + 3), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
        # cv2.putText(image, i['category'], (int(x1) + 5, int(y1) + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
    for (i, j) in edges:
        x1 = round(pose2d[i])
        y1 = round(pose2d[i + 13])
        x2 = round(pose2d[j])
        y2 = round(pose2d[j + 13])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def annotate_video(id, save_dir='dumps/toyota_skeleton_visualize', frame_list=None):
    with open(os.path.join(DATA_PATH, 'json', id + '.json'), 'r') as fin:
        labels = json.load(fin)['frames']
    frames, fps = read_video(os.path.join(DATA_PATH, 'mp4', id + '.mp4'))
    assert len(frames) == len(labels)
    n = len(frames)
    save_path = os.path.join(save_dir, 'frames', id)
    for i in (range(n) if frame_list is None else frame_list):
        label = labels[i]
        for x in label:
            if x['cumscore'] > 1:
                annotate_frame(frames[i], x['pose2d'])
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, '%04d.png' % i), frames[i])
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(os.path.join(save_dir, id + '.avi'), fourcc, round(fps), frames[0].shape[:2])
    # for frame in frames:
    #     out.write(frame)
    # out.release()
    make_video(save_path, os.path.join(save_dir, id), fps)


if __name__ == '__main__':
    data_ids = ['WatchTV_p19_r01_v02_c05', 'Usetelephone_p04_r00_v04_c04', 'Drink.Frombottle_p10_r01_v05_c05',
                'Getup_p06_r02_v09_c02', 'Enter_p03_r01_v18_c06', 'Walk_p18_r16_v05_c05',
                'Drink.Fromcup_p15_r00_v01_c04', 'Usetelephone_p02_r01_v01_c07', 'Walk_p16_r16_v14_c06',
                'Leave_p10_r02_v13_c07', 'Cook.Cleandishes_p25_r00_v16_c03', 'Leave_p06_r01_v12_c03',
                'Walk_p03_r02_v12_c01', 'Uselaptop_p10_r10_v10_c01', 'Drink.Fromcup_p19_r00_v02_c04',
                'Walk_p10_r08_v12_c07', 'Walk_p14_r05_v02_c05', 'Enter_p10_r02_v13_c03', 'Readbook_p15_r01_v02_c04',
                'Cook.Stir_p13_r05_v23_c06', 'Cook.Cleanup_p13_r02_v20_c06', 'Cook.Cleandishes_p20_r01_v13_c03',
                'Sitdown_p17_r04_v09_c02', 'Walk_p14_r00_v14_c06', 'Drink.Fromcup_p10_r02_v02_c04',
                'Drink.Fromglass_p03_r07_v18_c03', 'Walk_p03_r08_v13_c01', 'Cook.Cleanup_p03_r01_v18_c07',
                'Drink.Fromcup_p16_r03_v06_c01', 'Walk_p15_r09_v02_c05']
    for id in data_ids:
        annotate_video(id)
