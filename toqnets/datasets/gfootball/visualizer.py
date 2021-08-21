#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualizer.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import os

import matplotlib.pyplot as plt

from toqnets.utils import make_video
from toqnets.datasets.gfootball.LongVideoNvN import LongVideoNvN

class PlayGround:
    def __init__(self, title='', x_range=None, y_range=None, x_reverse=False, y_reverse=False):
        assert len(x_range) == 2
        assert len(y_range) == 2
        self.title = title
        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self.x_reverse = x_reverse
        self.y_reverse = y_reverse
        self.agents = []
        self.info = [('Agent', 'label', 'output')]

    def add_agent(self, x, y, name, color, size=20):
        x /= 60
        y /= 60
        if self.x_reverse:
            x = sum(self.x_range) - x
        if self.y_reverse:
            y = sum(self.y_range) - y
        self.agents.append((x, y, name, color, size))

    def add_info(self, name, ans, out):
        self.info.append((name, ans, out))

    def plot(self, save_file):
        plt.close()
        fig = plt.figure(figsize=(10.0, 8.4))
        ax = fig.add_subplot(2, 1, 1)
        axt = fig.add_subplot(2, 1, 2)
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        axt.get_xaxis().set_visible(False)
        axt.get_yaxis().set_visible(False)
        for x, y, name, color, size in self.agents:
            ax.scatter(x, y, s=size, c=color)
            ax.text(x, y, name)
        ax.set_title(self.title)

        axt.set_xlim([0, 1])
        axt.set_ylim([0, 1])

        for k in range(len(self.info)):
            # print(self.info[k])
            axt.text(0.05, 0.9 - k * 0.05, self.info[k][0])
            axt.text(0.15, 0.9 - k * 0.05, self.info[k][1])
            axt.text(0.25, 0.9 - k * 0.05, self.info[k][2],
                     color='black' if k == 0 else ('green' if self.info[k][1] == self.info[k][2] else 'red'))

        plt.tight_layout(pad=1)
        plt.savefig(save_file)
        plt.close()


def display(states, actions, outputs, save_dir):
    action_names = ['none', 'movement', 'ball_control', 'trap', 'short_pass', 'long_pass', 'high_pass', 'shot',
                    'deflect', 'interfere', 'trip', 'sliding']
    os.makedirs(save_dir, exist_ok=True)
    batch, length, n_agents, _ = states.size()
    n_players = n_agents // 2
    for b in range(batch):
        os.makedirs(os.path.join(save_dir, 'frames', str(b)), exist_ok=True)
        os.makedirs(os.path.join(save_dir, str(b)), exist_ok=True)
        for t in range(length):
            p = PlayGround('test', [-1.0, 1.0], [-0.42, 0.42], False, False)
            for k in range(n_agents):
                name = '' if k == 0 else ('A' + str(k) if k <= n_players else 'B' + str(k - n_players))
                color = 'orange' if k == 0 else ('blue' if k <= n_players else 'red')
                size = 30 if k == 0 else 20
                p.add_agent(states[b, t, k, 0], states[b, t, k, 1], name, color, size)
                if k > 0:
                    p.add_info(name, action_names[int(actions[b, t, k])], action_names[int(outputs[b, t, k])])
            p.plot(os.path.join(save_dir, 'frames', str(b), '%04d.png' % t))
        make_video(os.path.join(save_dir, 'frames', str(b)), os.path.join(save_dir, str(b), str(b)), 8)


def display_single(states, playerid, action, output, save_dir):
    print(states, playerid, action, output, save_dir)
    action_names = [x[0] for x in LongVideoNvN.default_config['actions']]
    os.makedirs(save_dir, exist_ok=True)
    batch, length, n_agents, _ = states.size()
    n_players = n_agents // 2
    for b in range(batch):
        os.makedirs(os.path.join(save_dir, 'frames', str(b)), exist_ok=True)
        os.makedirs(os.path.join(save_dir, str(b)), exist_ok=True)
        for t in range(length):
            p = PlayGround('test', [-1.0, 1.0], [-0.42, 0.42], False, False)
            for k in range(n_agents):
                name = '' if k == 0 else ('A' + str(k) if k <= n_players else 'B' + str(k - n_players))
                color = 'orange' if k == 0 else ('blue' if k <= n_players else 'red')
                size = 30 if k == 0 else 20
                p.add_agent(states[b, t, k, 0], states[b, t, k, 1], name, color, size)
                if k == playerid[b]:
                    p.add_info(name, action_names[action[b]], action_names[output[b]])
            p.plot(os.path.join(save_dir, 'frames', str(b), '%04d.png' % t))
        make_video(os.path.join(save_dir, 'frames', str(b)), os.path.join(save_dir, str(b), str(b)), 8)


def display_any(states, n_left_players, n_right_players, action, action_list, output, save_dir, out_filter=None,
                tar_filter=None, start_index=0):
    action_names = action_list
    os.makedirs(save_dir, exist_ok=True)
    batch, length, n_agents, _ = states.size()
    for b in range(batch):
        plot_index = start_index + b
        cur_tar = action_names[action[b]]
        cur_out = action_names[output[b]]
        if out_filter is not None and cur_out not in out_filter:
            continue
        if tar_filter is not None and cur_tar not in tar_filter:
            continue
        os.makedirs(os.path.join(save_dir, 'frames', str(plot_index)), exist_ok=True)
        os.makedirs(os.path.join(save_dir, str(plot_index)), exist_ok=True)
        for t in range(length):
            p = PlayGround('test', [-1.0, 1.0], [-0.42, 0.42], False, False)
            for k in range(n_agents):
                name = '' if k == 0 else ('A' + str(k) if k <= n_left_players else 'B' + str(k - n_left_players))
                color = 'orange' if k == 0 else ('blue' if k <= n_left_players else 'red')
                size = 30 if k == 0 else 20
                p.add_agent(states[b, t, k, 0], states[b, t, k, 1], name, color, size)
                if k == 1:
                    p.add_info(name, action_names[action[b]], action_names[output[b]])
            p.plot(os.path.join(save_dir, 'frames', str(plot_index), '%04d.png' % t))
        make_video(os.path.join(save_dir, 'frames', str(plot_index)),
                   os.path.join(save_dir, str(plot_index), str(plot_index)), 8)
