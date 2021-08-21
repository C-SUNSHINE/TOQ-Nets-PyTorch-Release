#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/26/2019
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import os
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import torch


def write_text_to_file(message, fullname):
    with open(fullname, 'w') as fin:
        fin.write(message)


def make_video(frames_dir, output_path, frequency=10):
    command = "ffmpeg -framerate %d -f image2 -pattern_type glob -i \"%s\" -pix_fmt yuv420p -vcodec libx264 %s -y >ffmpeg_log.txt" % (
        frequency, frames_dir + '/*.png',
        output_path + '.mp4')
    print(command)
    call(command, shell=True, stdout=open(os.devnull, 'wb'),
         stderr=open(os.devnull, 'wb'))
    print("Generated Video %s" % (output_path + '.mp4'))


def average_off_none(a, default=None):
    a_off_none = [x for x in a if x is not None]
    if len(a_off_none) == 0:
        return default
    else:
        return sum(a_off_none) / len(a_off_none)


def plot_confusing_matrix(action_list, matrix, savefile, figsize=(16, 16)):
    print("Plot confusing matrix to ", savefile)

    matrix_raw = False

    for i in range(matrix.shape[0]):
        if matrix[i].sum() > 1.5:
            matrix_raw = True
    N = 20
    if matrix_raw:
        if matrix.shape[0] == 176 and len(action_list) == 174:
            action_list = action_list + ['extra_1', 'extra_2']
        assert matrix.shape[0] == len(action_list)
        frequency = []
        for i in range(matrix.shape[0]):
            frequency.append((i, matrix[i].sum()))
        frequency = list(reversed(sorted(frequency, key=lambda x: x[1])))
        new_action_list = []
        new_matrix = np.zeros((min(len(action_list), N), min(len(action_list), N)))
        for i in range(min(len(action_list), N)):
            if i == N - 1 and len(action_list) > N:
                new_action_list.append('others')
            else:
                new_action_list.append(action_list[frequency[i][0]])
        for i in range(min(len(action_list), N)):
            for j in range(min(len(action_list), N)):
                for x in range(N, len(action_list)) if i == N - 1 and len(action_list) > N else range(i, i + 1):
                    for y in range(N, len(action_list)) if j == N - 1 and len(action_list) > N else range(j, j + 1):
                        new_matrix[i, j] += matrix[frequency[x][0], frequency[y][0]]
        action_list, matrix = new_action_list, new_matrix
        for i in range(matrix.shape[0]):
            if matrix[i].sum() != 0:
                matrix[i] /= matrix[i].sum()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix)
    n = len(action_list)
    ax.set_xlim((0 - 0.5, n - 0.5))
    ax.set_ylim((0 - 0.5, n - 0.5))
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(action_list)))
    ax.set_yticks(np.arange(len(action_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(action_list)
    ax.set_yticklabels(action_list)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(action_list)):
        for j in range(len(action_list)):
            text = ax.text(j, i, '%0.2f' % matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusing matrix output vs target")
    fig.tight_layout()
    plt.savefig(savefile)


def plot_dependency(dependency, input_names, output_names, binarize=None, savefile=None, figsize='auto'):
    assert savefile is not None
    print("Plot dependency to ", savefile)

    matrix = torch.cat([torch.cat(weight, dim=1) for weight in dependency.weights()], dim=0).float()
    if binarize is not None:
        if isinstance(binarize, str):
            if binarize.startswith('top'):
                topk = int(binarize[3:])
                matrix = (matrix - matrix.mean(dim=0, keepdim=True)) / matrix.std(dim=0, keepdim=True)
                topk_indices = matrix.topk(topk, dim=1, sorted=True)[1]
                matrix = torch.zeros_like(matrix).scatter(dim=1, index=topk_indices,
                                                          src=torch.arange(topk, 0, -1,
                                                                           dtype=torch.float).view(1, topk).repeat(
                                                              matrix.size(0), 1),
                                                          )
        else:
            matrix = torch.gt(matrix, binarize).float()
    matrix = matrix.numpy()
    input_labels = []
    for input_name in input_names:
        input_labels += input_name
    output_labels = []
    for output_name in output_names:
        output_labels += output_name
    # print(len(input_labels), len(output_labels), matrix.shape)
    assert matrix.shape == (len(output_labels), len(input_labels))

    if figsize == 'auto':
        figsize = (len(input_labels) * 0.3 + 1.5, len(output_labels) * 0.3 + 1.5)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix)
    # print(matrix[:9, :3])
    n = len(input_labels)
    m = len(output_labels)
    label_offset = -0.5
    ax.set_xlim((0 + label_offset, n + label_offset))
    ax.set_ylim((0 + label_offset, m + label_offset))
    # We want to show all ticks...
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    # ... and label them with the respective list entries
    ax.set_xticklabels(input_labels)
    ax.set_yticklabels(output_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # print(input_labels, output_labels)
    # for i in range(m):
    #     for j in range(n):
    #         text = ax.text(j, i, '%0.2f' % matrix[i, j],
    #                        ha="center", va="center", color="w")

    ax.set_title("Dependency output/input_transform")
    fig.tight_layout()
    plt.savefig(savefile)


def plot_weights(weights, input_names, output_names, log=print, save_dir=None):
    assert weights.size() == torch.Size((len(output_names), len(input_names)))
    for k in range(weights.size(0)):
        items = [(input_names[j], float(weights[k, j].item())) for j in range(weights.size(1))]
        items = list(reversed(sorted(items, key=lambda e: abs(e[1]))))
        with open(os.path.join(save_dir, output_names[k] + '_weights.txt'), 'w') as fout:
            for e in items:
                fout.write("%s: %f\n" % e)


def read_accuracies_per_instance_per_action(trial):
    try:
        eval_log_fin = open('dumps/%s/eval_log.txt' % trial, 'r')
        res = {'pi': {'reg': None, 'new': None}, 'pa': {'reg': None, 'new': None}}
        for line in eval_log_fin:
            if 'Eval acc_per_inst:' in line:
                s = line.find('Eval acc_per_inst:')
                t = 'pi'
            elif 'Eval acc_per_action:' in line:
                s = line.find('Eval acc_per_action:')
                t = 'pa'
            else:
                continue
            words = line[s:].split(' ')
            # print(words)
            reg_acc, new_acc, all_acc = float(words[4].strip(',')), float(words[7].strip(',')), float(
                words[10].strip(',.\n'))
            res[t]['reg'] = reg_acc
            res[t]['new'] = new_acc
            res[t]['all'] = all_acc
        # print(res)
        for t in ['pi', 'pa']:
            for a in 'reg', 'new':
                if res[t][a] is None:
                    raise ValueError()
        eval_log_fin.close()
    except:
        return None
    return res


def read_accuracies_all(trial):
    try:
        eval_log_fin = open('dumps/%s/eval_log.txt' % trial, 'r')
        res = {'pi': {'all': None}, 'pa': {'all': None}}
        for line in eval_log_fin:
            if 'Eval acc_per_inst:' in line:
                s = line.find('Eval acc_per_inst:')
                t = 'pi'
            elif 'Eval acc_per_action:' in line:
                s = line.find('Eval acc_per_action:')
                t = 'pa'
            else:
                continue
            words = line[s:].split(' ')
            all_acc = float(words[4].strip(",.\n"))
            res[t]['all'] = all_acc
        # print(res)
        for t in ['pi', 'pa']:
            if res[t]['all'] is None:
                raise ValueError()
        eval_log_fin.close()
    except:
        return None
    return res


def read_accuracies_dict(trial):
    try:
        eval_log_fin = open('dumps/%s/eval_log.txt' % trial, 'r')
        for line in eval_log_fin:
            if 'Eval acc_dict:' in line:
                s = line.find('Eval acc_dict:') + len('Eval acc_dict:')
            else:
                continue
            dict_repr = line[s:].strip()
            acc_dict = eval(dict_repr)
        eval_log_fin.close()
        return acc_dict
    except:
        return None
    return None
