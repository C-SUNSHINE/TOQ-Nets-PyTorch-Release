#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import torch


def get_temporal_indicator(indicator, length, batch_middle=None, device='cpu'):
    mid = (length - 1) / 2 if batch_middle is None else batch_middle.type(torch.float)
    if indicator.startswith('gaussian'):
        sigma = int(indicator[8:])
        if batch_middle is None:
            res = torch.arange(length, dtype=torch.float, device=device)
            res = torch.exp(-(res - mid) ** 2 / (2 * sigma ** 2))
            res /= torch.max(res, dim=0)[0]
        else:
            res = torch.arange(length, dtype=torch.float, device=device).unsqueeze(0).repeat(mid.size(0), 1)
            res = torch.exp(-(res - mid.view(-1, 1)) ** 2 / (2 * sigma ** 2))
            res /= torch.max(res, dim=1, keepdim=True)[0]
            # print(res[0])
        # print(mid)
        # input()
    elif indicator.startswith('linear'):
        radius = int(indicator[6:])
        if batch_middle is None:
            left_wing = (torch.arange(length, dtype=torch.float, device=device) - mid + radius) / radius
            right_wing = (-torch.arange(length, dtype=torch.float, device=device) + mid + radius) / radius
            left_wing = torch.max(left_wing, torch.zeros_like(left_wing))
            right_wing = torch.max(right_wing, torch.zeros_like(right_wing))
            res = torch.min(left_wing, right_wing)
        else:
            left_wing = (torch.arange(length, dtype=torch.float, device=device).unsqueeze(0).repeat(
                mid.size(0), 1) - mid.view(-1, 1) + radius) / radius
            right_wing = (-torch.arange(length, dtype=torch.float, device=device).unsqueeze(0).repeat(
                mid.size(0), 1) + mid.view(-1, 1) + radius) / radius
            left_wing = torch.max(left_wing, torch.zeros_like(left_wing))
            right_wing = torch.max(right_wing, torch.zeros_like(right_wing))
            res = torch.min(left_wing, right_wing)
    return res
