#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rlbench_cfg.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

DATA_PATH = "/data/vision/billf/scratch/lzz/TOQ-Nets-PyTorch/data/rlbench_v2"
N_PACKAGES_PER_ACTION = 2
N_SAMPLE_PER_ACTION = 100

default_action_lists = {
    'default18': ['CloseBox', 'CloseJar', 'CloseLaptopLid', 'GetIceFromFridge', 'OpenBox', 'OpenWineBottle',
                  'PickUpCup', 'PressSwitch', 'PushButtons', 'PutGroceriesInCupboard', 'PutItemInDrawer',
                  'PutRubbishInBin', 'PutTrayInOven', 'ScoopWithSpatula', 'SetTheTable', 'SlideCabinetOpenAndPlaceCups',
                  'TakePlateOffColoredDishRack', 'TakeToiletRollOffStand'],
    'default14': ['CloseBox', 'CloseJar', 'CloseLaptopLid', 'GetIceFromFridge', 'OpenBox', 'OpenWineBottle',
                  'PickUpCup', 'PressSwitch', 'PushButtons', 'PutItemInDrawer', 'ScoopWithSpatula', 'SetTheTable',
                  'SlideCabinetOpenAndPlaceCups', 'TakePlateOffColoredDishRack'],
    'default10': ['CloseBox', 'GetIceFromFridge', 'OpenBox', 'PickUpCup', 'PushButtons', 'PutItemInDrawer',
                  'ScoopWithSpatula', 'SetTheTable', 'SlideCabinetOpenAndPlaceCups', 'TakePlateOffColoredDishRack']
}

toy_actions = [
    'CloseBox', 'PickUpCup', 'CloseMicrowave', 'OpenBox', 'OpenMicrowave'
]

fewshot_default_split = {
    'OpenMicrowave': {
        'reg': ['OpenBox', 'OpenWineBottle'],
        'new': ['OpenMicrowave'],
    },
    'CloseMicrowave': {
        'reg': ['CloseBox', 'CloseDrawer', 'CloseFridge', 'CloseGrill', 'CloseJar', 'CloseLaptopLid'],
        'new': ['CloseMicrowave'],
    },
    'OpenBox': {
        'reg': ['OpenMicrowave', 'OpenWineBottle'],
        'new': ['OpenBox'],
    },
    'OpenMicrowave+WineBottle': {
        'reg': ['OpenBox'],
        'new': ['OpenWineBottle', 'OpenMicrowave'],
    },
    'CloseDrawer': {
        'reg': ['CloseBox', 'CloseFridge', 'CloseGrill', 'CloseJar', 'CloseLaptopLid', 'CloseMicrowave'],
        'new': ['CloseDrawer'],
    },
    'CloseBox+Fridge': {
        'reg': ['CloseDrawer', 'CloseGrill', 'CloseJar', 'CloseLaptopLid', 'CloseMicrowave'],
        'new': ['CloseBox', 'CloseFridge'],
    },
    'CloseDrawer+Grill': {
        'reg': ['CloseBox', 'CloseFridge', 'CloseJar', 'CloseLaptopLid', 'CloseMicrowave'],
        'new': ['CloseDrawer', 'CloseGrill'],
    },
    'CloseJar+LaptopLid': {
        'reg': ['CloseBox', 'CloseDrawer', 'CloseFridge', 'CloseGrill', 'CloseMicrowave'],
        'new': ['CloseJar', 'CloseLaptopLid'],
    },

    'OpenBox+CloseBox': {
        'reg': [],
        'new': ['OpenBox', 'CloseBox']
    },
    'OpenMicrowave+CloseMicrowave': {
        'reg': [],
        'new': ['OpenMicrowave', 'CloseMicrowave']
    },
    'OpenFridge+CloseFridge': {
        'reg': [],
        'new': ['OpenFridge', 'CloseFridge']
    },
    'OpenBox+Microwave+CloseDrawer+Fridge+Jar+Grill': {
        'reg': [],
        'new': ['OpenBox', 'OpenMicrowave', 'CloseDrawer', 'CloseFridge', 'CloseGrill', 'CloseJar']
    },
    'OpenFridge+WineBottle+CloseBox+Fridge+Microwave': {
        'reg': [],
        'new': ['OpenFridge', 'OpenWineBottle', 'CloseBox', 'CloseFridge', 'CloseMicrowave']
    },
    'OpenBox+Fridge+CloseDrawer+LaptopLid+Microwave': {
        'reg': [],
        'new': ['OpenBox', 'OpenFridge', 'CloseDrawer', 'CloseLaptopLid', 'CloseMicrowave']
    },
    'OpenMicrowave+WineBottle+CloseBox+Grill+Microwave': {
        'reg': [],
        'new': ['OpenMicrowave', 'OpenWineBottle', 'CloseBox', 'CloseGrill', 'CloseMicrowave']
    },

    'OpenWineBottle+Fridge+Microwave+CloseBox+Grill+Microwave': {  # new 1
        'reg': [],
        'new': ['OpenWineBottle', 'OpenFridge', 'OpenMicrowave', 'CloseBox', 'CloseGrill', 'CloseMicrowave']
    },
    'OpenDrawer+Fridge+CloseBox+Fridge+Microwave': {  # new 2
        'reg': [],
        'new': ['OpenDrawer', 'OpenFridge', 'CloseBox', 'CloseFridge', 'CloseMicrowave']
    },
    'OpenWineBottle+Grill+Microwave+CloseBox+Fridge+Grill': {  # new 3
        'reg': [],
        'new': ['OpenWineBottle', 'OpenGrill', 'OpenMicrowave', 'CloseBox', 'CloseFridge', 'CloseGrill']
    },
    'OpenBox+WineBottle+CloseFridge+LaptopLid+Microwave': {  # new 4
        'reg': [],
        'new': ['OpenBox', 'OpenWineBottle', 'CloseFridge', 'CloseLaptopLid', 'CloseMicrowave']
    },

    'OpenX+CloseX': {
        'reg': [],
        'new': ['OpenBox', 'OpenWineBottle', 'OpenDrawer', 'OpenFridge', 'OpenGrill', 'OpenMicrowave'] +
               ['CloseBox', 'CloseDrawer', 'CloseFridge', 'CloseGrill', 'CloseJar', 'CloseLaptopLid', 'CloseMicrowave']
    },
    'None': {
        'reg': [],
        'new': []
    }
}

fewshot_default_labels = {

    'Open': {
        'label_names': ['NotOpen', 'Open'],
        1: ['OpenBox', 'OpenWineBottle', 'OpenDrawer', 'OpenFridge', 'OpenGrill', 'OpenMicrowave'],
    },
    'Close': {
        'label_names': ['NotClose', 'Close'],
        1: ['CloseBox', 'CloseDrawer', 'CloseFridge', 'CloseGrill', 'CloseJar', 'CloseLaptopLid', 'CloseMicrowave'],
    },
    'OtherOpenClose': {
        'label_names': ['Other', 'Open', 'Close'],
        1: ['OpenBox', 'OpenWineBottle', 'OpenDrawer', 'OpenFridge', 'OpenGrill', 'OpenMicrowave'],
        2: ['CloseBox', 'CloseDrawer', 'CloseFridge', 'CloseGrill', 'CloseJar', 'CloseLaptopLid', 'CloseMicrowave'],
    }
}
