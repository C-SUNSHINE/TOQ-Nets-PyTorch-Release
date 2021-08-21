#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : config_update.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

def update_config(config, config_update):
    assert isinstance(config, dict), "config should be a dict"
    for k in config_update.keys():
        assert k in config, "Can't find config %s" % k
        v = config_update[k]
        if isinstance(v, ConfigUpdate):
            assert isinstance(config[k], dict), "%s is not a sub-config" % k
            update_config(config[k], v)
        else:
            config[k] = v


def insert_config_update(obj, key, value):
    if len(key) == 0:
        raise KeyError
    if len(key) == 1:
        obj[key[0]] = value
    else:
        assert isinstance(obj, ConfigUpdate)
        if key[0] not in obj:
            obj[key[0]] = ConfigUpdate()
        insert_config_update(obj[key[0]], key[1:], value)


def get_config_update(argv, config_update):
    for i in range(len(argv)):
        if argv[i].startswith('-M'):
            key = argv[i][2:].split('-')
            val = eval(argv[i + 1])
            insert_config_update(config_update, key, val)


class ConfigUpdate:

    def __init__(self, dic=None):
        self._dic = {k: dic[k] for k in dic.keys()} if dic is not None else {}

    def __getitem__(self, item):
        return self._dic[item]

    def __setitem__(self, key, value):
        self._dic[key] = value
        return value

    def __contains__(self, item):
        return item in self._dic

    def __str__(self):
        return str(self._dic)

    def keys(self):
        return self._dic.keys()

    def pop(self, k):
        return self._dic.pop(k)

    def update(self, other):
        for k in other:
            if k in self._dic:
                if isinstance(self._dic[k], ConfigUpdate) and isinstance(other[k], ConfigUpdate):
                    self._dic[k].update(other[k])
                else:
                    self._dic[k] = other[k]
            else:
                raise KeyError("No such config option")
