#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : logging.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/14/2020
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

from jacinle.logging import get_logger, set_output_file

import logging
import time

__all__ = ['get_logger', 'set_output_file', 'LZZLogger', 'setup_logger']


class LZZLogger(logging.getLoggerClass()):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.record = []

    def log_err(self, agent='...', message=''):
        self.warning("[%s %.3fs] %s" % (agent, time.time() - self.start_time, message))

    def log_begin(self, agent='...', message=''):
        self.record.append((agent, message, time.time() - self.start_time))
        self.info("[%s %.3fs] %s <start>" % (agent, time.time() - self.start_time, message))
        return len(self.record) - 1

    def log_end(self, index):
        duration = (time.time() - self.start_time) - self.record[index][2]
        self.info("[%s %.3fs]: %s <finish in %.3fs>" % (
            self.record[index][0], time.time() - self.start_time, self.record[index][1], duration
        ))


def setup_logger():
    logging.setLoggerClass(LZZLogger)
