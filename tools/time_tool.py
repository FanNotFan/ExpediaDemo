#!/usr/bin/env python
# encoding: utf-8
'''
@author: lennon
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: v-lefan@expedia.com
@software: pycharm
@file: time_tool.py
@time: 2019-08-20 11:28
@desc:
'''

import time


class TimeToolObject:

    def format_time_by_yourself(self, format_style):

        return time.strftime(format_style, time.localtime(time.time()))

    def format_time(self):

        return time.strftime('%Y-%m-%d', time.localtime(time.time()))

    def format_time2(self):

        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    def convert_date_to_int(self, dt):

        t = dt.year * 10000 + dt.month * 100 + dt.day
        return t