# -*- coding: utf-8 -*-
# @Time : 2021/8/2 10:28
# @Author : Stanley
# @EMail : gzlishouxian@gmail.com
# @File : make_regex.py
# @Software: VScode
import re


def make_regex(line):
    """
    文本变成正则字符串
    :param line:
    :return:
    """
    return re.sub(r'([()+*?\[\]])', r'\\\1', line)
