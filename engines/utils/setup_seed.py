# -*- coding: utf-8 -*-
# @Time : 2023/01/04 09:54
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : setup_seed.py
# @Software: PyCharm
import numpy as np
import random
import os
import torch


def setup_seed(seed):
    # 为CPU设置随机种子
    torch.manual_seed(seed)
    # 为当前GPU设置随机种子
    torch.cuda.manual_seed(seed)
    # 使用多块GPU训练时，为所有GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    # 为了禁止hash随机化，使得实验可复现
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 随机数的种子
    random.seed(seed)
    # numpy的种子
    np.random.seed(seed)

    # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    # torch.backends.cudnn.deterministic = True

    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    # torch.backends.cudnn.benchmark = False
