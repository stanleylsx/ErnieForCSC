# -*- coding: utf-8 -*-
# @Time : 2022/08/08 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm

# [train, interactive_predict, test, convert2tf]
mode = 'train'

# 使用GPU设备
use_cuda = True
cuda_device = -1

configure = {
    'batch_size': 16,
    'checkpoints_dir': 'checkpoints'
}