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
    'checkpoints_dir': 'checkpoints',
    'optimizer': 'AdamW',
    'max_sequence_length': 128,
    'learning_rate': 5e-5,
    'epochs': 5,
    'batch_size': 16,
    'model_name': 'ernie4csc'
}