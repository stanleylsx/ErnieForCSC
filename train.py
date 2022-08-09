# -*- coding: utf-8 -*-
# @Time : 2022/08/08 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import configure
from model import ErnieForCSC
import json
import torch
import time
import os


class Train:
    def __init__(self, data_manager, device, logger):
        self.device = device
        self.logger = logger
        self.data_manager = data_manager
        self.batch_size = configure['batch_size']
        self.checkpoints_dir = configure['checkpoints_dir']
        self.model_name = configure['model_name']
        self.epochs = configure['epochs']

        learning_rate = configure['learning_rate']
        pinyin_vocab_size = data_manager.pinyin_vocab_size

        self.model = ErnieForCSC(pinyin_vocab_size).to(device)

        params = list(self.model.parameters())
        optimizer_type = configure['optimizer']
        if optimizer_type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(params, lr=learning_rate)
        elif optimizer_type == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(params, lr=learning_rate)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, lr=learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=learning_rate)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        else:
            raise Exception('optimizer_type does not exist')

    def train(self):
        train_data = list(self.data_manager.read_data('datasets/Data/AutomaticCorpusGeneration.txt'))
        train_data.extend(list(self.data_manager.read_data('datasets/Data/sighanCntrain.txt')))
        test_data = list(self.data_manager.read_data('datasets/Data/sighanCntest.txt'))
        print('aswd')
