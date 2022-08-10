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
from utils.metrics import DetectionF1, CorrectionF1
from data import MyData
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

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer_type = configure['optimizer']
        if optimizer_type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer_type == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        else:
            raise Exception('optimizer_type does not exist')

        self.det_loss_act = torch.nn.NLLLoss(ignore_index=self.data_manager.ignore_label)
        self.corr_loss_act = torch.nn.CrossEntropyLoss(ignore_index=self.data_manager.ignore_label, reduction='none')

    def train(self):
        self.logger.info('loading data...')
        train_data = list(self.data_manager.read_data('datasets/Data/AutomaticCorpusGeneration.txt'))
        train_data.extend(list(self.data_manager.read_data('datasets/Data/sighanCntrain.txt')))
        val_data = list(self.data_manager.read_data('datasets/Data/sighanCntest.txt'))

        train_loader = DataLoader(
            dataset=MyData(train_data),
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data,
            shuffle=False
        )
        val_loader = DataLoader(
            dataset=MyData(val_data),
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data,
            shuffle=False
        )

        best_f1 = 0
        best_epoch = 0
        unprocessed = 0
        step_total = self.epochs * len(train_loader)
        global_steps = 0
        scheduler = None

        if configure['warmup']:
            scheduler_type = configure['scheduler_type']
            if configure['num_warmup_steps'] == -1:
                num_warmup_steps = step_total * 0.1
            else:
                num_warmup_steps = configure['num_warmup_steps']

            if scheduler_type == 'linear':
                from transformers.optimization import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            elif scheduler_type == 'cosine':
                from transformers.optimization import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            else:
                raise Exception('scheduler_type does not exist')

        very_start_time = time.time()
        for i in range(self.epochs):
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epochs))
            self.model.train()
            start_time = time.time()
            step, loss, loss_sum = 0, 0.0, 0.0
            for batch in tqdm(train_loader):
                input_ids, pinyin_ids, det_labels, corr_labels, _ = batch
                input_ids = input_ids.to(self.device)
                pinyin_ids = pinyin_ids.to(self.device)
                det_labels = det_labels.to(self.device)
                corr_labels = corr_labels.to(self.device)
                self.optimizer.zero_grad()
                det_error_probs, corr_logits, det_logits = self.model(input_ids, pinyin_ids)
                det_loss = self.det_loss_act(torch.log(det_error_probs).view(-1, det_error_probs.shape[-1]), det_labels.view(-1))
                corr_loss = self.corr_loss_act(corr_logits.view(-1, corr_logits.shape[-1]), corr_labels.view(-1)) * det_error_probs.max(dim=-1)[0].view(-1)
                loss = (det_loss + corr_loss).mean()
                loss.backward()
                self.optimizer.step()

                if configure['warmup']:
                    scheduler.step()

                if global_steps % configure['print_per_batch'] == 0 and global_steps != 0:
                    self.logger.info('global step %d, epoch: %d, batch: %d, loss: %f' %
                                     (global_steps, i + 1, step, loss))

                step = step + 1
                global_steps = global_steps + 1

            det_f1, corr_f1 = self.evaluate(val_loader)
            f1 = (det_f1 + corr_f1) / 2
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)

            if f1 > best_f1:
                unprocessed = 0
                best_f1 = f1
                best_epoch = i + 1
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, self.model_name))
                self.logger.info('saved model successful...')
            else:
                unprocessed += 1

            self.logger.info(
                'f1: %.4f, best_f1: %.4f, best_epoch: %d \n' % (f1, best_f1, best_epoch))
            if configure['is_early_stop']:
                if unprocessed > configure['patient']:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(
                        configure['patient']))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % (
                            (time.time() - very_start_time) / 60))
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        self.logger.info('start evaluate engines...')
        det_metric = DetectionF1()
        corr_metric = CorrectionF1()
        for step, batch in tqdm(enumerate(val_loader, start=1)):
            input_ids, pinyin_ids, det_labels, corr_labels, length = batch
            input_ids = input_ids.to(self.device)
            pinyin_ids = pinyin_ids.to(self.device)
            det_labels = det_labels.to(self.device)
            corr_labels = corr_labels.to(self.device)
            det_error_probs, corr_logits, det_logits = self.model(input_ids, pinyin_ids)
            det_metric.update(det_error_probs, det_labels, length)
            corr_metric.update(det_error_probs, det_labels, corr_logits,
                               corr_labels, length)

        det_f1, det_precision, det_recall = det_metric.accumulate()
        corr_f1, corr_precision, corr_recall = corr_metric.accumulate()
        self.logger.info('Sentence-Level Performance:')
        self.logger.info('Detection  metric: F1={:.4f}, Recall={:.4f}, Precision={:.4f}'.
                    format(det_f1, det_recall, det_precision))
        self.logger.info('Correction metric: F1={:.4f}, Recall={:.4f}, Precision={:.4f}'.
                    format(corr_f1, corr_recall, corr_precision))
        return det_f1, corr_f1
