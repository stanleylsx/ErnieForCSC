# -*- coding: utf-8 -*-
# @Time : 2022/08/08 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm
import torch
import os
import time
from torch.utils.data import DataLoader
from utils.utils import is_chinese_char
from config import configure
from model import ErnieForCSC
from utils.metrics import DetectionF1, CorrectionF1
from pypinyin import lazy_pinyin, Style
from data import MyData
from tqdm import tqdm


class Predictor:
    def __init__(self, data_manager, device, logger):
        self.device = device
        self.data_manager = data_manager
        self.logger = logger
        self.checkpoints_dir = configure['checkpoints_dir']
        self.model_name = configure['model_name']
        pinyin_vocab_size = data_manager.pinyin_vocab_size
        vocab_size = data_manager.vocab_size + 1
        self.model = ErnieForCSC(pinyin_vocab_size, vocab_size).to(device)
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
        self.model.eval()

    def predict_one(self, sentence):
        """
        预测接口
        """
        start_time = time.time()
        det_pred = []
        words = list(sentence)
        if len(words) > self.data_manager.max_sequence_length - 2:
            words = words[:self.data_manager.max_sequence_length - 2]
        words = ['[CLS]'] + words + ['[SEP]']
        input_ids = self.data_manager.tokenizer.convert_tokens_to_ids(words)

        # Use pad token in pinyin emb to map word emb [CLS], [SEP]
        pinyins = lazy_pinyin(sentence, style=Style.TONE3, neutral_tone_with_five=True)
        pinyin_ids = [0]
        # Align pinyin and chinese char
        # 对于长度不为1的字符或不为中文的字符 将pinyin_vocab['UNK']或pinyin['PAD']添加至pinyin_ids
        pinyin_offset = 0
        for i, word in enumerate(words[1:-1]):
            pinyin = '[UNK]' if word != '[PAD]' else '[PAD]'
            if len(word) == 1 and is_chinese_char(ord(word)):
                while pinyin_offset < len(pinyins):
                    current_pinyin = pinyins[pinyin_offset][:-1]
                    pinyin_offset += 1
                    if current_pinyin in self.data_manager.pinyin2id:
                        pinyin = current_pinyin
                        break
            pinyin_ids.append(self.data_manager.pinyin2id[pinyin])

        pinyin_ids.append(0)
        assert len(input_ids) == len(pinyin_ids), 'length of input_ids must be equal to length of pinyin_ids'
        input_ids = torch.tensor([input_ids]).to(self.device)
        pinyin_ids = torch.tensor([pinyin_ids]).to(self.device)
        det_error_probs, corr_logits, det_logits = self.model(input_ids, pinyin_ids)

        corr_predictions = list(torch.argmax(corr_logits.to('cpu'), dim=-1).numpy()[0])
        corr_sentence = self.data_manager.tokenizer.convert_ids_to_tokens(corr_predictions)
        det_predictions = list(det_error_probs.argmax(dim=-1).to('cpu').numpy()[0])[1:-1]
        result = zip(words[1:-1], det_predictions, corr_sentence[1:-1])
        for item in result:
            if item[1] == 1:
                det_pred.append((item[0], item[2]))
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        return ''.join(corr_sentence[1:-1]), det_pred

    def predict_test(self):
        self.logger.info('start test...')
        val_data = list(self.data_manager.read_data(configure['test_file']))
        test_loader = DataLoader(
            dataset=MyData(val_data),
            batch_size=self.data_manager.batch_size,
            collate_fn=self.data_manager.prepare_data
        )
        det_metric = DetectionF1()
        corr_metric = CorrectionF1()
        for step, batch in tqdm(enumerate(test_loader, start=1)):
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
