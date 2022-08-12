# -*- coding: utf-8 -*-
# @Time : 2022/08/08 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm
from transformers import AutoTokenizer
from pypinyin import lazy_pinyin, Style
from config import configure
from tqdm import tqdm
from utils.utils import is_chinese_char
from torch.utils.data import Dataset
import torch
import os


class DataManager:
    def __init__(self, logger):
        self.logger = logger
        self.batch_size = configure['batch_size']
        self.max_sequence_length = configure['max_sequence_length']
        self.tokenizer = AutoTokenizer.from_pretrained(configure['pretrained_model'])
        self.vocab_size = len(self.tokenizer)
        self.pinyin2id, self.id2pinyin = self.load_pinyin_vocab()
        self.pinyin_vocab_size = len(self.pinyin2id)
        self.ignore_label = -100

    def load_pinyin_vocab(self):
        if not os.path.exists('datasets/pinyin_vocab.txt'):
            self.logger.info('pinyin vocab file not exist...')
            raise Exception('pinyin vocab file not exist...')

        with open('datasets/pinyin_vocab.txt', 'r', encoding='utf-8') as infile:
            pinyin_token_list = [pinyin.rstrip('\n') for pinyin in infile.readlines()]
            token2id = dict(zip(pinyin_token_list, range(0, len(pinyin_token_list))))
            id2token = dict(zip(range(0, len(pinyin_token_list)), pinyin_token_list))
        return token2id, id2token

    @staticmethod
    def read_data(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                source, target = line.strip('\n').split('\t')[0: 2]
                yield {'source': source, 'target': target}

    def padding(self, token):
        if len(token) < self.max_sequence_length:
            token += [0 for _ in range(self.max_sequence_length - len(token))]
        else:
            token = token[:self.max_sequence_length]
        return token

    def prepare_data(self, data_list):
        input_ids_list = []
        pinyin_ids_list = []
        detection_labels_list = []
        correction_labels_list = []
        length_list = []
        for data in data_list:
            source = data['source']
            words = list(source)
            if len(words) > self.max_sequence_length - 2:
                words = words[:self.max_sequence_length - 2]
            length = len(words)
            words = ['[CLS]'] + words + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(words)

            # Use pad token in pinyin emb to map word emb [CLS], [SEP]
            pinyins = lazy_pinyin(source, style=Style.TONE3, neutral_tone_with_five=True)
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
                        if current_pinyin in self.pinyin2id:
                            pinyin = current_pinyin
                            break
                pinyin_ids.append(self.pinyin2id[pinyin])

            pinyin_ids.append(0)
            assert len(input_ids) == len(pinyin_ids), 'length of input_ids must be equal to length of pinyin_ids'

            target = data['target']
            correction_labels = list(target)
            if len(correction_labels) > self.max_sequence_length - 2:
                correction_labels = correction_labels[:self.max_sequence_length - 2]
            correction_labels = self.tokenizer.convert_tokens_to_ids(correction_labels)
            correction_labels = [self.ignore_label] + correction_labels + [self.ignore_label]

            detection_labels = []
            for input_id, label in zip(input_ids[1:-1], correction_labels[1:-1]):
                detection_label = 0 if input_id == label else 1
                detection_labels += [detection_label]
            detection_labels = [self.ignore_label] + detection_labels + [self.ignore_label]

            input_ids_list.append(self.padding(input_ids))
            pinyin_ids_list.append(self.padding(pinyin_ids))
            detection_labels_list.append(self.padding(detection_labels))
            correction_labels_list.append(self.padding(correction_labels))
            length_list.append(length)

        input_ids_list = torch.tensor(input_ids_list)
        pinyin_ids_list = torch.tensor(pinyin_ids_list)
        detection_labels_list = torch.tensor(detection_labels_list)
        correction_labels_list = torch.tensor(correction_labels_list)
        length_list = torch.tensor(length_list)
        return input_ids_list, pinyin_ids_list, detection_labels_list, correction_labels_list, length_list


class MyData(Dataset):
    def __init__(self, data):
        super(MyData, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
