# -*- coding: utf-8 -*-
# @Time : 2022/08/08 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm
from transformers import AutoTokenizer
from pypinyin import lazy_pinyin, Style
from config import configure
import torch
import numpy as np


class DataManager:
    def __init__(self, logger):
        self.logger = logger
        self.batch_size = configure['batch_size']
        self.max_sequence_length = configure['max_sequence_length']
        self.tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-1.0')

    def padding(self, token):
        if len(token) < self.max_sequence_length:
            token += [0 for _ in range(self.max_sequence_length - len(token))]
        else:
            token = token[:self.max_sequence_length]
        return token

    def prepare_data(self, data):
        text_list = []
        entity_results_list = []
        token_ids_list = []
        segment_ids_list = []
        attention_mask_list = []
        label_vectors = []
        for item in data:
            text = item.get('text')
            entity_results = {}
            token_results = self.tokenizer(text)
            token_ids = self.padding(token_results.get('input_ids'))
            segment_ids = self.padding(token_results.get('token_type_ids'))
            attention_mask = self.padding(token_results.get('attention_mask'))

            if self.configs['model_type'] == 'bp':
                label_vector = np.zeros((len(token_ids), len(self.categories), 2))
            else:
                label_vector = np.zeros((self.num_labels, len(token_ids), len(token_ids)))

            for entity in item.get('entities'):
                start_idx = entity['start_idx']
                end_idx = entity['end_idx']
                type_class = entity['type']
                token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True,
                                                         max_length=self.max_sequence_length,
                                                         truncation=True)['offset_mapping']
                start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
                end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
                if start_idx in start_mapping and end_idx in end_mapping:
                    class_id = self.categories[type_class]
                    entity_results.setdefault(class_id, set()).add(entity['entity'])
                    start_in_tokens = start_mapping[start_idx]
                    end_in_tokens = end_mapping[end_idx]
                    if self.configs['model_type'] == 'bp':
                        label_vector[start_in_tokens, class_id, 0] = 1
                        label_vector[end_in_tokens, class_id, 1] = 1
                    else:
                        label_vector[class_id, start_in_tokens, end_in_tokens] = 1

            text_list.append(text)
            entity_results_list.append(entity_results)
            token_ids_list.append(token_ids)
            segment_ids_list.append(segment_ids)
            attention_mask_list.append(attention_mask)
            label_vectors.append(label_vector)
        token_ids_list = torch.tensor(token_ids_list)
        segment_ids_list = torch.tensor(segment_ids_list)
        attention_mask_list = torch.tensor(attention_mask_list)
        label_vectors = torch.tensor(np.array(label_vectors))
        return text_list, entity_results_list, token_ids_list, segment_ids_list, attention_mask_list, label_vectors
