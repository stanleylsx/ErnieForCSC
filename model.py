# -*- coding: utf-8 -*-
# @Time : 2022/08/08 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm
import torch
from transformers import AutoModel


class ErnieForCSC(torch.nn.Module):
    def __init__(self, pinyin_vocab_size):
        super(ErnieForCSC, self).__init__()
        self.model = AutoModel.from_pretrained('nghuyong/ernie-1.0')
        embedding_dim = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        self.pinyin_embeddings = torch.nn.Embedding(pinyin_vocab_size, embedding_dim, padding_idx=0)
        self.detection_layer = torch.nn.Linear(hidden_size, 2)
        self.correction_layer = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, pinyin_ids, attention_mask=None):
        """
        :param input_ids:
        :param pinyin_ids:
        :param attention_mask:
                attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.
                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.
                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
        :return:
        """
        if attention_mask is None:
            attention_mask = torch.where(input_ids > 0, 1, 0)
        attention_mask = torch.unsqueeze(torch.unsqueeze((1 - attention_mask).to(torch.float32) * -1e9, dim=1), dim=2)

        embedding_output = self.model.embeddings(input_ids=input_ids)

        detection_outputs = self.model.encoder(embedding_output, attention_mask)
        detection_logits = self.detection_layer(detection_outputs['last_hidden_state'])
        detection_error_probs = self.softmax(detection_logits)

        pinyin_embedding_output = self.pinyin_embeddings(pinyin_ids)

        word_pinyin_embedding_output = detection_error_probs[:, :, 0:1] * embedding_output + \
            detection_error_probs[:, :, 1:2] * pinyin_embedding_output

        correction_outputs = self.model.encoder(word_pinyin_embedding_output, attention_mask)
        correction_logits = self.correction_layer(correction_outputs['last_hidden_state'])
        return detection_error_probs, correction_logits, detection_logits
