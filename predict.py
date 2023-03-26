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
        from train import Train
        train = Train(self.data_manager, self.device, self.logger)
        train.evaluate(self.model, test_loader)

    def convert_torch_to_tf(self):
        max_sequence_length = self.data_manager.max_sequence_length
        dummy_input = torch.ones([1, max_sequence_length]).to('cpu').long()
        dummy_input = (dummy_input, dummy_input)
        onnx_path = self.checkpoints_dir + '/model.onnx'
        torch.onnx.export(self.model.to('cpu'), dummy_input, f=onnx_path, opset_version=10,
                          input_names=['token_input', 'pinyin_input'],
                          output_names=['detection_error_probs', 'correction_logits', 'detection_logits'],
                          do_constant_folding=False,
                          dynamic_axes={'token_input': {0: 'batch_size'}, 'pinyin_input': {0: 'batch_size'},
                                        'detection_error_probs': {0: 'batch_size'},
                                        'correction_logits': {0: 'batch_size'}, 'detection_logits': {0: 'batch_size'}})
        self.logger.info('convert torch to onnx successful...')
