# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
from transformers import BertTokenizerFast
import torch
import os
import json
import pandas as pd
import numpy as np


class DataManager:
    def __init__(self, configs, logger):
        self.logger = logger
        self.configs = configs
        self.train_file = self.configs['train_file']
        self.dev_file = self.configs['dev_file']
        self.batch_size = configs['batch_size']
        self.token_file = configs['token_file']
        self.max_sequence_length = configs['max_sequence_length']
        self.UNKNOWN = '[UNK]'
        self.PADDING = '[PAD]'

        if 'ptm_' not in configs['model_type']:
            self.token2id, self.id2token = self.load_vocab()
            self.vocab_size = len(self.token2id) + 1
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(configs['ptm'])
            self.vocab_size = len(self.tokenizer)

        self.classes = configs['classes']

        self.num_labels = len(self.classes)
        self.categories = {configs['classes'][index]: index for index in range(0, len(configs['classes']))}
        self.reverse_categories = {class_id: class_name for class_name, class_id in self.categories.items()}

    def padding(self, token):
        if len(token) < self.max_sequence_length:
            token += [0 for _ in range(self.max_sequence_length - len(token))]
        else:
            token = token[:self.max_sequence_length]
        return token

    def load_vocab(self):
        """
        若不存在词表则生成，若已经存在则加载词表
        :return:
        """
        if not os.path.exists(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab()

        self.logger.info('loading vocab...')
        token2id, id2token = {}, {}

        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        return token2id, id2token

    def build_vocab(self):
        """
        根据训练集生成词表
        :return:
        """
        if self.configs['data_format'] == 'csv':
            df = pd.read_csv(self.train_file, names=['token', 'label'], sep=' ')
            if not self.dev_file == '':
                dev_df = pd.read_csv(self.dev_file, names=['token', 'label'], sep=' ').sample(frac=1)
                df = pd.concat([df, dev_df], axis=0)
            tokens = list(set(df['token'][df['token'].notnull()]))
        elif self.configs['data_format'] == 'json':
            tokens = []
            data = json.load(open(self.train_file, encoding='utf-8'))
            if not self.dev_file == '':
                dev_data = json.load(open(self.dev_file, encoding='utf-8'))
                data.extend(dev_data)
            for sentence in data:
                text = sentence.get('text')
                for char in list(text):
                    if char.strip() not in ['\t', '']:
                        tokens.extend(char)
            tokens = list(set(tokens))
        else:
            tokens = []

        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1
        # 保存词表及标签表
        with open(self.token_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        return token2id, id2token

    def csv_to_json(self, data):
        pass

    def json_to_csv(self, data):
        pass

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

            if self.configs['model_type'] == 'ptm_bp':
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
                    if self.configs['model_type'] == 'ptm_bp':
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

    def extract_entities(self, text, model_output):
        """
        从验证集中预测到相关实体
        """
        predict_results = {}
        token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True,
                                                 max_length=self.max_sequence_length,
                                                 truncation=True)['offset_mapping']
        start_mapping = {i: j[0] for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        end_mapping = {i: j[-1] - 1 for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        if self.configs['model_type'] == 'ptm_bp':
            model_output = torch.sigmoid(model_output)
            decision_threshold = float(self.configs['decision_threshold'])
            start = np.where(model_output[:, :, 0] > decision_threshold)
            end = np.where(model_output[:, :, 1] > decision_threshold)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        if _start in start_mapping and _end in end_mapping:
                            start_in_text = start_mapping[_start]
                            end_in_text = end_mapping[_end]
                            entity_text = text[start_in_text: end_in_text + 1]
                            predict_results.setdefault(predicate1, set()).add(entity_text)
                        break
        else:
            for class_id, start, end in zip(*np.where(model_output > 0)):
                if start <= end:
                    if start in start_mapping and end in end_mapping:
                        start_in_text = start_mapping[start]
                        end_in_text = end_mapping[end]
                        entity_text = text[start_in_text: end_in_text + 1]
                        predict_results.setdefault(class_id, set()).add(entity_text)
        return predict_results
