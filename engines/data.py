# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
from transformers import BertTokenizerFast
from tqdm import tqdm
import torch
import os
import re
import json
import math
import pandas as pd
import asyncio
import numpy as np


class DataManager:
    def __init__(self, configs, logger):
        self.logger = logger
        self.configs = configs
        self.train_file = self.configs['train_file']
        self.dev_file = self.configs['dev_file']
        self.file_format = self.train_file.split('.')[-1]
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
        self.categories = {configs['classes'][index]: index + 1 for index in range(0, len(configs['classes']))}
        self.reverse_categories = {class_id: class_name for class_name, class_id in self.categories.items()}

    def padding(self, token, pad_token=True):
        if len(token) < self.max_sequence_length:
            token += [0 for _ in range(self.max_sequence_length - len(token))]
        else:
            token = token[:self.max_sequence_length]
            if pad_token and 'ptm' in self.configs['model_type']:
                token[-1] = 102
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
        if self.file_format == 'csv':
            df = pd.read_csv(self.train_file, names=['token', 'label'], sep=' ')
            if not self.dev_file == '':
                dev_df = pd.read_csv(self.dev_file, names=['token', 'label'], sep=' ').sample(frac=1)
                df = pd.concat([df, dev_df], axis=0)
            tokens = list(set(df['token'][df['token'].notnull()]))
        elif self.file_format == 'json':
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

    @staticmethod
    def split_csv(data, batch_nums=2000):
        null = data.token.isnull().to_frame('isnull')
        null = null.loc[null['isnull']]
        df_num = len(null)
        n = math.ceil(df_num / batch_nums)
        df_list = []
        past_last_index = 0
        for index in range(n):
            if index < n - 1:
                df_null = null[batch_nums * index: batch_nums * (index + 1)]
            else:
                df_null = null[batch_nums * index:]
            df_part = df_null.iloc[[-1]].index.values[0] + 1
            df_list.append(data[past_last_index:df_part - 1])
            past_last_index = df_part
        return df_list

    def csv_to_json(self, df):
        async def csv_to_json_async(sub_df):
            result = []
            sentence = []
            each_label = []
            lines = sub_df.token.isnull().sum()
            with tqdm(total=lines, desc='loading data') as bar:
                for index, record in sub_df.iterrows():
                    token = record.token
                    label = record.label
                    entities = []
                    each_simple = {'text': ''}
                    if str(token) == str(np.nan):
                        each_simple['text'] = ''.join(sentence)
                        start_idx = 0
                        end_idx = 0
                        while start_idx <= len(sentence) - 1:
                            if each_label[start_idx] in self.classes:
                                if re.findall(r'^B-', each_label[start_idx]):
                                    entity_dict = {'start_idx': start_idx,
                                                   'type': re.split(r'^B-', each_label[start_idx])[-1]}
                                    entity = sentence[start_idx]
                                    end_idx = start_idx + 1
                                    while re.findall(r'^I-', each_label[end_idx]):
                                        entity += sentence[end_idx]
                                        end_idx += 1
                                        if end_idx == len(sentence):
                                            break
                                    entity_dict['end_idx'] = end_idx - 1
                                    entity_dict['entity'] = entity
                                    entities.append(entity_dict)
                            end_idx += 1
                            start_idx += 1
                        each_simple['entities'] = entities
                        result.append(each_simple)
                        sentence = []
                        each_label = []
                        bar.update()
                    else:
                        sentence.append(token)
                        each_label.append(label)
            return result
        self.logger.info('transfer csv to json!')
        data = []
        sub_data_list = self.split_csv(df)
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        r = asyncio.gather(*[csv_to_json_async(items) for items in tqdm(sub_data_list)])
        data_list = loop.run_until_complete(r)
        for item in data_list:
            data.extend(item)
        return data

    @staticmethod
    def get_sequence_label(item_dict):
        tokens = list(item_dict['text'])
        labels = len(tokens) * ['O']
        if item_dict['entities']:
            for entity in item_dict['entities']:
                try:
                    start_idx = entity['start_idx']
                    end_idx = entity['end_idx']
                    entity_type = entity['type']
                    labels[start_idx] = 'B-' + entity_type
                    for i in range(start_idx + 1, end_idx + 1):
                        labels[i] = 'I-' + entity_type
                except IndexError:
                    continue
        return labels

    def tokenizer_for_sentences(self, sent):
        sent = list(sent)
        tokens = []
        for token in sent:
            if token in self.token2id:
                tokens.append(self.token2id[token])
            else:
                tokens.append(self.token2id[self.UNKNOWN])
        return tokens

    def prepare_data(self, data):
        text_list = []
        entity_results_list = []
        token_ids_list = []
        label_vectors = []
        if self.configs['method'] == 'span':
            for item in data:
                text = item.get('text')
                entity_results = {}
                token_results = self.tokenizer(text)
                token_ids = self.padding(token_results.get('input_ids'))

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
                label_vectors.append(label_vector)
            token_ids_list = torch.tensor(token_ids_list)
            label_vectors = torch.tensor(np.array(label_vectors))
        elif self.configs['method'] == 'sequence_label':
            for item in data:
                text = item.get('text')
                if 'ptm' in self.configs['model_type']:
                    token_results = self.tokenizer(text)
                    token_ids = token_results.get('input_ids')
                else:
                    token_ids = self.tokenizer_for_sentences(text)
                token_ids = self.padding(token_ids)
                labels = [self.categories[label] for label in self.get_sequence_label(item)]
                label_vector = self.padding(labels, pad_token=False)
                token_ids_list.append(token_ids)
                label_vectors.append(label_vector)
            token_ids_list = torch.tensor(token_ids_list)
            label_vectors = torch.tensor(np.array(label_vectors))
        return text_list, entity_results_list, token_ids_list, label_vectors

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
