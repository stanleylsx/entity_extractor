# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: VScode
from transformers import BertTokenizerFast
from tqdm import tqdm
from engines.utils.make_regex import make_regex
from engines.utils.detokenizer import Detokenizer
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
        self.detokenizer = Detokenizer()

        if 'ptm' not in configs['model_type']:
            self.token2id, self.id2token = self.load_vocab()
            self.vocab_size = len(self.token2id) + 1
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(configs['ptm'])
            self.vocab_size = len(self.tokenizer)

        # csv格式的标签是BIO标签
        # json格式的标签是类别标签
        self.span_classes = configs['span_classes']
        self.span_categories = {self.span_classes[index]: index for index in range(0, len(self.span_classes))}
        self.span_reverse_categories = {class_id: class_name for class_name, class_id in self.span_categories.items()}
        self.span_num_labels = len(self.span_reverse_categories)

        self.sequence_tag_classes = configs['sequence_tag_classes']
        self.sequence_tag_categories = {self.sequence_tag_classes[index]: index for index
                                        in range(0, len(self.sequence_tag_classes))}
        self.sequence_tag_reverse_categories = {class_id: class_name for class_name, class_id
                                                in self.sequence_tag_categories.items()}
        self.sequence_tag_num_labels = len(self.sequence_tag_reverse_categories)

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
            df_list.append(data[past_last_index:df_part])
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
                        start_idx = 0
                        end_idx = 0
                        while start_idx <= len(sentence) - 1:
                            if each_label[start_idx] in self.sequence_tag_classes:
                                if re.findall(r'^B-', each_label[start_idx]):
                                    entity_type = re.split(r'^B-', each_label[start_idx])[-1]
                                    entity_dict = {'start_idx': start_idx, 'type': entity_type}
                                    entity = sentence[start_idx]
                                    end_idx = start_idx + 1
                                    if end_idx != len(each_label):
                                        while re.findall(r'^I-' + entity_type, each_label[end_idx]):
                                            if self.configs['token_level'] == 'word':
                                                char_ = ' ' + sentence[end_idx]
                                                entity += char_
                                            else:
                                                entity += sentence[end_idx]
                                            end_idx += 1
                                            if end_idx == len(sentence):
                                                break
                                    entity_dict['end_idx'] = end_idx - 1
                                    entity = entity.strip()
                                    entity_dict['entity'] = entity
                                    entities.append(entity_dict)
                            end_idx += 1
                            start_idx += 1
                        if self.configs['token_level'] == 'word':
                            word_entities = []
                            text = self.detokenizer.tokenize(sentence)
                            each_simple['text'] = text
                            entities = list(set([entity['entity'] for entity in entities]))
                            entities.sort(key=lambda i: len(i), reverse=True)
                            start_set = []
                            for entity in entities:
                                entity_locs = re.finditer(make_regex(entity), text)
                                for entity_loc in entity_locs:
                                    start, end = entity_loc.span()
                                    end = end - 1
                                    if start in start_set:
                                        continue
                                    entity_dict = {'start_idx': start, 'end_idx': end, 'type': entity}
                                    word_entities.append(entity_dict)
                                    each_simple['entities'] = word_entities
                                    start_set.append(start)

                        else:
                            each_simple['text'] = ''.join(sentence)
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
        entity_results_list = []
        token_ids_list = []
        label_vectors = []
        text_list = [item.get('text') for item in data]
        if self.configs['method'] == 'span':
            token_results = self.tokenizer.batch_encode_plus(text_list, padding=True, truncation=True,
                                                             max_length=self.max_sequence_length, return_tensors='pt')
            token_ids_list = token_results.get('input_ids')
            token_length = token_ids_list.size(1)
            for item in zip(data, text_list):
                data = item[0]
                text = item[1]
                entity_results = {}

                if self.configs['model_type'] == 'ptm_bp':
                    label_vector = np.zeros((token_length, len(self.span_categories), 2))
                else:
                    label_vector = np.zeros((self.span_num_labels, token_length, token_length))

                for entity in data.get('entities'):
                    start_idx = entity['start_idx']
                    end_idx = entity['end_idx']
                    type_class = entity['type']
                    token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True,
                                                             max_length=self.max_sequence_length,
                                                             truncation=True)['offset_mapping']
                    start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
                    end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
                    if start_idx in start_mapping and end_idx in end_mapping:
                        class_id = self.span_categories[type_class]
                        entity_results.setdefault(class_id, set()).add(entity['entity'])
                        start_in_tokens = start_mapping[start_idx]
                        end_in_tokens = end_mapping[end_idx]
                        if self.configs['model_type'] == 'ptm_bp':
                            label_vector[start_in_tokens, class_id, 0] = 1
                            label_vector[end_in_tokens, class_id, 1] = 1
                        else:
                            label_vector[class_id, start_in_tokens, end_in_tokens] = 1
                entity_results_list.append(entity_results)
                label_vectors.append(label_vector)
            token_ids_list = torch.tensor(token_ids_list)
            label_vectors = torch.tensor(np.array(label_vectors))
        elif self.configs['method'] == 'sequence_tag':
            for item in data:
                text = item.get('text')
                entity_results = {}
                if 'ptm' in self.configs['model_type']:
                    token_results = self.tokenizer(text)
                    token_ids = token_results.get('input_ids')
                else:
                    token_ids = self.tokenizer_for_sentences(text)
                token_ids = self.padding(token_ids)
                labels = [self.sequence_tag_categories[label] for label in self.get_sequence_label(item)]
                if item['entities']:
                    for entity in item['entities']:
                        if entity['type'] in self.span_categories:
                            class_id = self.span_categories[entity['type']]
                            entity_results.setdefault(class_id, set()).add(entity['entity'])
                label_vector = self.padding(labels, pad_token=False)
                token_ids_list.append(token_ids)
                entity_results_list.append(entity_results)
                label_vectors.append(label_vector)
            token_ids_list = torch.tensor(token_ids_list)
            label_vectors = torch.tensor(np.array(label_vectors))
        return text_list, entity_results_list, token_ids_list, label_vectors

    def extract_entities(self, text, model_output, inference=False):
        """
        从验证集中预测到相关实体
        """
        predict_results = {}
        if 'ptm' in self.configs['model_type']:
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True,
                                                     max_length=self.max_sequence_length,
                                                     truncation=True)['offset_mapping']
            start_mapping = {i: j[0] for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            end_mapping = {i: j[-1] - 1 for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        else:
            token2char_span_mapping = None
            start_mapping = None
            end_mapping = None
        if self.configs['method'] == 'span':
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
                                if inference:
                                    predict_results.setdefault(predicate1, []).append(
                                        {'entity': entity_text, 'entity_loc': [start_in_text, end_in_text]})
                                else:
                                    predict_results.setdefault(predicate1, set()).add(entity_text)
                            break
            else:
                for class_id, start, end in zip(*np.where(model_output > 0)):
                    if start <= end:
                        if start in start_mapping and end in end_mapping:
                            start_in_text = start_mapping[start]
                            end_in_text = end_mapping[end]
                            entity_text = text[start_in_text: end_in_text + 1]
                            if inference:
                                predict_results.setdefault(class_id, []).append(
                                    {'entity': entity_text, 'entity_loc': [start_in_text, end_in_text]})
                            else:
                                predict_results.setdefault(class_id, set()).add(entity_text)
        else:
            model_output = model_output.tolist()
            if self.configs['model_type'] == 'ptm':
                start_mapping = {i: j[0] for i, j in enumerate(token2char_span_mapping)}
                end_mapping = {i: j[-1] - 1 for i, j in enumerate(token2char_span_mapping)}
                model_output = model_output[:len(token2char_span_mapping)]
            else:
                if 'ptm' in self.configs['model_type']:
                    model_output = model_output[:len(token2char_span_mapping) - 2]
            predict_label = [str(self.sequence_tag_reverse_categories[int(lab)]) for lab in model_output]
            start, end = 0, 0

            while end < len(predict_label):
                if predict_label[start] in self.sequence_tag_classes:
                    if predict_label[start] == 'O':
                        start = start + 1
                        end = end + 1
                        continue
                    if re.findall(r'^B-', predict_label[start]):
                        entity_type = re.split(r'^B-', predict_label[start])[-1]
                        end = start + 1
                        if end != len(predict_label):
                            while re.findall(r'^I-' + entity_type, predict_label[end]):
                                end = end + 1
                                if end == len(predict_label):
                                    break
                            if self.configs['model_type'] == 'ptm':
                                entity = text[start_mapping[start]: end_mapping[end - 1] + 1]
                                entity_loc = [start_mapping[start], end_mapping[end - 1]]
                            else:
                                if 'ptm' in self.configs['model_type']:
                                    entity = text[start_mapping[start + 1]: end_mapping[end] + 1]
                                    entity_loc = [start_mapping[start + 1], end_mapping[end]]
                                else:
                                    entity = text[start: end]
                                    entity_loc = [start, end - 1]
                            if inference:
                                predict_results.setdefault(self.span_categories[entity_type], []).append(
                                    {'entity': entity, 'entity_loc': entity_loc})
                            else:
                                predict_results.setdefault(self.span_categories[entity_type], set()).add(entity)
                        start = end
                        continue
                start = start + 1
                end = end + 1
        return predict_results

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成token
        :param sentence:
        :return:
        """
        if 'ptm' in self.configs['model_type']:
            token_results = self.tokenizer(sentence)
            token_results = token_results.get('input_ids')
            token_ids = torch.unsqueeze(torch.LongTensor(token_results), 0)
        else:
            token_results = self.tokenizer_for_sentences(sentence)
            token_ids = torch.unsqueeze(torch.LongTensor(token_results), 0)
        return token_ids
