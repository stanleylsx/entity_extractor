# -*- coding: utf-8 -*-
# @Time : 2022/12/2 6:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : LabelSequence.py
# @Software: PyCharm
from abc import ABC

import torch
from transformers import BertModel
from torch import nn
from configure import configure
from torchcrf import CRF


class IDCNN(nn.Module):
    def __init__(self, filter_nums, embedding_dim, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {'dilation': 1},
            {'dilation': 1},
            {'dilation': 2}]
        net = nn.Sequential()
        for i in range(len(self.layers)):
            dilation = self.layers[i]['dilation']
            single_block = nn.Conv1d(in_channels=filter_nums,
                                     out_channels=filter_nums,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module('layer%d' % i, single_block)
            net.add_module('relu', nn.ReLU())

        self.linear = nn.Linear(embedding_dim, filter_nums)
        self.idcnn = nn.Sequential()

        for i in range(num_block):
            self.idcnn.add_module('block%i' % i, net)
            self.idcnn.add_module('relu', nn.ReLU())

    def forward(self, inputs):
        inputs = self.linear(inputs)
        inputs = inputs.permute(0, 2, 1)
        output = self.idcnn(inputs)
        return output


class LabelSequence(nn.Module, ABC):
    def __init__(self, vocab_size, num_labels):
        super(LabelSequence, self).__init__()
        # ptm crf: ptm_crf
        # ptm bilstm crf: ptm_bilstm_crf
        # ptm idcnn crf: ptm_idcnn_crf
        # idcnn crf: idcnn_crf
        # bilstm crf: bilstm_crf
        embedding_dim = configure['embedding_dim']
        hidden_dim = configure['hidden_dim']
        dropout_rate = configure['dropout_rate']
        self.multisample_dropout = configure['multisample_dropout']
        if 'ptm' in configure['model_type']:
            self.ptm_model = BertModel.from_pretrained(configure['ptm'])
            embedding_dim = self.ptm_model.config.hidden_size
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if 'bilstm' in configure['model_type']:
            self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
            hidden_dim = 2 * hidden_dim
        elif 'idcnn' in configure['model_type']:
            filter_nums = configure['filter_nums']
            self.idcnn = IDCNN(filter_nums, embedding_dim)
            self.liner = nn.Linear(filter_nums, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, labels=None):
        input_mask = torch.where(input_ids > 0, True, False)
        if 'ptm' in configure['model_type']:
            output = self.ptm_model(input_ids, attention_mask=input_mask)[0]
        else:
            output = self.word_embeddings(input_ids)

        if 'bilstm' in configure['model_type']:
            output, _ = self.bilstm(output)

        elif 'idcnn' in configure['model_type']:
            output = self.idcnn(output).permute(0, 2, 1)
            output = self.liner(output)

        if self.multisample_dropout and configure['dropout_round'] > 1:
            dropout_round = configure['dropout_round']
            logits = torch.mean(torch.stack([self.fc(
                self.dropout(output)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            dropout_output = self.dropout(output)
            logits = self.fc(dropout_output)
        if labels is not None:
            loss = -self.crf(emissions=logits, tags=labels, mask=input_mask)
            return loss
        else:
            decode = self.crf.decode(emissions=logits, mask=input_mask)
            return decode
