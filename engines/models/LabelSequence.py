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


class IDCNN(nn.Module):
    def __init__(self, embedding_dim, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {'dilation': 1},
            {'dilation': 1},
            {'dilation': 2}]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(256) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(256) for _ in range(num_block)])
        filter_nums = configure['filter_nums']
        for i in range(len(self.layers)):
            dilation = self.layers[i]['dilation']
            single_block = nn.Conv1d(in_channels=filter_nums,
                                     out_channels=filter_nums,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module('layer%d' % i, single_block)
            net.add_module('relu', nn.ReLU())
            net.add_module('layernorm', norms_1[i])

        self.linear = nn.Linear(embedding_dim, filter_nums)
        self.idcnn = nn.Sequential()

        for i in range(num_block):
            self.idcnn.add_module('block%i' % i, net)
            self.idcnn.add_module('relu', nn.ReLU())
            self.idcnn.add_module('layernorm', norms_2[i])

    def forward(self, inputs):
        output = self.idcnn(inputs)
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


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
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if 'bilstm' in configure['model_type']:
            self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        elif 'idcnn' in configure['model_type']:
            self.idcnn = IDCNN(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids):
        input_mask = torch.where(input_ids > 0, 1, 0)
        if 'ptm' in configure['model_type']:
            output = self.ptm_model(input_ids, attention_mask=input_mask)[0]
        else:
            output = self.word_embeddings(input_ids)

        if 'bilstm' in configure['model_type']:
            output, _ = self.rnn(output)

        elif 'idcnn' in configure['model_type']:
            output = self.idcnn(output).permute(0, 2, 1)

        if self.multisample_dropout and configure['dropout_round'] > 1:
            dropout_round = configure['dropout_round']
            output = torch.mean(torch.stack([self.fc(
                self.dropout(output)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            dropout_output = self.dropout(output)
            output = self.fc(dropout_output)
        return output


