# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py
# @Software: PyCharm
from abc import ABC

import torch
from torch import nn
from configure import configure
from transformers import BertModel


class BinaryPointer(nn.Module, ABC):
    def __init__(self, num_labels):
        super(BinaryPointer, self).__init__()
        self.num_labels = num_labels
        self.ptm_model = BertModel.from_pretrained(configure['ptm'])
        hidden_size = self.bert_model.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.fc = nn.Linear(hidden_size, 2 * num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        input_mask = torch.where(input_ids > 0, 1, 0)
        bert_hidden_states = self.ptm_model(input_ids, attention_mask=input_mask)[0]
        layer_hidden = self.layer_norm(bert_hidden_states)
        fc_results = self.fc(layer_hidden)
        batch_size = fc_results.size(0)
        logits = fc_results.view(batch_size, -1, self.num_labels, 2)
        probs = torch.sigmoid(logits)
        return logits, probs
