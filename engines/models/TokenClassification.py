# -*- coding: utf-8 -*-
# @Time : 2022/12/2 6:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : TokenClassification.py
# @Software: VScode
import torch
from config import configure
from transformers import AutoModelForTokenClassification


class TokenClassification(torch.nn.Module):
    def __init__(self, num_labels):
        super(TokenClassification, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(configure['ptm'])

    def forward(self, input_ids):
        attention_mask = torch.where(input_ids > 0, 1, 0)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        predicted_token_class_ids = torch.argmax(logits, -1)
        return logits, predicted_token_class_ids
