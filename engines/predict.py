# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import torch
import os
import time
import json
import pandas as pd
from torch.utils.data import DataLoader


class Predictor:
    def __init__(self, configs, data_manager, device, logger):
        self.device = device
        self.configs = configs
        self.data_manager = data_manager
        self.logger = logger
        self.checkpoints_dir = configs['checkpoints_dir']
        self.model_name = configs['model_name']
        if self.configs['model_type'].lower() == 'ptm_bp':
            from engines.models.BinaryPointer import BinaryPointer
            self.model = BinaryPointer(num_labels=self.data_manager.span_num_labels).to(self.device)
        elif self.configs['model_type'].lower() == 'ptm_gp':
            from engines.models.GlobalPointer import EffiGlobalPointer
            self.model = EffiGlobalPointer(num_labels=self.data_manager.span_num_labels,
                                           device=self.device).to(self.device)
        else:
            from engines.models.SequenceTag import SequenceTag
            self.model = SequenceTag(vocab_size=self.data_manager.vocab_size,
                                num_labels=self.data_manager.sequence_tag_num_labels).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
        self.model.eval()

    def predict_one(self, sentence):
        """
        预测接口
        """
        start_time = time.time()
        token_ids = self.data_manager.prepare_single_sentence(sentence).to(self.device)
        if self.configs['method'] == 'sequence_tag':
            results = torch.squeeze(self.model(token_ids))
        else:
            logits, _ = self.model(token_ids)
            results = torch.squeeze(logits.to('cpu'))
        predict_results = self.data_manager.extract_entities(sentence, results)
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        results_dict = {}
        for class_id, result_set in predict_results.items():
            results_dict[self.data_manager.span_reverse_categories[class_id]] = list(result_set)
        return results_dict

    def predict_test(self):
        test_file = self.configs['test_file']
        if test_file == '' or not os.path.exists(test_file):
            self.logger.info('test dataset does not exist!')
            return
        file_format = test_file.split('.')[-1]
        if file_format == 'json':
            test_data = json.load(open(test_file, encoding='utf-8'))
        elif file_format == 'csv':
            test_data = pd.read_csv(test_file, names=['token', 'label'], sep=' ', skip_blank_lines=False)
            test_data = self.data_manager.csv_to_json(test_data)
        else:
            self.logger.info('data format error!')
            return

        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.data_manager.batch_size,
            collate_fn=self.data_manager.prepare_data,
        )
        from engines.train import Train
        train = Train(self.configs, self.data_manager, self.device, self.logger)
        train.validate(self.model, test_loader)

    def convert_onnx(self):
        max_sequence_length = self.data_manager.max_sequence_length
        dummy_input = torch.ones([1, max_sequence_length]).to('cpu').int()
        onnx_path = self.checkpoints_dir + '/model.onnx'
        if self.configs['method'] == 'sequence_tag':
            torch.onnx.export(self.model.to('cpu'), dummy_input, f=onnx_path, opset_version=13,
                              input_names=['tokens'], output_names=['decode'],
                              do_constant_folding=False,
                              dynamic_axes={'tokens': {0: 'batch_size'}, 'decode': {0: 'decode'}})
        else:
            torch.onnx.export(self.model.to('cpu'), dummy_input, f=onnx_path, opset_version=13,
                              input_names=['tokens'], output_names=['logits', 'probs'],
                              do_constant_folding=False,
                              dynamic_axes={'tokens': {0: 'batch_size'}, 'logits': {0: 'batch_size'},
                                            'probs': {0: 'batch_size'}})
        self.logger.info('convert torch to onnx successful...')

    def show_model_info(self):
        import textpruner
        info = textpruner.summary(self.model, max_level=3)
        self.logger.info(info)
