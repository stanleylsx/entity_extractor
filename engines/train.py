# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import json
import torch
import time
import pandas as pd
import os


class Train:
    def __init__(self, configs, data_manager, device, logger):
        self.configs = configs
        self.device = device
        self.logger = logger
        self.data_manager = data_manager
        self.batch_size = self.configs['batch_size']
        self.checkpoints_dir = configs['checkpoints_dir']
        self.model_name = configs['model_name']
        self.epoch = configs['epoch']
        self.learning_rate = configs['learning_rate']
        self.kfold = configs['kfold']
        self.optimizer = None
        self.gan = None

        if self.configs['method'] == 'span':
            if configs['use_multilabel_categorical_cross_entropy']:
                from engines.utils.losses import MultilabelCategoricalCrossEntropy
                self.loss_function = MultilabelCategoricalCrossEntropy()
            else:
                self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_function = torch.nn.CrossEntropyLoss()

    def calculate_loss(self, logits, labels, attention_mask):
        batch_size = logits.size(0)
        if self.configs['method'] == 'span':
            if self.configs['use_multilabel_categorical_cross_entropy']:
                if self.configs['model_type'] == 'ptm_bp':
                    num_labels = self.data_manager.span_num_labels * 2
                else:
                    num_labels = self.data_manager.span_num_labels
                model_output = logits.reshape(batch_size * num_labels, -1)
                label_vectors = labels.reshape(batch_size * num_labels, -1)
                loss = self.loss_function(model_output, label_vectors)
            else:
                if self.configs['model_type'] == 'ptm_bp':
                    loss = self.loss_function(logits, labels)
                    loss = torch.sum(torch.mean(loss, 3), 2)
                    loss = torch.sum(loss * attention_mask) / torch.sum(attention_mask)
                else:
                    model_output = logits.reshape(batch_size * self.data_manager.span_num_labels, -1)
                    label_vectors = labels.reshape(batch_size * self.data_manager.span_num_labels, -1)
                    loss = self.loss_function(model_output, label_vectors).mean()
        else:
            logits = logits.permute(0, 2, 1)
            loss = self.loss_function(logits, labels)
        return loss

    def init_model(self):
        if self.configs['model_type'].lower() == 'ptm_bp':
            from engines.models.BinaryPointer import BinaryPointer
            model = BinaryPointer(num_labels=self.data_manager.span_num_labels).to(self.device)
        elif self.configs['model_type'].lower() == 'ptm_gp':
            from engines.models.GlobalPointer import EffiGlobalPointer
            model = EffiGlobalPointer(num_labels=self.data_manager.span_num_labels, device=self.device).to(self.device)
        elif self.configs['model_type'].lower() == 'ptm':
            from engines.models.TokenClassification import TokenClassification
            model = TokenClassification(num_labels=self.data_manager.sequence_tag_num_labels).to(self.device)
        else:
            from engines.models.SequenceTagCRF import SequenceTagCRF
            model = SequenceTagCRF(vocab_size=self.data_manager.vocab_size,
                                num_labels=self.data_manager.sequence_tag_num_labels).to(self.device)

        if 'ptm' in self.configs['model_type'] and self.configs['noisy_tune']:
            for name, para in model.named_parameters():
                noise_lambda = self.configs['noise_lambda']
                model.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)

        if 'ptm' not in self.configs['model_type'] and self.configs['init_network']:
            from engines.utils.init_network_parameter import init_network
            model = init_network(model, method=self.configs['init_network_method'])

        if self.configs['use_gan']:
            if self.configs['gan_method'] == 'fgm':
                from engines.utils.gan_utils import FGM
                self.gan = FGM(model)
            elif self.configs['gan_method'] == 'fgsm':
                from engines.utils.gan_utils import FGSM
                self.gan = FGSM(model)
            elif self.configs['gan_method'] == 'pgd':
                from engines.utils.gan_utils import PGD
                self.gan = PGD(model)
            elif self.configs['gan_method'] == 'freelb':
                from engines.utils.gan_utils import FreeLB
                self.gan = FreeLB(model)
            elif self.configs['gan_method'] == 'awp':
                from engines.utils.gan_utils import AWP
                self.gan = AWP(model)
            else:
                self.gan = None

        params = list(model.parameters())
        optimizer_type = self.configs['optimizer']
        if optimizer_type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(params, lr=self.learning_rate)
        elif optimizer_type == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(params, lr=self.learning_rate)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, lr=self.learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        else:
            raise Exception('optimizer_type does not exist')
        return model

    def split_data(self):
        train_file = self.configs['train_file']
        dev_file = self.configs['dev_file']
        train_data, dev_data = None, None
        if self.data_manager.file_format == 'json':
            train_data = json.load(open(train_file, encoding='utf-8'))
            if dev_file != '':
                dev_data = json.load(open(dev_file, encoding='utf-8'))

        elif self.data_manager.file_format == 'csv':
            train_data = pd.read_csv(train_file, names=['token', 'label'], sep=' ', skip_blank_lines=False)
            train_data = self.data_manager.csv_to_json(train_data)
            if dev_file != '':
                dev_data = pd.read_csv(dev_file, names=['token', 'label'], sep=' ', skip_blank_lines=False)
                dev_data = self.data_manager.csv_to_json(dev_data)

        if dev_file == '':
            self.logger.info('generate validation dataset...')
            validation_rate = self.configs['validation_rate']
            ratio = 1 - validation_rate
            train_data, dev_data = train_data[:int(ratio * len(train_data))], train_data[int(ratio * len(train_data)):]

        self.logger.info('loading train data...')
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data,
            shuffle=True
        )
        self.logger.info('loading validation data...')
        dev_loader = DataLoader(
            dataset=dev_data,
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data
        )
        return train_loader, dev_loader

    def input_model(self, model, token_ids, labels):
        if self.configs['method'] == 'sequence_tag':
            if 'crf' in self.configs['model_type']:
                loss = model(token_ids, labels)
            else:
                logits, _ = model(token_ids)
                attention_mask = torch.where(token_ids > 0, 1, 0).to(self.device)
                loss = self.calculate_loss(logits, labels, attention_mask)
        else:
            logits, _ = model(token_ids)
            attention_mask = torch.where(token_ids > 0, 1, 0).to(self.device)
            loss = self.calculate_loss(logits, labels, attention_mask)
        return loss

    def train_each_fold(self, model, train_loader, dev_loader, fold_index=None):
        best_f1 = 0
        best_epoch = 0
        unprocessed = 0
        step_total = self.epoch * len(train_loader)
        global_step = 0
        scheduler = None
        model_name = self.model_name + '_' + str(fold_index) if fold_index else self.model_name

        if self.configs['warmup']:
            scheduler_type = self.configs['scheduler_type']
            if self.configs['num_warmup_steps'] == -1:
                num_warmup_steps = step_total * 0.1
            else:
                num_warmup_steps = self.configs['num_warmup_steps']

            if scheduler_type == 'linear':
                from transformers.optimization import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            elif scheduler_type == 'cosine':
                from transformers.optimization import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            else:
                raise Exception('scheduler_type does not exist')

        if self.configs['ema']:
            from engines.utils.ema import EMA
            ema = EMA(model)
            ema.register()
        else:
            ema = None

        if self.configs['swa']:
            from torch.optim.swa_utils import AveragedModel, SWALR
            model = AveragedModel(model).to(self.device)
            swa_lr = self.configs['swa_lr']
            anneal_epochs = self.configs['anneal_epochs']
            swa_scheduler = SWALR(optimizer=self.optimizer, swa_lr=swa_lr, anneal_epochs=anneal_epochs,
                                  anneal_strategy='linear')
        else:
            swa_scheduler = None

        very_start_time = time.time()
        for i in range(self.epoch):
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epoch))
            model.train()
            start_time = time.time()
            step, loss, loss_sum = 0, 0.0, 0.0
            for batch in tqdm(train_loader):
                _, _, token_ids, label_vectors = batch
                token_ids = token_ids.to(self.device)
                label_vectors = label_vectors.to(self.device)
                self.optimizer.zero_grad()
                loss = self.input_model(model, token_ids, label_vectors)
                loss.backward()
                loss_sum += loss.item()
                if self.configs['use_gan']:
                    k = self.configs['attack_round']
                    if self.configs['gan_method'] in ('fgm', 'fgsm'):
                        self.gan.attack()
                        loss = self.input_model(model, token_ids, label_vectors)
                        loss.backward()
                        self.gan.restore()  # 恢复embedding参数
                    elif self.configs['gan_method'] == 'pgd':
                        self.gan.backup_grad()
                        for t in range(k):
                            self.gan.attack(is_first_attack=(t == 0))
                            if t != k - 1:
                                model.zero_grad()
                            else:
                                self.gan.restore_grad()
                            loss = self.input_model(model, token_ids, label_vectors)
                            loss.backward()
                        self.gan.restore()
                    elif self.configs['gan_method'] == 'awp':
                        if i + 1 >= self.gan.awp_start:
                            self.gan.attack_backward()
                            loss = self.input_model(model, token_ids, label_vectors)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.gan.restore()
                self.optimizer.step()

                if self.configs['ema']:
                    ema.update()

                if self.configs['swa']:
                    if global_step > self.configs['swa_start_step']:
                        model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        if self.configs['warmup']:
                            scheduler.step()

                if self.configs['warmup']:
                    scheduler.step()

                if step % self.configs['print_per_batch'] == 0 and step != 0:
                    avg_loss = loss_sum / (step + 1)
                    self.logger.info('training_loss:%f' % avg_loss)

                step = step + 1
                global_step = global_step + 1

            if self.configs['ema']:
                ema.apply_shadow()

            f1 = self.validate(model, dev_loader)
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)
            if f1 >= best_f1:
                unprocessed = 0
                best_f1 = f1
                best_epoch = i + 1
                if self.configs['swa']:
                    torch.optim.swa_utils.update_bn(train_loader, model, device=self.device)
                if not self.kfold:
                    optimizer_checkpoint = {'optimizer': self.optimizer.state_dict()}
                    torch.save(optimizer_checkpoint, os.path.join(self.checkpoints_dir, model_name + '.optimizer'))
                torch.save(model.state_dict(), os.path.join(self.checkpoints_dir, model_name))
                self.logger.info('saved model successful...')
            else:
                unprocessed += 1
            aver_loss = loss_sum / step
            self.logger.info(
                'aver_loss: %.4f, f1: %.4f, best_f1: %.4f, best_epoch: %d \n' % (aver_loss, f1, best_f1, best_epoch))
            if self.configs['is_early_stop']:
                if unprocessed > self.configs['patient']:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(
                        self.configs['patient']))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % (
                            (time.time() - very_start_time) / 60))
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    def train(self):
        if self.kfold:
            kfold_start_time = time.time()
            train_file = self.configs['train_file']
            dev_file = self.configs['dev_file']
            fold_splits = self.configs['fold_splits']
            train_data, dev_data = [], []
            if self.data_manager.file_format == 'json':
                train_data = json.load(open(train_file, encoding='utf-8'))
                if dev_file != '':
                    dev_data = json.load(open(dev_file, encoding='utf-8'))
                    train_data.extend(dev_data)

            elif self.data_manager.file_format == 'csv':
                train_data = pd.read_csv(train_file, names=['token', 'label'], sep=' ', skip_blank_lines=False)
                train_data = self.data_manager.csv_to_json(train_data)
                if dev_file != '':
                    dev_data = pd.read_csv(dev_file, names=['token', 'label'], sep=' ', skip_blank_lines=False)
                    dev_data = self.data_manager.csv_to_json(dev_data)
                    train_data.extend(dev_data)
            kf = KFold(n_splits=fold_splits, random_state=2, shuffle=True)
            fold = 1
            for train_index, val_index in kf.split(train_data):
                self.logger.info(f'\nTraining fold {fold}...\n')
                model = self.init_model()
                self.optimizer.zero_grad()
                train_data = train_data.loc[train_index]
                val_data = train_data.loc[val_index]
                train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size,
                                          collate_fn=self.data_manager.prepare_data, shuffle=True)
                val_loader = DataLoader(dataset=val_data, batch_size=self.batch_size,
                                        collate_fn=self.data_manager.prepare_data)
                self.train_each_fold(model, train_loader, val_loader, fold_index=fold)
                fold = fold + 1
            self.logger.info('\nKfold: total training time consumption: %.3f(min)' % (
                    (time.time() - kfold_start_time) / 60))
        else:
            model = self.init_model()
            if os.path.exists(os.path.join(self.checkpoints_dir, self.model_name)):
                self.logger.info('resuming from checkpoint...')
                model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
                optimizer_checkpoint = torch.load(os.path.join(self.checkpoints_dir, self.model_name + '.optimizer'))
                self.optimizer.load_state_dict(optimizer_checkpoint['optimizer'])
            else:
                self.logger.info('initializing from scratch.')
            train_loader, dev_loader = self.split_data()
            self.train_each_fold(model, train_loader, dev_loader)

    def validate(self, model, dev_loader):
        counts = {}
        results_of_each_entity = {}
        for class_name, class_id in self.data_manager.span_categories.items():
            counts[class_id] = {'A': 0.0, 'B': 1e-10, 'C': 1e-10}
            class_name = self.data_manager.span_reverse_categories[class_id]
            results_of_each_entity[class_name] = {}

        with torch.no_grad():
            model.eval()
            self.logger.info('start evaluate engines...')
            for batch in tqdm(dev_loader):
                texts, entity_results, token_ids, label_vectors = batch
                token_ids = token_ids.to(self.device)
                if self.configs['method'] == 'sequence_tag':
                    if self.configs['model_type'].lower() == 'ptm':
                        _, results = model(token_ids)
                    else:
                        results = model(token_ids)
                else:
                    logits, _ = model(token_ids)
                    results = logits.to('cpu')
                for text, result, entity_result in zip(texts, results, entity_results):
                    p_results = self.data_manager.extract_entities(text, result)
                    for class_id, entity_set in entity_result.items():
                        p_entity_set = p_results.get(class_id)
                        if p_entity_set is None:
                            # 没预测出来
                            p_entity_set = set()
                        # 预测出来并且正确个数
                        counts[class_id]['A'] += len(p_entity_set & entity_set)
                        # 预测出来的结果个数
                        counts[class_id]['B'] += len(p_entity_set)
                        # 真实的结果个数
                        counts[class_id]['C'] += len(entity_set)
        for class_id, count in counts.items():
            f1, precision, recall = 2 * count['A'] / (
                    count['B'] + count['C']), count['A'] / count['B'], count['A'] / count['C']
            class_name = self.data_manager.span_reverse_categories[class_id]
            results_of_each_entity[class_name]['f1'] = f1
            results_of_each_entity[class_name]['precision'] = precision
            results_of_each_entity[class_name]['recall'] = recall

        f1 = 0.0
        for entity, performance in results_of_each_entity.items():
            f1 += performance['f1']
            # 打印每个类别的指标
            self.logger.info('entity_name: %s, precision: %.4f, recall: %.4f, f1: %.4f'
                        % (entity, performance['precision'], performance['recall'], performance['f1']))
        # 这里算得是所有类别的平均f1值
        f1 = f1 / len(results_of_each_entity)
        return f1
