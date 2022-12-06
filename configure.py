# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : configure.py
# @Software: PyCharm

# 模式
# train:训练分类器
# interactive_predict:交互模式
# test:跑测试集
# convert2tf:将torch模型保存为tf框架的pb格式文件
# [train, interactive_predict, test, convert_onnx]
mode = 'train'

# 使用GPU设备
use_cuda = True
cuda_device = -1


configure = {
    # 训练数据集
    'train_file': 'data/example_datasets/dev_data.csv',
    # 验证数据集
    'dev_file': '',
    # 没有验证集时，从训练集抽取验证集比例
    'validation_rate': 0.15,
    # 测试数据集
    'test_file': '',
    # 存放词表的地方
    'token_file': 'data/example_datasets/token2id.txt',
    # 使用的预训练模型
    'ptm': 'bert-base-chinese',
    # 使用的方法
    # sequence label:序列标注
    # span:方式
    'method': 'sequence_label',
    # 使用的模型
    # sequence label方式:
    # ptm crf: ptm_crf
    # ptm bilstm crf: ptm_bilstm_crf
    # ptm idcnn crf: ptm_idcnn_crf
    # idcnn crf: idcnn_crf
    # bilstm crf: bilstm_crf
    # span方式:
    # binary pointer: ptm_bp
    # global pointer: ptm_gp
    'model_type': 'ptm_idcnn_crf',
    # 选择lstm时，隐藏层大小
    'hidden_dim': 200,
    # Embedding向量维度
    'embedding_dim': 300,
    # 选择idcnn时filter的个数
    'filter_nums': 64,
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints/example_datasets',
    # 模型名字
    'model_name': 'best_model.pkl',
    # 类别列表
    'classes': ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'O'],
    # decision_threshold
    'decision_threshold': 0.5,
    # 是否使用苏神的多标签分类的损失函数，默认使用BCELoss
    'use_multilabel_categorical_cross_entropy': True,
    # 使用对抗学习
    'use_gan': False,
    # 目前支持FGM和PGD两种方法
    # fgm:Fast Gradient Method
    # pgd:Projected Gradient Descent
    'gan_method': 'pgd',
    # 对抗次数
    'attack_round': 3,
    # 使用Multisample Dropout
    # 使用Multisample Dropout后dropout会失效
    'multisample_dropout': False,
    'dropout_round': 5,
    # 是否进行warmup
    'warmup': False,
    # warmup方法，可选：linear、cosine
    'scheduler_type': 'linear',
    # warmup步数，-1自动推断为总步数的0.1
    'num_warmup_steps': -1,
    # 句子最大长度
    'max_sequence_length': 200,
    # epoch
    'epoch': 50,
    # batch_size
    'batch_size': 16,
    # dropout rate
    'dropout_rate': 0.5,
    # 每print_per_batch打印损失函数
    'print_per_batch': 100,
    # learning_rate
    'learning_rate': 5e-5,
    # 优化器选择
    'optimizer': 'AdamW',
    # 训练是否提前结束微调
    'is_early_stop': True,
    # 训练阶段的patient
    'patient': 5,
}
