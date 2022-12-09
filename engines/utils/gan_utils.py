# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : Stanley
# @EMail : gzlishouxian@corp.netease.com
# @File : gan_utils.py
# @Software: PyCharm
import torch


class Adversarial(object):
    def __init__(self, model, eps=0.2):
        self.model = model
        self.eps = eps
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='word_embeddings.'):
        pass

    def restore(self, emb_name='word_embeddings.'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.emb_backup
                para.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FGSM(Adversarial):
    def __init__(self, model, eps=0.2):
        super(FGSM, self).__init__(model, eps)

    def attack(self, emb_name='word_embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.emb_backup[name] = param.data.clone()
                # FGSM对抗扰动
                r_hat = self.eps * param.grad.sign()
                param.data.add_(r_hat)


class FGM(Adversarial):
    def __init__(self, model, eps=0.2):
        super(FGM, self).__init__(model, eps)

    def attack(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_hat = self.eps * param.grad / norm
                    param.data.add_(r_hat)


class PGD(Adversarial):
    def __init__(self, model, eps=1.0, alpha=0.3):
        super(PGD, self).__init__(model, eps)
        self.alpha = alpha

    def attack(self, emb_name='word_embeddings.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r


class FreeLB(Adversarial):

    def __init__(self, model):
        super(FreeLB, self).__init__(model)
        self.adv_lr = 1e-4
        self.adv_max_norm = 1e-4
        self.adv_init_mag = 2e-2
        self.adv_norm_type = 'l2'
        self.gradient_accumulation_steps = 1

    def get_embeddings(self, inputs):
        return getattr(self.model, 'model').embeddings.word_embeddings(inputs)

    def init_delta(self, inputs):
        embeds_init = self.get_embeddings(inputs)
        if self.adv_init_mag > 0:
            input_mask = torch.where(inputs > 0, 1, 0)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == 'linf':
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
            else:
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
        else:
            delta = torch.zeros_like(embeds_init)
        return delta


class AWP:
    """
    Implements weighted adversarial perturbation
    adv_param (str): 要攻击的layer name，一般攻击第一层 或者全部weight参数效果较好
    adv_lr (float): 攻击步长，这个参数相对难调节，如果只攻击第一层embedding，一般用1比较好，全部参数用0.1比较好。
    adv_eps (float): 参数扰动最大幅度限制，范围（0~+∞），一般设置（0，1）之间相对合理一点。
    start_epoch (int): （0~+∞）什么时候开始扰动，默认是0，如果效果不好可以调节值模型收敛一半的时候再开始攻击。
    """

    def __init__(self, model, adv_param='weight'):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = 1
        self.adv_eps = 1e-4
        self.awp_start = 2
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self):
        if self.adv_lr == 0:
            return
        self.save()
        self.attack_step()

    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1])

    def save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
