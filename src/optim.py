import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf


def get_param(model, lr, wd, lr_min):
    param_groups = [{'params': [], 'lr_max': lr,   'lr_min': lr_min,   'wd_max': 0},     # 0:bias
                    {'params': [], 'lr_max': lr,   'lr_min': lr_min,   'wd_max': 0},     # 1:scale
                    {'params': [], 'lr_max': lr,   'lr_min': lr_min,   'wd_max': wd},    # 2:weight
                    {'params': [], 'lr_max': lr/2, 'lr_min': lr_min/2, 'wd_max': wd*2}]  # 3:head

    for n, p in model.named_parameters():
        if n.find('_encoder') > 0: param_groups[0]['params'].append(p)
        elif n.endswith('bias'):   param_groups[0]['params'].append(p)
        elif n.endswith('scale'):  param_groups[1]['params'].append(p)
        elif n.find('head') > 0:   param_groups[3]['params'].append(p)
        elif n.endswith('weight'): param_groups[2]['params'].append(p)
        else: raise Exception('Unknown parameter name:', n)
        for pg in param_groups: assert len(pg) > 0

    return param_groups

def clamp_param(param, eps=1e-2):
    with pt.no_grad():
        for p in param[1]['params']:
            p.clamp_(eps, 1)

# weight decay: https://arxiv.org/abs/2011.11152
class Scheduler(object):
    def __init__(self, optim, lr_warmup=12, cos_period=12):
        super().__init__()
        self.optim = optim
        self.lr_decay = (5 ** 0.5 - 1) / 2
        self.lr_warmup = lr_warmup
        self.cos_period = cos_period

    def step(self, epoch, eps):
        if epoch == 0:
            for pg in self.optim.param_groups:
                pg['lr'] = eps
                pg['weight_decay'] = eps
        elif epoch <= self.lr_warmup:
            for pg in self.optim.param_groups:
                pg['lr'] = pg['lr_max'] / self.lr_warmup * epoch
                pg['weight_decay'] = pg['wd_max'] / self.lr_warmup * epoch
        else:
            i = (epoch - self.lr_warmup) // self.cos_period
            j = (epoch - self.lr_warmup) % self.cos_period
            for pg in self.optim.param_groups:
                lr_max = max(pg['lr_max'] * self.lr_decay ** i, pg['lr_min'])
                lr_cos = np.cos(j / (self.cos_period - 1) * np.pi / 2) * self.lr_decay + (1 - self.lr_decay)
                pg['lr'] = lr_max * lr_cos
                pg['weight_decay'] = pg['wd_max']
        return self.optim.param_groups[-2]['lr'], self.optim.param_groups[-2]['weight_decay']

