import torch
import torch.nn as nn
import os
import json
import numpy as np
import torch.nn.functional as F
from .basemodel import BaseModule, Model
import math
from .transe import TransE
from torch.autograd import Variable
import time


class TranseRgat(BaseModule):

    def __init__(self, transe1, transe2, rgat, dim_trans,  dim_rgat, p_norm=1, norm_flag=True, margin=None):
        super(TranseRgat, self).__init__()
        self.transe1 = transe1
        self.transe2 = transe2
        self.rgat = rgat
        self.p_norm = p_norm
        self.dim_trans = dim_trans
        self.dim_rgat = dim_rgat
        self.norm_flag= norm_flag
        self.mh = nn.Parameter(torch.Tensor(dim_trans, dim_rgat))
        self.mr = nn.Parameter(torch.Tensor(dim_trans, dim_rgat))
        self.mt = nn.Parameter(torch.Tensor(dim_trans, dim_rgat))
        self.set_parameter()
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def set_parameter(self):
        stdv = 1. / math.sqrt(self.mh.size(1))
        self.mh.data.uniform_(-stdv, stdv)
        self.mr.data.uniform_(-stdv, stdv)
        self.mt.data.uniform_(-stdv, stdv)

    def forward(self, data):

        # dflag == -1： test
        if data['dflag'] == 1:
            # train model1
            he, re, te = self.transe1(data)
            hs, rs, ts = self.rgat(data, data['bflag'])
            mode = data['mode']

        elif data['dflag'] == -1:
            # test model1
            self.transe1.train(False)
            self.rgat.train(False)
            he, re, te = self.transe1(data)
            data['dflag'] = 1
            hs, rs, ts = self.rgat(data, -1)
            mode = data['mode']
        elif data['dflag'] == -2:
            # test model2
            self.transe2.train(False)
            self.rgat.train(False)
            he, re, te = self.transe2(data)
            data['dflag'] = 2
            hs, rs, ts = self.rgat(data, -1)
            mode = data['mode']
        else:
            # train model2
            he, re, te = self.transe2(data)
            hs, rs, ts = self.rgat(data, data['bflag'])
            mode = data['mode']

        h = he + torch.mm(hs, self.mh)
        r = re + torch.mm(rs, self.mr)
        t = te + torch.mm(ts, self.mt)
        # (26000, 200)

        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        # score (26000, 200)
        score = torch.norm(score, self.p_norm, -1).flatten()
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        if data['dflag'] == 1:
            h = self.transe1.ent_embeddings(batch_h)
            t = self.transe1.ent_embeddings(batch_t)
            r = self.transe1.rel_embeddings(batch_r)
        else :
            h = self.transe2.ent_embeddings(batch_h)
            t = self.transe2.ent_embeddings(batch_t)
            r = self.transe2.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):

        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()


