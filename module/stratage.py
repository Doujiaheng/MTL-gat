
import torch
import torch.nn as nn
import os
import json
import numpy as np
import torch.nn.functional as F
from .basemodel import BaseModule, Model


class Strategy(BaseModule):

	def __init__(self):
		super(Strategy, self).__init__()


class NegativeSampling(Strategy):

    def __init__(self, model=None, loss = None, batch_size1 = 256, batch_size2 = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
        super(NegativeSampling, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size1 = batch_size1
        self.batch_size2 = batch_size2
        self.regul_rate = regul_rate
        self.l3_regul_rate = l3_regul_rate

    def _get_positive_score(self, score, dflag):
        if dflag == 1:
            positive_score = score[:self.batch_size1]
            positive_score = positive_score.view(-1, self.batch_size1).permute(1, 0)
        else:
            positive_score = score[:self.batch_size2]
            positive_score = positive_score.view(-1, self.batch_size2).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score, dflag):
        if dflag == 1:
            negative_score = score[self.batch_size1:]
            negative_score = negative_score.view(-1, self.batch_size1).permute(1, 0)
        else:
            negative_score = score[self.batch_size2:]
            negative_score = negative_score.view(-1, self.batch_size2).permute(1, 0)
        return negative_score

    def forward(self, data):
        score = self.model(data)
        p_score = self._get_positive_score(score, data['dflag'])
        n_score = self._get_negative_score(score, data['dflag'])
        # (1000,1), (1000,25)
        loss_res = self.loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.model.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.model.l3_regularization()
        return loss_res
