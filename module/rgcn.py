import torch
import torch.nn as nn
from .basemodel import BaseModule, Model
import math

class VRGCN(Model):

    def __init__(self, ent_tot, rel_tot, dim=100):
        super(VRGCN, self).__init__(ent_tot, rel_tot)
        self.dim = dim
        print('The size of the tot embedding of VRGCN is:', ent_tot, dim)
        print('The size of the rel embedding of VRGCN is:', rel_tot, dim)
        self.ent_embedding = nn.Embedding(ent_tot, dim).cuda()
        self.rel_embedding = nn.Embedding(rel_tot, dim).cuda()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.reset_parameters()
        self.h_rt1, self.t_rh1 = self.get_neighbor("./data/FB15K237/", 1)
        self.h_rt2, self.t_rh2 = self.get_neighbor("./data/WN18/", 2)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def get_neighbor(self, path, dflag):
        base_path =path
        file_set = ["train2id.txt", "valid2id.txt", "test2id.txt"]

        h_rt = {}
        t_rh = {}
        for file_name in file_set:
            triple = open(base_path+file_name, "r")
            tot = (int)(triple.readline())
            for i in range(tot):
                content = triple.readline()
                h, t, r = content.strip().split()
                if dflag != 1:
                    h = (int)(h) + 14541
                    t = (int)(t) + 14541
                    r = (int)(r) + 237
                else:
                    h = (int)(h)
                    t = (int)(t)
                    r = (int)(r)
                if not h in h_rt:
                    h_rt[h] = [[], []]
                if not t in t_rh:
                    t_rh[t] = [[], []]

                h_rt[h][0].append(r)
                h_rt[h][1].append(t)
                t_rh[t][0].append(r)
                t_rh[t][1].append(h)
            return h_rt, t_rh

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        dflag = data['dflag']
        if dflag != 1:
            batch_h = batch_h + 14541
            batch_r = batch_r + 237
            batch_t = batch_t + 14541
            h_rt, t_rh = self.h_rt2, self.t_rh2
        else:
            h_rt, t_rh = self.h_rt1, self.t_rh1
        # (26000,100)
        h = self.ent_embedding(batch_h)
        r = self.rel_embedding(batch_r)
        t = self.ent_embedding(batch_t)
        hhrt = None
        for i in range(len(batch_h)):
            bh = batch_h[i]
            hrt = torch.zeros(self.dim).cuda()
            d = 0
            bhi = bh.item()
            if bhi in h_rt:
                d = d + len(h_rt[bhi][0])
                rr = torch.LongTensor(h_rt[bhi][0]).cuda()
                tt = torch.LongTensor(h_rt[bhi][1]).cuda()
                rr = self.rel_embedding(rr)
                tt = self.ent_embedding(tt)
                rr = rr.sum(0)
                tt = tt.sum(0)
                hrt = hrt + tt - rr
            if bhi in t_rh:
                d = d + len(t_rh[bhi])
                rr = torch.LongTensor(t_rh[bhi][0]).cuda()
                hh = torch.LongTensor(t_rh[bhi][1]).cuda()
                rr = self.rel_embedding(rr)
                hh = self.ent_embedding(hh)
                rr = rr.sum(0)
                hh = hh.sum(0)
                hrt = hrt + hh + rr
            if d != 0:
                hrt = hrt / d
            hrt = hrt.unsqueeze(0)
            if hhrt is None:
                hhrt = hrt
            else:
                hhrt = torch.cat((hhrt, hrt), 0)
        h = h + hhrt
        h = torch.mm(h, self.weight)
        h = torch.sigmoid(h)

        hhrt = None
        for i in range(len(batch_t)):
            bt = batch_t[i]
            hrt = torch.zeros(self.dim).cuda()
            d = 0
            bti = bt.item()
            if bti in h_rt:
                d = d + len(h_rt[bti])
                rr = torch.LongTensor(h_rt[bti][0]).cuda()
                tt = torch.LongTensor(h_rt[bti][1]).cuda()
                rr = self.rel_embedding(rr)
                tt = self.ent_embedding(tt)
                rr = rr.sum(0)
                tt = tt.sum(0)
                hrt = hrt + tt - rr
            if bti in t_rh:
                d = d + len(t_rh[bti])
                # for rh in t_rh[bt]:
                #     hrt = hrt + self.ent_embedding(rh[1]) + self.rel_embedding(rh[0])
                rr = torch.LongTensor(t_rh[bti][0]).cuda()
                hh = torch.LongTensor(t_rh[bti][1]).cuda()
                rr = self.rel_embedding(rr)
                hh = self.ent_embedding(hh)
                rr = rr.sum(0)
                hh = hh.sum(0)
                hrt = hrt + hh + rr
            if d != 0:
                hrt = hrt / d
            hrt = hrt.unsqueeze(0)
            if hhrt is None:
                hhrt = hrt
            else:
                hhrt = torch.cat([hhrt, hrt], 0)
        t = t + hhrt
        t = torch.mm(t, self.weight)
        t = torch.sigmoid(t)

        return h, r, t

    def predict(self, data):

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        dflag = data['dflag']
        if dflag != 1:
            batch_h = batch_h + 14541
            batch_r = batch_r + 237
            batch_t = batch_t + 14541
        h = self.ent_embedding(batch_h)
        r = self.rel_embedding(batch_r)
        t = self.ent_embedding(batch_t)
        h = torch.mm(h, self.weight)
        h = torch.sigmoid(h)
        t = torch.mm(t, self.weight)
        t = torch.sigmoid(t)
        return h, r, t



