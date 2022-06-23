from module.gatlayers import GraphAttentionLayer, SpGraphAttentionLayer, get_dim_act
import torch.nn as nn
import torch
import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F


class GAT(nn.Module):
    # Graph Attention Networks.

    def __init__(self, ent1, rel1, ent2, rel2, num_layers=2, n_heads=1, dropout=0.1, alpha=1e-2, use_gpu=True):
        super(GAT, self).__init__()
        assert num_layers > 0
        # dims, acts = get_dim_act(args)
        gat_layers = []
        self.wt = []
        self.bt = []
        dims = [200, 200, 200]
        activation = torch.sigmoid
        acts = [activation] * 3
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            out_dim = dims[i + 1]
            concat = False
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, dropout, act, alpha, n_heads, concat))
        self.W = nn.Parameter(torch.zeros(size=(200, 200)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, 200)))
        nn.init.xavier_normal_(self.b.data, gain=1.414)
        for i in range(len(dims) - 1):
            self.wt.append(self.W)
            self.bt.append(self.b)
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True
        self.ent1 = ent1
        self.ent2 = ent2
        self.rel1 = rel1
        self.rel2 = rel2
        self.output = None
        # self.adj1 = self.get_adj("./data/FB15K237/")
        # self.adj2 = self.get_adj("./data/WN18/")

        # self.ent_embed_1 = nn.Parameter(torch.randn((ent1, dims[0])))
        # self.ent_embed_2 = nn.Parameter(torch.randn((ent2, dims[0])))
        self.ent_embed_1 = nn.Parameter(self.get_pretrained_embed('./data/FB15K237/'))
        self.ent_embed_2 = nn.Parameter(self.get_pretrained_embed('./data/WN18RR/'))
        self.rel_embed_1 = nn.Parameter(torch.randn((rel1, dims[0])))
        self.rel_embed_2 = nn.Parameter(torch.randn((rel2, dims[0])))


    def get_pretrained_embed(self, path):
        res = []
        with open(os.path.join(path, 'sentence.vec'), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = line[1:-1]
                line = line.replace("'", "").replace(" ", "").split(',')
                line = list(map(float, line))
                res.append(line)
        res = torch.tensor(res)
        return res

    # too big
    def get_adj(self, path):
        files = ["train2id.txt", "test2id.txt", "valid2id.txt"]
        indices = [[], []]
        values = []
        max_h = 0
        max_t = 0
        # max_r = 0
        for file in files:
            with open(os.path.join(path, file)) as f:
                cnt = (int)(f.readline())
                for i in range(cnt):
                    h, t, r = f.readline().strip().split(' ')
                    h, t, r = (int)(h), (int)(t), (int)(r)
                    indices[0].append(h)
                    indices[1].append(t)
                    values.append(r)
                    max_h = max(max_h, h)
                    max_t = max(max_t, t)
                    # max_r = max(max_r, r)
        indices = torch.LongTensor(indices)
        values = torch.LongTensor(values)
        m = max(max_h, max_t) + 1
        # print(max_r)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size([m, m]))
        return adj


    def forward(self, data, bflag):
        batch_h = data['batch_h']
        batch_r = data['batch_r']
        batch_t = data['batch_t']
        # 每个epoch只训练一次
        if bflag == 0:
            if data['dflag'] == 1:
                r = self.rel_embed_1[batch_r]
            else:
                r = self.rel_embed_2[batch_r]
            h = self.output[batch_h]
            t = self.output[batch_t]
            return h, r, t
        # bflag == -1:test
        if bflag == -1:
            if self.output is not None:
                if data['dflag'] == 1:
                    r = self.rel_embed_1[batch_r]
                else:
                    r = self.rel_embed_2[batch_r]
                h = self.output[batch_h]
                t = self.output[batch_t]
                return h, r, t
        if data['dflag'] == 1:
            input = (self.ent_embed_1, self.rel_embed_1, self.get_adj("./data/FB15K237/"))
            r = self.rel_embed_1[batch_r]
        else:
            input = (self.ent_embed_2, self.rel_embed_2, self.get_adj("./data/WN18RR/"))
            r = self.rel_embed_2[batch_r]
        # self.output, _, _ = self.layers.forward(input)
        for i in range(len(self.layers)):
            output, a, b = self.layers[i](input)
            sigma = torch.sigmoid(torch.mm(input[0], self.wt[i]) + self.bt[i])
            output = sigma*output + (1-sigma)*input[0]
            input = (output, a, b)

        self.output = output
        h = self.output[batch_h]
        t = self.output[batch_t]
        return h, r, t