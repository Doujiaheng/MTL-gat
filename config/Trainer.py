
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
import gc

class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader1=None,
                 data_loader2=None,
                 train_times=1000,
                 alpha=0.2,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None):

        self.work_threads = 0
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir


    def train_one_step(self, data, dflag, bflag):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode'],
            'dflag': dflag,
            'bflag': bflag,
        })
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                [{'params': self.model.model.transe1.parameters()},
                 {'params': self.model.model.transe2.parameters()}],
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        self.optimizer_rgat = optim.SGD(
            self.model.model.rgat.parameters(),
            lr=0.01,
            weight_decay=self.weight_decay,
        )
        t1 = time.time()
        loss_save = []
        for epoch in training_range:
            res = 0.0
            # cnt = self.data_loader1.get_nbatchs()
            # self.optimizer_rgat.zero_grad()
            # cnt = 200 215
            cnt = 200

            for i in range(cnt):
                # print('epoch:{}, batch:{}/{}'.format(epoch, i, cnt))
                # if epoch == 0 and i == 0:
                if i == 0:
                    bflag = 1
                else:
                    bflag = 0
                # do not need tot train rgat
                # bflag = -1
                loss = self.train_one_step(self.data_loader1.sampling(), 1, bflag)
                res += loss
                loss = self.train_one_step(self.data_loader2.sampling(), 2, bflag)
                res += loss
                torch.cuda.empty_cache()
                if i % 10 == 0:
                    self.optimizer_rgat.step()
                    self.optimizer_rgat.zero_grad()
            # self.optimizer_rgat.step()
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            loss_save.append(res)
            if res == 0 :
                print("loss is zero in epoch {}".format(epoch))
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))
        print('total time is{}'.format(time.time()-t1))
        print("the loss is:", loss_save)
    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir