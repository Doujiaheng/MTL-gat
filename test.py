
from config.Tester import Tester

from utils.data_utils import TrainDataLoader, TestDataLoader
from module.transe import TransE
from module.loss import MarginLoss
from module.stratage import NegativeSampling
from module.model import TranseRgat
from config.Trainer import Trainer
import torch
import torch.nn as nn
from module.rgat import GAT
import os
from module.transr import TransR
from module.transh import TransH
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

test_dataloader = TestDataLoader("./data/FB15K237/", 'link', base_add='Base.so')

train_dataloader1 = TrainDataLoader(
    in_path="./data/FB15K237/",
    batch_size=1000,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=1,
    neg_rel=0)

train_dataloader2 = TrainDataLoader(
    bfile='./release/Base1.so',
    in_path="./data/WN18RR/",
    batch_size=1000,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=1,
    neg_rel=0)


transe1 = TransE(
    ent_tot=train_dataloader1.get_ent_tot(),
    rel_tot=train_dataloader1.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True)

transe2 = TransE(
    ent_tot=train_dataloader2.get_ent_tot(),
    rel_tot=train_dataloader2.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True)
transr1 = TransR(
    ent_tot=train_dataloader1.get_ent_tot(),
    rel_tot=train_dataloader1.get_rel_tot(),
    dim_e=200,
    dim_r=200,
    p_norm=1,
    norm_flag=True)

transr2 = TransR(
    ent_tot=train_dataloader2.get_ent_tot(),
    rel_tot=train_dataloader2.get_rel_tot(),
    dim_e=200,
    dim_r=200,
    p_norm=1,
    norm_flag=True)
transh1 = TransH(
    ent_tot=train_dataloader1.get_ent_tot(),
    rel_tot=train_dataloader1.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True)

transh2 = TransH(
    ent_tot=train_dataloader2.get_ent_tot(),
    rel_tot=train_dataloader2.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True)
rgat = GAT(
    train_dataloader1.get_ent_tot(),
    train_dataloader1.get_rel_tot(),
    train_dataloader2.get_ent_tot(),
    train_dataloader2.get_rel_tot(),
)
transergat = TranseRgat(transe1=transr1, transe2=transr2, rgat=rgat, dim_trans=200, dim_rgat=200)

transergat.load_checkpoint("./checkpoint/dtransrrgat_2_2000.ckpt")
tester = Tester(model=transergat, data_loader=test_dataloader, use_gpu=True, dflag=-1)
tester.run_link_prediction(type_constrain=True)
tester.run_triple_classification()