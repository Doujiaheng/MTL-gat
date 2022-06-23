from utils.data_utils import TrainDataLoader, TestDataLoader
from module.transe import TransE
from module.transr import TransR
from module.transh import TransH
from module.loss import MarginLoss
from module.stratage import NegativeSampling
from module.model import TranseRgat
from config.Trainer import Trainer
from module.rgat import GAT
from config.Tester import Tester
import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

train_dataloader1 = TrainDataLoader(
    in_path="./data/FB15K237/",
    batch_size=500,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=1,
    neg_rel=0)

train_dataloader2 = TrainDataLoader(
    bfile='./release/Base1.so',
    in_path="./data/WN18RR/",
    batch_size=500,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=1,
    neg_rel=0)


# transe1 = TransE(
#     ent_tot=train_dataloader1.get_ent_tot(),
#     rel_tot=train_dataloader1.get_rel_tot(),
#     dim=200,
#     p_norm=1,
#     norm_flag=True)
#
# transe2 = TransE(
#     ent_tot=train_dataloader2.get_ent_tot(),
#     rel_tot=train_dataloader2.get_rel_tot(),
#     dim=200,
#     p_norm=1,
#     norm_flag=True)

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
# 215
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

# 215
# transergat.load_checkpoint('./checkpoint/dtranshrgat_2500.ckpt')
# 214
# transergat.load_checkpoint('./checkpoint/dtransrrgat_pretrain_7000_214.ckpt')

model = NegativeSampling(
    model=transergat,
    loss=MarginLoss(margin=5.0),
    batch_size1=train_dataloader1.get_batch_size(),
    batch_size2=train_dataloader2.get_batch_size()
)
trainer = Trainer(model=model, data_loader1=train_dataloader1, data_loader2=train_dataloader2, train_times=2000, use_gpu=True)
trainer.run()

# transergat.save_checkpoint('./checkpoint/dtranshrgat_3000.ckpt')
transergat.save_checkpoint('./checkpoint/dtransrrgat_2_2000.ckpt')
test_dataloader = TestDataLoader("./data/WN18RR/", "link", base_add='Base1.so')
# 1000 0.446
tester = Tester(model=transergat, data_loader=test_dataloader, use_gpu=True, dflag=-2)
tester.run_link_prediction(type_constrain=False)