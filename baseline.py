from utils.data_utils import TrainDataLoader, TestDataLoader
from module.transe import TransE
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


model = NegativeSampling(
    model=transe1,
    loss = MarginLoss(margin=5.0),
    batch_size1 = train_dataloader1.get_batch_size(),
)
trainer = Trainer(model=model, data_loader1=train_dataloader1, train_times=1000, use_gpu=True)
trainer.run()
transe1.save_checkpoint('./checkpoint/transe_1000.ckpt')
test_dataloader = TestDataLoader("./data/FB15K237/", "link")
# 1000 0.446
tester = Tester(model=transe1, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)