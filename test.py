import torch
import numpy as np
import os
import shutil
import pytorch_lightning as pl
import torch.nn.functional as F
from model import EVALUATE_MLP as MLP
from dataset.dataset import EvaluateDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
import torch.utils.data
import evaluate_normalxy

data=np.loadtxt('dataset/data_test.csv',delimiter=',')
rank=np.loadtxt('dataset/rank_test.csv',delimiter=',')
kwargs = {'data': data,'rank':rank}

model = evaluate_normalxy.MLP_EVALUATE_SYSTEM.load_from_checkpoint(os.path.abspath(r'lightning_logs/version_9/checkpoints/epoch=2844-step=469425.ckpt'))
trainer = pl.Trainer(max_epochs=1)
result = trainer.test(model)