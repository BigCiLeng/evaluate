import torch
import numpy as np
import os
import shutil
import pytorch_lightning as pl
import torch.nn.functional as F
from models.mlp import EVALUATE_MLP as MLP
from dataset.dataset import EvaluateDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
import torch.utils.data
import evaluate_normalxy

x = [170,15.7,11.6,5.8,0.463,93.28810949,0,76.71189051,-0.010056713,1.012542936,0.770039352,1.995795053,13.37544251,41.8571774,14.90676656,4.740788186,3.223911227,1.749396969,1.301728523,37.21801281,11.11564023,12.33511395,12.37046611,12.62059961,13.42847616,18.31458705,19.09199975,20.51640078]
y = evaluate_normalxy.predict(x)
print(y)