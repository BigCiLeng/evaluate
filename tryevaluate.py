import pytorch_lightning as pl
import torch
import evaluate_normalxy
import numpy as np
import pandas as pd
data_1, data_2 = evaluate_normalxy.ori_data_std()
rank_1, rank_2 = evaluate_normalxy.ori_rank()
model = evaluate_normalxy.MLP_EVALUATE_SYSTEM.load_from_checkpoint("lightning_logs/version_12/checkpoints/epoch=9999-step=300000.ckpt")
model.eval()
x=np.loadtxt('dataset/data_test.csv',delimiter=',')[:,0:28]
x=evaluate_normalxy.normalize(x,data_1,data_2)
x=torch.tensor(x,dtype=torch.float32)
with torch.no_grad():
    y = model(x)
y = evaluate_normalxy.denormalize(y,rank_1,rank_2)
    # x1,x2,x3,x4,x5=y.numpy()
    # print(x1,x2,x3,x4,x5)
pd.DataFrame(y.numpy()).to_csv('predict.csv',index=False,header=False)