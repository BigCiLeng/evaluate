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
import pandas as pd

loss_dict = {'mse': F.mse_loss,'l1':F.l1_loss,'sl1':F.smooth_l1_loss}
class MLP_EVALUATE_SYSTEM(pl.LightningModule):
    
    def __init__(self, hparams):
        super(MLP_EVALUATE_SYSTEM, self).__init__()
        self.h = hparams
        self.mlp = MLP(self.h['in_Channel'], self.h['hidden_Channel_list'], self.h['out_Channel'])
        self.loss = loss_dict[self.h['loss_type']]
        self.val_loss=0
        self.save_hyperparameters()
        
    def forward(self, input):
        result = self.mlp(input)
        return result
    
    def decode_batch(self, batch):
        data = batch['data']
        label = batch['label']
        return data, label
    
    def test_decode_batch(self, batch):
        data = batch['data']
        return data
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def prepare_data(self):
        rank=rank_normalize(self.h['rank'],self.h['data_max'],self.h['data_min'])
        kwargs = {'data': self.h['data'],
                  'rank': rank}   
        train_dataset = EvaluateDataset(**kwargs)
        self.test_dataset = train_dataset
        train_set_size = int(len(train_dataset) * 0.9)
        valid_set_size = len(train_dataset) - train_set_size
        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
        self.train_dataset=train_set
        self.val_dataset=valid_set
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=10,
                          pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=10,
                          pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=10,
                          pin_memory=True)
    
    def configure_optimizers(self): 
        eps = 1e-8
        self.optimizer = Adam(self.mlp.parameters(), lr=self.h['lr'], 
                              eps=eps, weight_decay=self.h['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.h['num_epochs'],
                                           eta_min=eps)
        return [self.optimizer], [self.scheduler]
    # def on_train_batch_start(self,batch, batch_idx):
    #     return 0
    def training_step(self, batch, batch_nb):
        log = {'lr': self.get_lr(self.optimizer)}
        data, label = self.decode_batch(batch)
        results = self(data).squeeze(1)
        log['train/loss'] = loss = self.loss(results, label)
        return {'loss': loss, 'log': log}
    
    def training_epoch_end(self, outputs):
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('mean_train_loss', mean_loss.clone().detach())
        
    def validation_step(self, batch, batch_nb):
        data, label = self.decode_batch(batch)
        results = self(data).squeeze(1)
        log = {'val_loss': self.loss(results, label)}
        return log
    
    def validation_epoch_end(self, outputs):
        all_loss=[x['val_loss'] for x in outputs]
        mean_loss=torch.tensor(all_loss).mean()
        self.val_loss=mean_loss
        return {'progress_bar': {'val_loss': mean_loss}, 'log':{'val/loss': mean_loss}}
    
    def test_step(self, batch, batch_idx):
        data,rank= self.decode_batch(batch)
        results = self(data).squeeze(0)
        return results
    
    def test_epoch_end(self, outputs):
        temp=np.array([i.numpy() for i in outputs])
        result=rank_denormalize(temp,self.h['data_max'],self.h['data_min'])
        pd.DataFrame(result).to_csv("data_output/output.csv", index=False, header=0)
        # with open(os.path.abspath(f'./logs/weight_opt/result'))

def train_solve(hparams):
    if(os.path.exists(os.path.abspath(r'./ckpts/evaluate/'))):
        shutil.rmtree(os.path.abspath(r'./ckpts/evaluate/'))
    else:
        os.mkdir(os.path.abspath(r'./ckpts/evaluate/'))
    if(os.path.exists(os.path.abspath(r'./logs/evaluate'))):
        shutil.rmtree(os.path.abspath(r'./logs/evaluate'))
    else:
        os.mkdir(os.path.abspath(r'./logs/evaluate/'))
    
    system = MLP_EVALUATE_SYSTEM(hparams)
    # checkpoint = ModelCheckpoint(dirpath=os.path.join(os.path.abspath(f'./ckpts/{h["exp_name"]}')),
    #                             filename='best',
    #                             monitor='mean_train_loss',
    #                             mode='min',
    #                             save_top_k=5,
    #                             )
    
    # logger = loggers.CSVLogger(
    #     save_dir = 'logs',
    #     name = hparams['exp_name'],
    #     flush_logs_every_n_steps=10
    # )
    trainer = pl.Trainer(max_epochs=h['num_epochs'],
                        # callbacks=[checkpoint],
                        # logger=logger,
                        accelerator='gpu',devices=1,
                        check_val_every_n_epoch=5,
                        num_sanity_val_steps=1,
                        benchmark=True,
                        # profiler=hparams['num_gpus']==1,
                        log_every_n_steps=1)
    trainer.fit(system)

def test_solve(hparams):
    kwargs = {'data': hparams['data'],'rank':hparams['rank']}
    model = MLP_EVALUATE_SYSTEM.load_from_checkpoint(os.path.abspath(r'ckpts/best.ckpt'), hparams=hparams)
    trainer = pl.Trainer(max_epochs=1)
    result = trainer.test(model, EvaluateDataset(**kwargs))
    print(result)
def rank_normalize(data,std,mean):
    return (data-mean)/std
def rank_denormalize(data,std,mean):
    return data*std+mean
def ori_data():
    data=np.loadtxt('../dataset/ship_design/main_five.csv',delimiter=',')
    data=data.T
    data_mean=[]
    data_std=[]
    for i in data:
        data_mean.append(np.mean(i))
        data_std.append(np.std(i))
    return np.array(data_mean),np.array(data_std)
if __name__ == "__main__":
    data=np.loadtxt('dataset/data_train.csv',delimiter=',')
    rank=np.loadtxt('dataset/rank_train.csv',delimiter=',')

    data_max,data_min=ori_data()


    h = {'loss_type': 'sl1',
               'in_Channel': 31,
               'hidden_Channel_list': [128,128,128,128,128],
               'out_Channel': 5,
               'lr': 3e-8,
               'weight_decay': 1e-4,
               'num_epochs': 10000,
               'exp_name': 'evaluate',
               'num_gpus': 1, 
               'data':data,
               'rank':rank,
               'data_max':data_max,
               'data_min':data_min
               }
    result=train_solve(hparams=h)
    # result = test_solve(hparams=h)
    print(result)
    # train_solve(hparams=h)