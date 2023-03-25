import torch
import numpy as np
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
import pandas as pd

loss_dict = {'mse': F.mse_loss, 'l1': F.l1_loss, 'sl1': F.smooth_l1_loss}


class MLP_EVALUATE_SYSTEM(pl.LightningModule):

    def __init__(self, hparams):
        super(MLP_EVALUATE_SYSTEM, self).__init__()
        self.h = hparams
        self.mlp = MLP(
            self.h['in_Channel'], self.h['hidden_Channel_list'], self.h['out_Channel'])
        self.loss = loss_dict[self.h['loss_type']]
        self.val_loss = 0
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

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def setup(self, stage):
        data_test = np.loadtxt('dataset/data_test.csv', delimiter=',')[:,0:28]
        rank_test = np.loadtxt('dataset/rank_test.csv', delimiter=',')
        data_test = normalize(
            data_test, self.h['data_1'], self.h['data_2'])
        rank_test = normalize(
            rank_test, self.h['rank_1'], self.h['rank_2']
        )
        kwargs = {'data': data_test, 
                  'rank': rank_test}
        self.test_dataset = EvaluateDataset(**kwargs)

        data_train = np.loadtxt('dataset/data_train.csv', delimiter=',')[:,0:28]
        rank_train = np.loadtxt('dataset/rank_train.csv', delimiter=',')
        data_train = normalize(
            data_train, self.h['data_1'], self.h['data_2'])
        rank_train = normalize(
            rank_train, self.h['rank_1'], self.h['rank_2'])
        kwargs = {'data': data_train,
                  'rank': rank_train}
        train_dataset = EvaluateDataset(**kwargs)
        train_set_size = int(len(train_dataset) * 0.9)
        valid_set_size = len(train_dataset) - train_set_size
        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = torch.utils.data.random_split(
            train_dataset, [train_set_size, valid_set_size], generator=seed)
        self.train_dataset = train_set
        self.val_dataset = valid_set

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
        self.log('mean_train_loss', mean_loss.clone().detach(), sync_dist=True)

    def validation_step(self, batch, batch_nb):
        data, label = self.decode_batch(batch)
        results = self(data).squeeze(1)
        log = {'val_loss': self.loss(results, label)}
        return log

    def validation_epoch_end(self, outputs):
        all_loss = outputs[0]['val_loss']
        mean_loss = all_loss.mean()
        return {'progress_bar': {'val_loss': mean_loss}, 'log': {'val/loss': mean_loss}}

    def test_step(self, batch, batch_idx):
        data, rank = self.decode_batch(batch)
        results = self(data).squeeze(0)
        return results

    def test_epoch_end(self, outputs):
        temp = torch.cat([i for i in outputs])
        result = denormalize(
            temp.numpy(), self.h['rank_1'], self.h['rank_2'])
        pd.DataFrame(result).to_csv(
            "data_output/output_0.csv", index=False, header=0)
        # with open(os.path.abspath(f'./logs/weight_opt/result'))

def predict(x):
    data_1, data_2 = ori_data_std()
    rank_1, rank_2 = ori_rank()
    model = MLP_EVALUATE_SYSTEM.load_from_checkpoint("ckpts/best_normalxy_28.ckpt")
    model.eval()
    x = normalize(x,data_1,data_2)
    x = torch.tensor(x,dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    y = denormalize(y,rank_1,rank_2)
    y = y.numpy()
    return y

def normalize(data, std, mean):
    result = (data-mean)/std
    result[np.isnan(result)] = 0
    result[np.isinf(result)] = 0
    return result

def denormalize(data, std, mean):
    return data*std+mean


def ori_data_std(path="dataset/main_value_3_21.csv"):
    data = np.loadtxt(path, delimiter=',')[:,0:28]
    data = data.T
    data_mean = []
    data_std = []
    for i in data:
        data_mean.append(np.mean(i))
        data_std.append(np.std(i))
    return np.array(data_std), np.array(data_mean)

def ori_rank(path="dataset/main_five_3_21.csv"):
    rank = np.loadtxt(path, delimiter=',')
    rank = rank.T
    rank_mean = []
    rank_std = []
    for i in rank:
        rank_mean.append(np.mean(i))
        rank_std.append(np.std(i))
    return np.array(rank_std), np.array(rank_mean)


if __name__ == "__main__":
    print('ok')
