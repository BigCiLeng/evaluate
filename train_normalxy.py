import numpy as np
import os
import shutil
import pytorch_lightning as pl
from evaluate_normalxy import MLP_EVALUATE_SYSTEM
from evaluate_normalxy import ori_rank
from evaluate_normalxy import ori_data_std

if __name__ == '__main__':
     data = np.loadtxt('dataset/data_train.csv', delimiter=',')
     rank = np.loadtxt('dataset/rank_train.csv', delimiter=',')

     rank_std, rank_mean = ori_rank()
     data_1, data_2 = ori_data_std()

     h = {'loss_type': 'sl1',
          'in_Channel': 31,
          'hidden_Channel_list': [128, 128, 128, 128, 128],
          'out_Channel': 5,
          'lr': 3e-6,
          'weight_decay': 1e-4,
          'num_epochs': 10000,
          'exp_name': 'evaluate',
          'num_gpus': 2,
          'data': data,
          'rank': rank,
          'rank_std': rank_std,
          'rank_mean': rank_mean,
          'data_1': data_1,
          'data_2': data_2
          }
     if(os.path.exists(os.path.abspath(r'./ckpts/evaluate/'))):
          shutil.rmtree(os.path.abspath(r'./ckpts/evaluate/'))
     else:
          os.mkdir(os.path.abspath(r'./ckpts/evaluate/'))
     if(os.path.exists(os.path.abspath(r'./logs/evaluate'))):
          shutil.rmtree(os.path.abspath(r'./logs/evaluate'))
     else:
          os.mkdir(os.path.abspath(r'./logs/evaluate/'))

     system = MLP_EVALUATE_SYSTEM(hparams=h)
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
                         accelerator='gpu',devices=h['num_gpus'],
                         strategy="ddp",
                         check_val_every_n_epoch=5,
                         num_sanity_val_steps=1,
                         benchmark=True,
                         # profiler=hparams['num_gpus']==1,
                         log_every_n_steps=1)
     trainer.fit(system)
