import os
import sys
import time
import pickle
import json
import shutil
# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pandas as pd

import matplotlib.pyplot as plt
# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer

from datasets.g_dataset import *

def weigth_init(net):
    '''初始化网络'''
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)    
            torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight.data, 1)
            torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight.data, std=1e-3)
            torch.nn.init.constant_(m.bias.data, 0)

def backtest(signal, label):
    """
    param: signal  type 2-dim  samples * asset   ndarray
    param: label  type 2-dim  samples * asset   ndarray
    """
    signal_table = pd.DataFrame(signal).stack()
    label_table = pd.DataFrame(label).stack()
    signal_table.columns = ['signal']
    label_table.columns = ['label']
    table = pd.merge(signal_table, label_table, right_index=True, left_index=True, how='left')
    benchmark = table.groupby(as_index=True, level=0).apply(lambda x:x.sort_values(by='signal').mean())
    table = table.groupby(as_index=True, level=0).apply(lambda x:x.sort_values(by='signal').head(5).mean())
    plt.plot((table.loc[:,['label']]+1).cumprod())
    plt.plot((benchmark.loc[:,['label']]+1).cumprod())
    plt.show()
    plt.close()

if __name__ == '__main__':

    # X_t = pd.read_hdf(r'total_pd_norm.h5').loc[pd.IndexSlice['2021-07-21':,:],:'Target_y']  # 15min收益率标签
    # x_label = pd.read_hdf(r'total_pd.h5').loc['2021-07-21':,'Target_y']
    # pd.merge()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset = G_Dataset('D:\mvts_transformer\src\datasets',512,15,'train')
    test_dataset = G_Dataset('D:\mvts_transformer\src\datasets',512,15,'test')
    print(device)
    loader = DataLoader(dataset=dataset,
                                batch_size=512,
                                shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=512,
                                shuffle=False)

    model = G_Trans(device ,feat_dim=30, max_len=15, d_model=8, n_heads=2,
                                                            num_layers=1, dim_feedforward=8,
                                                            num_classes=8,
                                                            dropout=0.1, pos_encoding='fixed',
                                                            activation='gelu',
                                                            norm='BatchNorm', freeze=False,asset=14)
    model.apply(weigth_init)
    model.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    loss_module = nn.MSELoss(reduction='none').to(device)
    tensorboad_path = 'runs'
    if os.path.exists(tensorboad_path):
        shutil.rmtree(tensorboad_path)
        os.mkdir(tensorboad_path)
    writer = SummaryWriter(tensorboad_path)


    for j in range(100):
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        model = model.train()
        for i, batch in enumerate(loader):

            X, targets = batch[0].to(device), batch[1].to(device)
            # print(X.device,targets.device)
            # targets = torch.sigmoid(targets)
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
            predictions = model(X)
            
            if np.any(np.isnan(predictions.detach().cpu().numpy())):
                print('out is nan')
            loss = loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.mean(loss)
            # mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization
            # print(batch_loss.item())
            # if self.l2_reg:
            #     total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            # else:
            total_loss = batch_loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # metrics = {"loss": mean_loss.item()}
            # if i % self.print_interval == 0:
            #     ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            #     self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch
        
        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        writer.add_scalar('train/loss',epoch_loss , global_step=j)
        
        
        with torch.no_grad():
            model = model.eval()

            test_epoch_loss = 0  # total loss of epoch
            test_total_samples = 0  # total samples in epoch

            for i, batch in enumerate(test_loader):

                X, targets = batch[0].to(device), batch[1].to(device)
                # targets = torch.sigmoid(targets)
                predictions = model(X)
                # backtest(predictions)
                loss = loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
                batch_loss =  torch.mean(loss)
                # mean_loss = batch_loss / len(loss)  # mean loss (over samples)



                test_total_samples += len(loss)
                test_epoch_loss += batch_loss.item()  # add total loss of batch

            test_epoch_loss = test_epoch_loss / test_total_samples  # average loss per element for whole epoch



        
        writer.add_scalar('test/loss',test_epoch_loss, global_step=j)
        print('test',test_epoch_loss,'train',epoch_loss)
