
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from torch.utils.data import DataLoader
import sys
sys.path.append('..')
from models.ts_transformer import  TSTransformerEncoderClassiregressor

class G_Dataset(Dataset):

    def __init__(self, data_path,batch_size,WINDOW_SIZE,split='train'):
        super(G_Dataset, self).__init__()
        self.X_train = np.load(os.path.join(data_path,'X_train_30.npy'))
        self.X_test = np.load(os.path.join(data_path,'X_test_30.npy'))
        self.Y_train = np.load(os.path.join(data_path,'y_train_30.npy'))
        self.Y_test = np.load(os.path.join(data_path,'y_test_30.npy'))
        # print(self.X_train.shape,self.Y_train.shape)
        self.batch_size = batch_size
        self.split = split
        self.length = WINDOW_SIZE
        
    def __getitem__(self, idx):
        batch_x=[]
        batch_y=[]
        padding_masks = []
        if self.split == 'train':
            x = self.X_train
            y = self.Y_train
            size = self.X_train.shape[0]
        elif self.split == 'test':  
            x = self.X_test
            y = self.Y_test    
            size = self.X_test.shape[0]        
        # for i in range(self.batch_size):
        start_ind = idx#self.length *idx
        end_ind = start_ind + self.length 
        if end_ind <= size:
            batch_x = x[start_ind : end_ind]
            batch_y  = y[end_ind -1]
        batch_x = np.array(batch_x,dtype=np.float32)
        batch_y = np.array(batch_y,dtype=np.float32) #/ 0.00513
        batch_x[np.isnan(batch_x)] = 0
        batch_x[np.isinf(batch_x)] = 0
        # batch_y = F.sigmoid(batch_y)
        # print(batch_x.shape)
        # mean = np.mean(batch_x,axis=0)
        # std =  np.std(batch_x,axis=0)
        # batch_x = ( batch_x - mean ) / std
        # # print(batch_x.shape)
        # for i in range(batch_x.shape[0]):
        #     batch_x[i][np.where(std==0)] = mean[np.where(std==0)]
        if np.any(np.isnan(batch_x)) or np.any(np.isinf(batch_x)):
            print(idx)

            print('X is nan')
            exit()
        if np.any(np.isnan(batch_y)):
            print('Y is nan')
        
        return batch_x, batch_y

    def __len__(self):
        if self.split == 'train':
            return self.X_train.shape[0]  - self.length
        elif self.split == 'test':  
            return self.X_test.shape[0] - self.length


class G_Trans(nn.modules.Module):
    def __init__(self,device, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, asset=2):
        super(G_Trans, self).__init__()

        self.encoder_list = []
        for i in range(asset):
            self.encoder_list.append(TSTransformerEncoderClassiregressor(feat_dim, max_len, d_model,
                                                        n_heads,
                                                        num_layers, dim_feedforward,
                                                        num_classes,
                                                        dropout, pos_encoding,
                                                        activation,
                                                        norm, freeze).to(device))
        self.f1 = nn.Linear(asset*num_classes,num_classes)
        self.f2 = nn.Linear(num_classes,asset)
        self.sigmod = nn.Sigmoid()
    def forward(self, X):
        batch_size,window_size,asset,feature = X.shape
        output = []
        for i in range(asset):
            x = X[:,:,i,:]
            out = self.encoder_list[i](x)
            output.append(out)
        final = torch.cat(output,1)
        final = self.f1(final)
        final = self.f2(final)
        # final = self.sigmod(final)
        return final
if __name__ == "__main__":
    # dataset = G_Dataset('D:\mvts_transformer\src\datasets',512,15,'test')
    # print(dataset.__len__())
    # loader = DataLoader(dataset=dataset,
    #                             batch_size=512,
    #                             shuffle=True)



    # model = G_Trans(feat_dim=25, max_len=15, d_model=64, n_heads=8,
    #                                                         num_layers=3, dim_feedforward=256,
    #                                                         num_classes=128,
    #                                                         dropout=0.1, pos_encoding='fixed',
    #                                                         activation='gelu',
    #                                                         norm='BatchNorm', freeze=False,asset=14)

    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter('graph') #建立一个保存数据用的东西                   

    # for data in loader:
    #     print(data[0].shape,data[1].shape)
        # input =data[0]
        # # with SummaryWriter(comment='G_trans') as w:
        # #     w.add_graph(model, (input,))
        # output = model(input)
        # # print(output.shape,data[1].shape)
        # break
    # train = np.load('train_raw.npy')
    # print(train.shape)
    # targets = np.load('targets_raw.npy')
    winsize = 60 * 24 * 5

    # for i in range(winsize,train.shape[0]):
    #     clip = train[i-winsize:i]
    #     mean = np.mean(clip,axis=0)
    #     std = np.std(clip,axis=0)
    #     print(i)
    #     # std[]

    #     train[i] =  (train[i]  - mean) /  std
    #     train[i][np.where(std==0)] = mean[np.where(std==0)]
        

    # train_new = train[winsize:]
    # train_new = train[winsize::winsize]
    # np.save('train_new',train_new)
    # print(train_new.shape)
    import os
    file_list = os.listdir('D:/mvts_transformer/src/datasets/alpha_data_16')
    print(file_list)
    total_pd = pd.DataFrame()
    for file_name in file_list:

        pdd = pd.read_hdf(os.path.join('D:/mvts_transformer/src/datasets/alpha_data_16',file_name))
        pdd.columns = [file_name.split('.')[0]]
        total_pd = pd.merge(total_pd,pdd,left_index=True,right_index=True,how='all')

    