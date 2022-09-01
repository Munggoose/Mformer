
import os
import numpy as np
from data.dataLoader import Dataset_ETT_hour
from torch.utils.data import DataLoader
from models.DLinear import DLinear
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime
from data.air_loader import Dataset_AIR_hour

def getparser():
    #basic config
    parser = argparse.ArgumentParser(description='MunLab for Time Series Forecasting')
    parser.add_argument('--is_trainig',type=int, default = 1 , help='status')
    parser.add_argument('--epoch',type=int, default=10, help='Epochs')

    #dataloader
    parser.add_argument('--batch_size',type=int, default=1, help ='batch size')
    parser.add_argument('--dataset',type=str, default='ETTH', help='Select dataset in ...')
    parser.add_argument('--features', type=str, default='S',help='for ETTH dataset, select in S,M, ')


    #model config
    parser.add_argument('--model', type=str, default='DLinear',help='select model in informer, DLinear, munformer,styleformer')

    #HpyerParameter
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
    parser.add_argument('--pred_len',type=int, default=168, help='forcasting length')
    parser.add_argument('--d_model', type=int, default=512, help='length lantet vector')
    
    #device
    parser.add_argument('--device',type=str, default='cuda:0', help='device type cuda:X , cpu ')

    return parser.parse_args()


if __name__ =='__main__':
    args = getparser()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    wandb.init(project="Munlab")
    wandb.run.name = f'{args.model}_{now}'
    wandb.config.update(args)

    # root = 'F:\\data\\ETDataset\\ETT-small'
    # trainset = Dataset_ETT_hour(root,features='S')
    root = 'F:\\data\\ETDataset\\ETT-small'
    root = 'F:\\data\\AIR'
            # self.seq_len = size[0]
            # self.label_len = size[1]
            # self.pred_len = size[2]

    trainset = Dataset_AIR_hour(root_path=root, flag='train',size=[384,192,192],
            features='S',data_path='5Year_Training.pkl',target='CO',
            sclae=True, timeenc=0, freq='h', cols=None)
    trainloader = DataLoader(trainset, batch_size= 2,shuffle=True)

    model = DLinear(seq_l=384,pred_l=192, enc_in=7,individual=False)
    model = model.cuda()

    criterion = nn.MSELoss()
    model_optim = optim.Adam(model.parameters(), lr = 1e-4)


    hp_epoch = args.epoch
    epoch_bar = tqdm(range(hp_epoch),desc='Epoch',leave=False)
    for epoch in epoch_bar:
        total_loss = []
        dataloader_bar = tqdm(trainloader,desc='data Iterator',leave=False)

        for seq_x, seq_y, seq_x_mark, seq_y_mark in dataloader_bar:
            model_optim.zero_grad()
            seq_x = seq_x.float().to(torch.device("cuda:0"))
            seq_y = seq_y.float() #.to(torch.device("cuda:0"))
            
            seq_x_mark = seq_x_mark.float().to(torch.device("cuda:0"))
            seq_y_mark = seq_y_mark.float().to(torch.device("cuda:0"))
            
            dec_inp = torch.zeros([seq_y.shape[0], 192, seq_y.shape[-1]]).float()
            dec_inp = torch.cat([seq_y[:,:192,:], dec_inp], dim=1).float().to(torch.device('cuda:0'))

            out = model(seq_x)
            pred = trainset.inverse_transform(out)
            true = seq_y[:,192:,-1:].to(torch.device("cuda:0"))
            # true = seq_y.to(torch.device("cuda:0"))
            loss = criterion(pred, true)
            total_loss.append(loss.item())
            # dataloader_bar.set_postfix({" data loss" : loss})

            loss.backward()
            model_optim.step()
    
        dataloader_bar.close()
        avg_loss = np.average(total_loss)
        wandb.log({
            "Train Loss": avg_loss
        })
        epoch_bar.set_postfix({f"{epoch+1} avgrage_loss : ": avg_loss})
    epoch_bar.close()
