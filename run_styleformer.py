import os
import numpy as np
from data.dataLoader import Dataset_ETT_hour
from torch.utils.data import DataLoader
from models.munformer import Munformer
from models.styleformer2 import StyleFormer
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import random
import argparse

#setting random seed
# fix_seed = 2021
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)
import wandb
from datetime import datetime

SEQ_LEN = 720
LABEL_LEN = 168
PRED_LEN = 168
D_MODEL = 512

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
    parser.add_argument('--model', type=str, default='styleformer2',help='select model in informer, DLinear, munformer,styleformer')

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


    root = 'F:\\data\\ETDataset\\ETT-small'
    trainset = Dataset_ETT_hour(root,size=[args.seq_len,args.pred_len, args.pred_len],features=args.features)    
    trainloader = DataLoader(trainset, batch_size= args.batch_size,shuffle=True,drop_last=True)

    # model = Munformer( enc_in=1, dec_in=1, c_out=1, seq_len=args.seq_len, label_len=args.pred_len, out_len=args.pred_len, 
    #             factor=5, d_model=args.d_model, n_heads=8, e_layers=2, d_layers=1, d_ff=512, 
    #             dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
    #             output_attention = True, distil=True, mix=True)
    model = StyleFormer(seq_l=args.seq_len,pred_l=args.pred_len,d_model=args.d_model,c_out=1,d_layers=1,m_layers=1,factor=5, dropout=0.0,n_heads=8)
    model = model.cuda()
    wandb.watch(model)
    criterion = nn.MSELoss()
    model_optim = optim.Adam(model.parameters(), lr = 1e-4)


    hp_epoch = args.epoch
    epoch_bar = tqdm(range(hp_epoch),desc='Epoch',leave=False)
    final_loss = 0
    for epoch in epoch_bar:
        total_loss = []
        dataloader_bar = tqdm(trainloader,desc='data Iterator',leave=False)

        for seq_x, seq_y, seq_x_mark, seq_y_mark in dataloader_bar:

            model_optim.zero_grad()
            seq_x = seq_x.float().to(torch.device(args.device))
            seq_y = seq_y.float()[:,:args.pred_len,:] #.to(torch.device("cuda:0"))

            seq_x_mark = seq_x_mark.float().to(torch.device(args.device))
            seq_y_mark = seq_y_mark.float().to(torch.device(args.device))
            
            dec_inp = torch.zeros([seq_y.shape[0], args.pred_len, seq_y.shape[-1]]).float()
            dec_inp = torch.cat([seq_y[:,:args.pred_len,:], dec_inp], dim=1).float().to(torch.device(args.device))

            seq_x2 = torch.empty_like(seq_x).copy_(seq_x)

            # out,attn = model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
            out = model(seq_x, seq_x2)

            pred = trainset.inverse_transform(out)
            #true = seq_y #seq_y[:,192:,-1:].to(torch.device("cuda:0"))
            true = seq_y.to(torch.device(args.device))

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
        final_loss = avg_loss
    epoch_bar.close()
    print(f'Styleformer Result!!! 개판이누')
    print(f'Final Average Loss: {final_loss:.3f}')
