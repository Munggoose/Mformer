import os
import numpy as np
from data.dataLoader import Dataset_ETT_hour
from torch.utils.data import DataLoader
from models.informer import Informer
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
 
if __name__ =='__main__':
    root = 'F:\\data\\ETDataset\\ETT-small'
    trainset = Dataset_ETT_hour(root,features='S')
    trainloader = DataLoader(trainset, batch_size= 2,shuffle=True)
    model = Informer( enc_in=1, dec_in=1, c_out=1, seq_len=384, label_len=192, out_len=192, 
                factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = True, distil=True, mix=True)
    model = model.cuda()

    criterion = nn.MSELoss()
    model_optim = optim.Adam(model.parameters(), lr = 1e-4)


    hp_epoch = 10
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

            out,attn = model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
            pred = trainset.inverse_transform(out)
            #true = seq_y #seq_y[:,192:,-1:].to(torch.device("cuda:0"))
            true = seq_y.to(torch.device("cuda:0"))
            loss = criterion(pred, true)
            total_loss.append(loss.item())
            # dataloader_bar.set_postfix({" data loss" : loss})

            loss.backward()
            model_optim.step()
    
        dataloader_bar.close()
        avg_loss = np.average(total_loss)
        epoch_bar.set_postfix({f"{epoch+1} avgrage_loss : ": avg_loss})
    epoch_bar.close()
