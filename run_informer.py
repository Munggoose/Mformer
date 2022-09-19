import os
import numpy as np
from data.dataLoader import Dataset_ETT_hour
from torch.utils.data import DataLoader
from models.informer import Informer
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import random
import argparse

from lossFunction.DTWLoss import SoftDTW
from lossFunction.fft_loss import FFTLoss

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

def make_result_folder(args):
    dir_name = f"{args.model}_{args.dataset}"
    dir_path = os.path.join('result',dir_name)
    os.makedirs(dir_path,exist_ok=True)
    return dir_name,dir_path

def getparser():
    #basic config
    parser = argparse.ArgumentParser(description='MunLab for Time Series Forecasting')
    parser.add_argument('--is_trainig',type=int, default = 1 , help='status')
    parser.add_argument('--epoch',type=int, default=10, help='Epochs')
    parser.add_argument('--criterion',type=str, default='MSE',choices=['MSE','DTW','FFT'])


    #dataloader
    parser.add_argument('--batch_size',type=int, default=2, help ='batch size')
    parser.add_argument('--dataset',type=str, required=True,choices=['air','etth','ettm'], help='Select dataset in ...')
    parser.add_argument('--features', type=str, default='S',help='for ETTH dataset, select in S,M, ')
    parser.add_argument('--inverse', type= bool, default= False)

    #model config
    parser.add_argument('--model', type=str, default='Informer',help='select model in informer, DLinear, munformer')
    

    #HpyerParameter
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
    parser.add_argument('--pred_len',type=int, default=336, help='forcasting length')
    parser.add_argument('--label_len',type=int, default=336, help='forcasting length')
    parser.add_argument('--d_model', type=int, default=512, help='length lantet vector')
    parser.add_argument('--n_heads', type=int, default= 8)
    parser.add_argument('--e_layers', type=int, default= 2)
    parser.add_argument('--d_layers', type=int, default= 1)
    parser.add_argument('--d_ff', type=int, default= 512)
    


    #device
    parser.add_argument('--device',type=str, default='cuda:0', help='device type cuda:X , cpu ')

    return parser.parse_args()

if __name__ =='__main__':
    args = getparser()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    wandb.init(project="Munlab")
    wandb.run.name = f'{args.model}_{now}'
    wandb.config.update(args)


    if args.dataset == 'etth':
        from data.dataLoader import Dataset_ETT_hour
        root = 'F:\\data\\ETDataset\\ETT-small'
        trainset = Dataset_ETT_hour(root,features=args.features,flag='train',size=[args.seq_len,args.pred_len,args.pred_len],target='OT')
        testset = Dataset_ETT_hour(root,features=args.features,flag='test',size=[args.seq_len,args.pred_len,args.pred_len],target='OT')
    #AIR
    elif args.dataset == 'ettm':
        from data.dataLoader import Dataset_ETT_minute
        root = 'F:\\data\\ETDataset\\ETT-small'
        trainset = Dataset_ETT_minute(root,features=args.features,flag='train',size=[args.seq_len,args.pred_len,args.pred_len],target='OT')
        testset = Dataset_ETT_minute(root,features=args.features,flag='test',size=[args.seq_len,args.pred_len,args.pred_len],target='OT')

    elif args.dataset == 'AIR':
        from data.air_loader import Dataset_AIR_hour
        root = 'F:\\data\\AIR'

        trainset = Dataset_AIR_hour(root_path=root, flag='train',size=[args.seq_len,args.pred_len,args.pred_len],
                features='S',target='CO',
                scale=True, timeenc=0, freq='h', cols=None)
        
        # #month day weekday hour
        testset = Dataset_AIR_hour(root_path=root, flag='test',size=[args.seq_len,args.pred_len,args.pred_len],
                    features='S',target='CO',
                    scale=True, timeenc=0, freq='h', cols=None)


    trainloader = DataLoader(trainset, batch_size= args.batch_size,shuffle=True,drop_last=True)
    testloader = DataLoader(testset, batch_size= args.batch_size,shuffle=True,drop_last=True)

    model = Informer( enc_in=1, dec_in=1, c_out=1, seq_len=args.seq_len, label_len=args.pred_len, out_len=args.pred_len, 
                factor=5, d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers, d_ff=args.d_ff, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = True, distil=True, mix=True)

    model = model.cuda()
    wandb.watch(model)
    criterions = {'MSE': nn.MSELoss(), 'DTW': SoftDTW(use_cuda=True, gamma=0.1), 'FFT': FFTLoss(seq_l=args.pred_len)}

    criterion = criterions[args.criterion]#nn.MSELoss()
    

    model_optim = optim.Adam(model.parameters(), lr = 1e-4)
    exp_name,result_path = make_result_folder(args)

    best_val_loss = torch.inf
    hp_epoch = args.epoch
    epoch_bar = tqdm(range(hp_epoch),desc='Epoch',leave=False)
    final_loss = 0
    
    for epoch in epoch_bar:
        total_loss = []
        dataloader_bar = tqdm(trainloader,desc='data Iterator',leave=False)
        last_pred = None
        last_label = None
        model.train()
        for seq_x, seq_y, seq_x_mark, seq_y_mark in dataloader_bar:
            model_optim.zero_grad()
            seq_x = seq_x.float().to(torch.device(args.device))
            seq_y = seq_y.float() #.to(torch.device("cuda:0"))

            seq_x_mark = seq_x_mark.float().to(torch.device(args.device))
            seq_y_mark = seq_y_mark.float().to(torch.device(args.device))
            
            dec_inp = torch.zeros([seq_y.shape[0], args.pred_len, seq_y.shape[-1]]).float()
            dec_inp = torch.cat([seq_y[:,:args.label_len,:], dec_inp], dim=1).float().to(torch.device(args.device))

            out,attn = model(seq_x, seq_x_mark, dec_inp, seq_y_mark)

            pred = out
            true = seq_y[:,-args.pred_len:,:].to(torch.device(args.device))
            # true = trainset.inverse_transform(true)

            if args.inverse:
                true = testset.inverse_transform(true)
                pred = testset.inverse_transform(pred)
            loss = criterion(pred, true)
            
            # loss2 = MSE_criterion(pred, true) 
            total_loss.append(loss.item())
            # dataloader_bar.set_postfix({" data loss" : loss})

            loss.backward()
            model_optim.step()
        
        dataloader_bar.close()
        avg_loss = np.average(total_loss)
        val_loss = []
        val_avg_loss = None
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(testloader):
            seq_x = seq_x.float().to(torch.device(args.device))
            seq_y = seq_y.float().to(torch.device("cuda:0"))

            seq_x_mark = seq_x_mark.float().to(torch.device(args.device))
            seq_y_mark = seq_y_mark.float().to(torch.device(args.device))
            
            dec_inp = torch.zeros([seq_y.shape[0], args.pred_len, seq_y.shape[-1]]).float().to(torch.device(args.device))
            dec_inp = torch.cat([seq_y[:,:args.pred_len,:], dec_inp], dim=1).float().to(torch.device(args.device))

            out,attn = model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
            # pred = testset.inverse_transform(out)
            # true = seq_y #seq_y[:,192:,-1:].to(torch.device("cuda:0"))
            
            true = seq_y[:,-args.pred_len:,:] #.to(torch.device(args.device))
            pred = out
            # true = testset.inverse_transform(true)
            if args.inverse:
                true = testset.inverse_transform(true)
                pred = testset.inverse_transform(pred)

            loss =  criterion(pred,true) #mse_lw * criterion(out, true) + sub_lw * sub_criterion(out.squeeze(),true.squeeze())
            
            # loss2 = MSE_criterion(pred, true) 
            val_loss.append(loss.item())
            # dataloader_bar.set_postfix({" data loss" : loss})

        
        val_avg_loss = np.average(val_loss)
        wandb.log({"Train Loss": avg_loss,
                    'Validation Loss': val_avg_loss})

        epoch_bar.set_postfix({f"{epoch+1} avgrage_loss : ": avg_loss})
        
        
        if val_avg_loss < best_val_loss:
            pkl_path = os.path.join(result_path,f'Epoch_best_{epoch}.pth')
            torch.save(model,pkl_path)
            best_val_loss = val_avg_loss

        elif(epoch %10):
            pkl_path = os.path.join(result_path,f'Epoch_{epoch}.pth')
            torch.save(model,pkl_path)

    epoch_bar.close()
    print(f'{args.model} Result!!! 개판이누')
    print(f'Final Average Loss: {final_loss:.3f}')