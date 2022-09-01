import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.scinet import SCINet
import argparse
from data.dataLoader import Dataset_ETT_hour
from tqdm import tqdm
import wandb

def get_parser():
    parser = argparse.ArgumentParser(description='SCINet on ETT dataset')

    parser.add_argument('--model', type=str, required=False, default='SCINet', help='model of the experiment')
    ### -------  dataset settings --------------
    parser.add_argument('--data', type=str, required=False, default='ETTh1', choices=['ETTh1', 'ETTh2', 'ETTm1'], help='name of dataset')
    parser.add_argument('--root_path', type=str, default='./datasets/ETT-data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')
    parser.add_argument('--features', type=str, default='S', choices=['S', 'M'], help='features S is univariate, M is multivariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='exp/ETT_checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')


    ### -------  device settings --------------
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')
    
                                                                                    
    ### -------  input/output length settings --------------                                                                            
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of SCINet encoder, look back window')
    parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length, horizon')
    parser.add_argument('--concat_len', type=int, default=0)
    parser.add_argument('--single_step', type=int, default=0)
    parser.add_argument('--single_step_output_One', type=int, default=0)
    parser.add_argument('--lastWeight', type=float, default=1.0)
    parser.add_argument('--in_dim', type =int, default =1)
                                                                
    ### -------  training settings --------------  
    parser.add_argument('--cols', type=str, nargs='+', help='file list')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=0, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mae',help='loss function')
    parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--save', type=bool, default =False, help='save the output results')
    parser.add_argument('--model_name', type=str, default='SCINet')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)

    ### -------  model settings --------------  
    parser.add_argument('--hidden-size', default=1, type=float, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--window_size', default=12, type=int, help='input size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--positionalEcoding', type=bool, default=False)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')
    parser.add_argument('--num_decoder_layer', type=int, default=1)
    parser.add_argument('--RIN', type=bool, default=False)
    parser.add_argument('--decompose', type=bool,default=False)

    return  parser.parse_args()

from datetime import datetime

if __name__ =='__main__':
    args = get_parser()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    now = datetime.now()

    wandb.init(project="Munlab")
    wandb.run.name = f'SCINET_ETTH_{now}'
    wandb.config.update(args)


    root = 'F:\\data\\ETDataset\\ETT-small'

    trainset = Dataset_ETT_hour(root,size = [args.seq_len,args.label_len, args.pred_len],features=args.features)    


    model = SCINet(output_len=args.pred_len,
                input_len=args.seq_len,
                input_dim= args.in_dim,
                hid_size = args.hidden_size,
                num_stacks=args.stacks,
                num_levels=args.levels,
                num_decoder_layer=args.num_decoder_layer,
                concat_len = args.concat_len,
                groups = args.groups,
                kernel = args.kernel,
                dropout = args.dropout,
                single_step_output_One = args.single_step_output_One,
                positionalE = args.positionalEcoding,
                modified = True,
                RIN=args.RIN).double().cuda()
    wandb.watch(model)
    model_optim = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    trainloader = DataLoader(trainset, batch_size=args.batch_size,drop_last=True, shuffle=True)

    
    


    model.add_module

    epoch_bar = tqdm(range(args.train_epochs),desc = 'Train Epoch' , leave=False)
    for epoch  in epoch_bar:
        iterator_bar = tqdm(trainloader, desc='train iterator', leave=False)
        train_loss = []
        last_pred = None
        last_label = None
        for seq_x,seq_y, _,_ in iterator_bar:


            model_optim.zero_grad()
            seq_x = seq_x.double().to(torch.device('cuda'))
            seq_y = seq_y.double()
            if args.stacks == 1:
                outputs = model(seq_x)
            else:
                outputs,mid = model(seq_x)
            
            outputs_scaled = trainset.inverse_transform(outputs)
            if args.stacks == 2:
                mid_scaled = trainset.inverse_transform(mid)
            f_dim = -1 if args.features=='MS' else 0
            seq_y = seq_y[:,-args.pred_len:,f_dim:].cuda()
            # batch_y_scaled = trainset.inverse_transform(batch_y)

            if args.stacks == 1:
                pred = model(seq_x)
                loss = criterion(pred,seq_y)

            elif args.stacks == 2:
                pred,mid = model(seq_x)
                loss = criterion(pred, seq_y) + criterion(mid,seq_y)
            last_pred = pred
            last_label = seq_y

            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        iterator_bar.close()
        avg_loss = np.average(train_loss)

        pred_line  = [[x,y] for (x,y) in zip(range(len(last_pred[0,:,0])),last_pred[0,:,0])]
        label_line = [[x,y] for (x,y) in zip(range(len(last_label[0,:,0])),last_label[0,:,0])]
        
        pred_data = wandb.Table(data=pred_line, columns = ["x","y"])
        label_data = wandb.Table(data=label_line, columns = ["x","y"])
        wandb.log({
            "Train Loss": avg_loss,
            "SCINet_pred_plot" : wandb.plot.line(pred_data,
                                "x", "y", title=f"pred_plot"),
            "SCINet_label_plot" : wandb.plot.line(label_data,
                                "x", "y", title=f"label_plot")

        })

        epoch_bar.set_postfix({f"{epoch+1} loss : ": avg_loss})
    epoch_bar.close()

