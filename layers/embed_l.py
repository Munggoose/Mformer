import torch
from torch import nn
import torch.nn.functional as F

import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=720):
        
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2

        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        #init weight
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):

        x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) #if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x,x_mark):

        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_inp, d_model):
        super(TimeFeatureEmbedding,self).__init__()

        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)


class TimeCustomEmbedding(nn.Module):
    def __init__(self, d_inp, stamp_ch=4):
        super(TimeCustomEmbedding,self).__init__()

        self.stamp_ch = stamp_ch
        self.d_inp = d_inp
        self.time_layers = nn.ModuleList([nn.Conv2d(d_inp, d_inp,kernel_size=2) for i in range(stamp_ch)])
        # self.norm = nn.BatchNorm1d()
    
    def forward(self, x, mark):
        B,T,V = x.shape
        mark_B,mark_T, mark_V = mark.shape
        # x = x.permute(0,2,1)
        # mark = mark.permute(0,2,1)

        marks = torch.chunk(mark,mark_V,dim=2)
        # stamp_outs = torch.ones((mark_B,mark_V,mark_T,1))
        stamp_outs = torch.ones([mark_V,B,T,V],dtype=torch.float,device='cuda')
        for i,(time_l,mark_input) in enumerate(zip(self.time_layers,marks)):

            stamp_input = torch.cat([x,mark_input],dim=2)

            out = torch.cat([stamp_input,stamp_input[:,-1,:].unsqueeze(1)],1).unsqueeze(1)
    
            stamp_outs[i] = time_l(out).squeeze(1)
        stamp_outs = stamp_outs.squeeze(3)

        return stamp_outs.permute(1,2,0)

class TimeCustomEmbedding2(nn.Module):
    def __init__(self, d_inp, stamp_ch=4):
        super(TimeCustomEmbedding,self).__init__()

        self.stamp_ch = stamp_ch
        self.d_inp = d_inp
        self.time_layers = nn.ModuleList([nn.Conv2d(d_inp, d_inp,kernel_size=2) for i in range(stamp_ch)])

        self.conv_month = nn.Conv2d(d_inp,d_inp, kernel_size=2)
        self.conv_time = nn.Conv2d(d_inp, d_inp, kernel_size=2)
        self.conv_week = nn.Conv2d(d_inp, d_inp, kernel_size=2)
        # self.norm = nn.BatchNorm1d()
    
    def forward(self, x, mark):
        B,T,V = x.shape
        mark_B,mark_T, mark_V = mark.shape
        # x = x.permute(0,2,1)
        # mark = mark.permute(0,2,1)

        marks = torch.chunk(mark,mark_V,dim=2)
        # stamp_outs = torch.ones((mark_B,mark_V,mark_T,1))
        stamp_outs = torch.ones([mark_V,B,T,V],dtype=torch.float,device='cuda')
        for i,(time_l,mark_input) in enumerate(zip(self.time_layers,marks)):

            stamp_input = torch.cat([x,mark_input],dim=2)

            out = torch.cat([stamp_input,stamp_input[:,-1,:].unsqueeze(1)],1).unsqueeze(1)
    
            stamp_outs[i] = time_l(out).squeeze(1)
        stamp_outs = stamp_outs.squeeze(3)

        return stamp_outs.permute(1,2,0)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

if __name__ == '__main__':
    seq_x = torch.randn((2,336,1),dtype=torch.float)
    seq_mark = torch.randn((2,336,4),dtype=torch.float)
    embed_l = TimeCustomEmbedding(1,4)
    stamp_outs = embed_l(seq_x,seq_mark)
    print(stamp_outs.shape)
    