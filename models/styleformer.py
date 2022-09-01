import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.attn import AttentionLayer, FullAttention
from models.EncDec import DecoderLayer,Decoder
from layers.embed_l import PositionalEmbedding



class StyleFormer(nn.Module):

    def __init__(self,seq_l,pred_l, d_model,c_out,d_layers,m_layers,factor, dropout,n_heads):
        super(StyleFormer,self).__init__()
        self.pred_len = pred_l

        self.dec_embedding  = PositionalEmbedding(d_model=d_model)
        self.map_embedding  = PositionalEmbedding(d_model=d_model)
        
        self.mapnet = MappingNet(seq_l,d_model,m_layers)

        self.decoder_layers = nn.ModuleList([Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                    d_model,n_heads, mix=True),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    512,
                    dropout=True,
                    activation='relu',
                ) 
            ],
            norm_layer=nn.LayerNorm(d_model)
        )for i in range(d_layers) ])
        
        self.projection= nn.Linear(d_model, c_out,bias=True)

    def forward(self, x_1,x_2):
        map_in = self.map_embedding(x_1)
        dec_in = self.dec_embedding(x_2)
        map_out = self.mapnet(map_in)
        
        for dec_layer in self.decoder_layers:
            map_out = dec_layer(dec_in, map_out)

        map_out =  self.projection(map_out)
        return map_out[:,-self.pred_len:,:] # [B, L, D]



class MappingNet(nn.Module):

    def __init__(self, seq_l,d_model,layer_n):
        super(MappingNet,self).__init__()
        self.fc_input = torch.nn.Linear(d_model,d_model)
        
        self.fc_modules = nn.ModuleList()
        for i in range(layer_n):
            self.fc_modules.append(nn.Linear(d_model, d_model))
            self.fc_modules.append(nn.ReLU())
            self.fc_modules.append(nn.BatchNorm1d(seq_l))

        self.projection = nn.Linear(d_model, d_model)


    def forward(self, x):
        out = self.fc_input(x)
        for layer in self.fc_modules:
            out = layer(out)
        
        return self.projection(out)
        