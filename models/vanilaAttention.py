import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed_l import PositionalEmbed
from layers.attn_l import MultiheadAttnetion,PositionWiseFFNN


def create_padding_mask(x):
  mask = torch.zeros_like(x).float() 
    # (batch_size, 1, 1, key의 문장 길이)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = x.shape[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)

class Attention_block:
    def __init__(self,n_heads,embed_d,d_ff):
        super(Attention_block,self).__init__()
        
        self.m_atten = MultiheadAttnetion(n_heads, embed_d)
        self.norm_1 = nn.LayerNorm()
        self.norm_2 = nn.LayerNorm()
        self.ffnn = PositionWiseFFNN(d_ff,embed_d)


    def forward(self, x):
        out = self.m_atten(x)
        out = self.norm_1( x + out)
        out = self.norm_2(out + self.ffnn(out))
        return out



class AttentionEncoder(nn.Module):

    def __init__(self, n_att_block,n_heads,embed_d,seq_len,d_ff):
        super(AttentionEncoder,self).__init__()
        self.pos_embed = PositionalEmbed(embed_d, seq_len)
        self.att_layers = nn.Sequential()
        for i in range(n_att_block):
            self.att_layers.append(Attention_block(n_heads,embed_d,d_ff))


    def forward(self, x):
        out = self.pos_embed(x)
        out = self.att_layers(out)
        return out



class AttentionDecoder(nn.Module):
    def __init__(self, n_att_block,n_heads, embed_d, seq_len, d_ff):
        super(AttentionDecoder,self).__init__()
        
