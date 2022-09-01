import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from layers.attn import AttentionLayer, FullAttention
from models.EncDec import DecoderLayer,Decoder
from layers.embed_l import PositionalEmbedding
from models.DLinear import Series_decomp
from math import sqrt

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting,self).__init__()

    def even(self,x):
        return x[:,::2,:]
    
    def odd(self, x):
        return x[:,1::2,:]

    def forward(self,x):
        return (self.even(x),self.odd(x))


