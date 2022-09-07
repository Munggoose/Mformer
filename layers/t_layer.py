import torch
from torch import nn
# from torch.nn import Functional as F


class PostionEncoding(nn.Module):

    def __init__(self,embed_l,embed_d):
        super(PostionEncoding,self).__init__()
        self.embed_l = embed_l
        self.embed_d = embed_d

        self.pe = torch.zeros(embed_l, embed_d)
        self.pe.requires_grad = False

        pos = torch.arange(0, embed_d)
        _2i = torch.arange(0,embed_d, step=2)
        self.pe[:,::2] = torch.sin(pos/torch.pow(1000,_2i/self.embed_d))
        self.pe[:,1::2] = torch.cos(pos/torch.pow(1000,_2i/self.embed_d))

    def forward(self,x):
        batch_s, seq_len = x.size()
        return x + self.pe[:,:seq_len,:]

if __name__=='__main__':
    PE =  PostionEncoding(3,4)
    print(PE.pe)
       
