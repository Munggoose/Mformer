import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):

    def __init__(self, n_heads, embed_d,mask=None):
        super(AttentionLayer, self).__init__()
        self.mask = mask
        self.d_k = embed_d//n_heads
        self.W_q = nn.Linear(embed_d,embed_d//n_heads)#nn.Parameter(torch.ones(embed_d,self.d_k)) #nn.Linear(embed_d,embed_d//n_heads)
        self.W_k = nn.Linear(embed_d,embed_d//n_heads)#nn.Parameter(torch.ones(embed_d,self.d_k)) #nn.Linear(embed_d,embed_d//n_heads)
        self.W_v = nn.Linear(embed_d,embed_d//n_heads)#nn.Parameter(torch.ones(embed_d,self.d_k)) #nn.Linear(embed_d,embed_d//n_heads)


    def forward(self, x):
        # Q = torch.matmul(x,self.W_q)
        # K = torch.matmul(x,self.W_k)
        # V = torch.matmul(x,self.W_v)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        K_t = K.transpose(2,1)

        
        logit = torch.matmul(Q,K_t)/math.sqrt(self.d_k)

        #masking 한 영역을 지운다고 생각하면 됨 softmax 결과 0이 되버림
        if self.mask is not None:
            logit += (self.mask * -1e9)

        attn_weight = F.softmax(logit,dim=1)
        attn_score = torch.matmul(attn_weight,V)

        return attn_score,attn_weight


class MultiheadAttnetion(nn.Module):
    def __init__(self, n_heads, embed_d,mask=None):
        super(AttentionLayer, self).__init__()
        self.mask = mask
        self.embed_d = embed_d
        self.n_heads = n_heads
        self.d_k = embed_d//n_heads
        self.W_q = nn.Linear(embed_d)#nn.Parameter(torch.ones(embed_d,self.d_k)) #nn.Linear(embed_d,embed_d//n_heads)
        self.W_k = nn.Linear(embed_d)#nn.Parameter(torch.ones(embed_d,self.d_k)) #nn.Linear(embed_d,embed_d//n_heads)
        self.W_v = nn.Linear(embed_d)#nn.Parameter(torch.ones(embed_d,self.d_k)) #nn.Linear(embed_d,embed_d//n_heads)

        self.out_layer = nn.Linear(embed_d)


    def split_head(self, inputs, bt_size):
        return inputs.view(bt_size, self.n_heads,-1,self.d_K)


    def forward(self, x, mask=None):
        # Q = torch.matmul(x,self.W_q)
        # K = torch.matmul(x,self.W_k)
        # V = torch.matmul(x,self.W_v)
        bt_size = x.shape[0]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = self.split_head(Q, bt_size)
        K = self.split_head(K, bt_size)
        V = self.split_head(V, bt_size)
        
        K_t = K.transpose(3,2)

        
        logit = torch.matmul(Q,K_t)/math.sqrt(self.d_k)

        #masking 한 영역을 지운다고 생각하면 됨 softmax 결과 0이 되버림
        if self.mask is not None:
            logit += (self.mask * -1e9)

        attn_weight = F.softmax(logit,dim=1)
        attn_score = torch.matmul(attn_weight,V)
        attn_score = attn_score.view(bt_size, -1, self.embed_d)
        attn_score = self.out_layer(attn_score)
        return attn_score


class PositionWiseFFNN(nn.Module):
    def __init__(self,d_ff, embed_d):
        super(PositionWiseFFNN,self).__init__()
        self.in_layer = nn.Linear(embed_d,d_ff)
        self.relu = F.relu()
        self.out_layer = nn.Linear(d_ff, embed_d)
    

    def forward(self, x):
        out= self.in_layer(x)
        out = self.relu(out)
        return self.out_layer(out)

    
