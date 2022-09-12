import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed_l import DataEmbedding
from layers.attn import FullAttention,AttentionLayer
from layers.informer_layer import EncoderLayer,ConvLayer,DecoderLayer
from models.EncDec import Encoder,Decoder,EncoderStack
from layers.exp_layer import CustomLayer
from layers.embed_l import TimeCustomEmbedding

class Munformer(nn.Module):
    
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        
        super(Munformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        #Encoding 


        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        #Select Attention Layer
        Attn = FullAttention
        #Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation) for i in range(e_layers)

            ],
            [
                ConvLayer(
                    d_model
                )for i in range(e_layers-1)
            ] if distil else None,
            norm_layer = torch.nn.LayerNorm(d_model)
        )
        

        #Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                    d_model,n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for i in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.projection= nn.Linear(d_model, c_out,bias=True)

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):


        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

    
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


