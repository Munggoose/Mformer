import torch
import torch.nn as nn
from utils.torch_utils import dct


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type='dct', norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        # if self.type == 'dct1':
        #     self.weight.data = dct1(I).data.t()
        # elif self.type == 'idct1':
        #     self.weight.data = idct1(I).data.t()
        if self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        # elif self.type == 'idct':
        #     self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!