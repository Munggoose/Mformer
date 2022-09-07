import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class Trainer:

    def __init__(self,args):
        self.model = None
        
        self.build_model(args)

    def build_model(self, args):
        print('check')



if __name__=='__main__':
    module = __import__('models.munformer') 
    func = getattr(module, 'Munformer')
    a = func(1,1,1,10,10,10)
    print(a)

    