'''
Description: 
Author: Yu Sha
Date: 2021-09-28 09:30:05
LastEditors: Yu Sha
LastEditTime: 2022-01-29 12:37:07
'''
import torch
import torch.nn as nn

class GlobalFilterLayer(nn.Module):
    def __init__(self, dim=233472):
        super(GlobalFilterLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(dim)

    def forward(self,x):
        # signal example
        # [batch_size, channel, H, W] = [batch_size, 1, N], where channel is 1 and N = H*W
        B, C, N = x.shape
        x = self.layernorm(x)
        x = torch.fft.fft(x, dim = -1, norm = 'ortho')
        x = torch.abs(x) # you can use other forms
        return x