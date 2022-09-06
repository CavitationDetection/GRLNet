'''
Description: 
Author: Yu Sha
Date: 2022-01-24 14:36:20
LastEditors: Yu Sha
LastEditTime: 2022-01-24 15:47:23
'''
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_dirs(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def gaussian(x, snr, mean=0,stddev=1):
    ps = torch.sum(torch.abs(x)**2) / x.shape[2]
    pn = ps/(10**((snr/10)))
    noise = x.data.new(x.size()).normal_(mean, stddev)
    seq_add_noise = x + noise
    return seq_add_noise
