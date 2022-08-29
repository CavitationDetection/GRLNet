'''
Description: 
Author: Yu Sha
Date: 2021-09-28 09:30:05
LastEditors: Yu Sha
LastEditTime: 2022-02-02 14:08:50
'''
from main import train
from opts import parse_opts
from utils import create_dirs


# train and test our model
if __name__ == '__main__':
    opts = parse_opts()
    create_dirs("./results/positive_6_Regional_GNet")
    create_dirs("./results/positive_6_Local_GNet")
    create_dirs("./results/positive_6_DNet")
    train(opts)
