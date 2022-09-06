'''
Description: 
Author: Yu Sha
Date: 2021-03-12 13:11:55
LastEditors: Yu Sha
LastEditTime: 2022-01-24 16:57:30
'''
import pandas as pd
import numpy as np
import torch
import os

"""
For using 'opts.train_label_path':

    without any changes

    In 'main.py', we use the following codes:
        for idx, seq, _ in data_loader_train:
        seq = seq.to(device)
        seq = seq.unsqueeze(1)

For not using 'opts.train_label_path':

    You need to delete the following code:
    self.labeldict = self.labelDict()  

    def labelDict(self):
        label_df = pd.read_csv(self.opt.train_label_path)
        labels = label_df['label'].values
        index2label = {}
        for i, label in enumerate(labels):
            index2label[i] = label
        return index2label
    
    label = self.labeldict[labelIndex] 
    if str(label) == "0":
        label = [0]
    elif str(label) == "1":
        label = [1]
    elif str(label) == "2":
        label = [2]
    elif str(label) == "3":
        label = [3]
    label = torch.as_tensor(label, dtype=torch.int64) 
And 
    you need to use the following codes:
        return idx, infodata 

And
    In 'main.py', we use the following codes:
        for idx, seq in data_loader_train:
        seq = seq.to(device)
        seq = seq.unsqueeze(1)
"""

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.train_root
        self.csv_files = [file for file in list(sorted(os.listdir(self.root))) if '.csv' in file] 
        self.labeldict = self.labelDict()  

    
    def labelDict(self):
        label_df = pd.read_csv(self.opt.train_label_path)
        labels = label_df['label'].values
        index2label = {}
        for i, label in enumerate(labels):
            index2label[i] = label
        return index2label

    def getFileIndex(self, csv_file):
        return int(csv_file.strip().replace('.csv', ''))

    def trunate_and_pad(self, data, pad=0):
        "Data truncation and padding"
        if len(data) >= self.opt.mess_length:
            data = data[:self.opt.mess_length]
        else:
            padding = [pad] * (self.opt.mess_length - len(data))
            data += padding
        assert len(data) == self.opt.mess_length
        return data

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.csv_files[idx])
        infodata = pd.read_csv(file_path, header=None)
        infodata = infodata[0].tolist()
        infodata = self.trunate_and_pad(infodata)
        infodata = torch.as_tensor(infodata, dtype=torch.float32)
        labelIndex = self.getFileIndex(self.csv_files[idx]) 
        label = self.labeldict[labelIndex] 
        if str(label) == "0":
            label = [0]
        elif str(label) == "1":
            label = [1]
        elif str(label) == "2":
            label = [2]
        elif str(label) == "3":
            label = [3]
        label = torch.as_tensor(label, dtype=torch.int64) 
        return idx, infodata, label  
        # return idx, infodata 

    def __len__(self):
        return len(self.csv_files)



        
