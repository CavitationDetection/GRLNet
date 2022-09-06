'''
Description: 
Author: Yu Sha
Date: 2021-09-28 09:30:05
LastEditors: Yu Sha
LastEditTime: 2022-01-29 12:37:07
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from globalfilterlayer import GlobalFilterLayer

class Local_G_Net(nn.Module):
    def __init__(self, dim=233472):
        super(Local_G_Net, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1,out_channels = 64, kernel_size = 3,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True), 

            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),

            nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1),
            nn.ReLU(inplace = True), 
            nn.BatchNorm1d(256),

            nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 3,stride = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(512),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels = 512, out_channels = 256, kernel_size = 3,stride = 1), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True), 

            nn.ConvTranspose1d(in_channels = 256, out_channels = 128, kernel_size = 3,stride = 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(in_channels = 128, out_channels = 64, kernel_size = 3,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(in_channels = 64, out_channels = 1, kernel_size = 3,stride = 1),
            nn.Tanh()
        )
        self.gfl = GlobalFilterLayer(dim)


    def forward(self,x):
        # print('L_G',x.shape)
        x = self.gfl(x)   
        # print('L_G_GFL',x.shape)     
        x = self.encoder(x)
        x = self.decoder(x)
        return x

## you can change the kernel size, such as 5*5.
class Regional_G_Net(nn.Module):
    def __init__(self, dim=233472):
        super(Regional_G_Net,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1,out_channels = 64, kernel_size = 7,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True), 

            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 7, stride = 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),

            nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 7, stride = 1),
            nn.ReLU(inplace = True), 
            nn.BatchNorm1d(256),

            nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 7,stride = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(512),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels = 512, out_channels = 256, kernel_size = 7,stride = 1), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True), 

            nn.ConvTranspose1d(in_channels = 256, out_channels = 128, kernel_size = 7,stride = 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(in_channels = 128, out_channels = 64, kernel_size = 7,stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(in_channels = 64, out_channels = 1, kernel_size = 7,stride = 1),
            nn.Tanh()
        )
        self.gfl = GlobalFilterLayer(dim)

    def forward(self,x):
        # print('R_G',x.shape)
        x = self.gfl(x)
        # print('R_G_GFL',x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

## you can change the kernel size, such as 7*7.
class D_Net(nn.Module):
    def __init__(self, dim=233472):
        super(D_Net,self).__init__()
        self.gfl = GlobalFilterLayer(dim)
        self.discriminator = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 5, stride = 2),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 2),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 5, stride = 2),
            nn.ReLU(True),
            nn.Flatten(),
            # 233472--7469568  80000--2558464  16000--510464 
            nn.Linear(7469568,1),
            nn.Sigmoid()
        )    

    def forward(self,x):
        # print('D',x.shape)
        x = self.gfl(x)
        # print('D_GFL',x.shape)
        x = self.discriminator(x)
        x = x.squeeze(-1)
        return x

# the following codes are used for testing
if __name__ == '__main__':
    input = torch.randn((2, 1, 233472)) # [batch_size, channel,N=H*W]
    model = Local_G_Net()
    # model = Regional_G_Net()
    # model = D_Net()
    out = model(input)
    print('inputshape',input.shape)
    print('outshape',out.shape)
    
