# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 18:11:54 2022
"""
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch,torchvision

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 64, 3,padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3,padding=1)
   
        
        self.conv3 = torch.nn.Conv2d(64, 128, 3,padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, 3,padding=1)
        
        
        self.conv5 = torch.nn.Conv2d(128, 256, 3,padding=1)
        self.conv6 = torch.nn.Conv2d(256, 256, 3,padding=1)

        #self.dropout = torch.nn.Dropout2d(p = 0.1) 
        self.pool = torch.nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        
        x = self.conv1(x)
        x = torch.nn.functional .relu(x)
        x = self.conv2(x)
        ftrs.append(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = torch.nn.functional .relu(x)
        x = self.conv4(x)
        ftrs.append(x)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = torch.nn.functional .relu(x)
        x = self.conv6(x)
        ftrs.append(x)
        x = self.pool(x)
    
        return ftrs


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convTr1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2,stride=2)
        self.conv1 = torch.nn.Conv2d(256, 128, 3,padding=1)
        
        self.convTr2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2,stride=2)
        self.conv2 = torch.nn.Conv2d(128, 64, 3,padding=1)
     
        
    def forward(self, x, encoder_features):
        
        
        x = self.convTr1(x)
        enc_ftrs = self.crop(encoder_features[0], x)  
        x        = torch.cat([x, enc_ftrs], dim=1)
        x        = self.conv1(x)
        x = torch.nn.functional .relu(x)
        
        x = self.convTr2(x)
        enc_ftrs = self.crop(encoder_features[1], x)  
        x        = torch.cat([x, enc_ftrs], dim=1)
        x        = self.conv2(x)
        x = torch.nn.functional .relu(x)
        return x
        
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(torch.nn.Module):
  
    def __init__(self, num_class=1, retain_dim=False):
        super().__init__()
        self.encoder     = Encoder()
        self.decoder     = Decoder()
        self.head        = torch.nn.Conv2d(64, num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = torch.nn.functional.interpolate(out, (512,512))
        return out