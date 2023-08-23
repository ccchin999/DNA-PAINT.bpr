#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:30:00 2019

@author: lhjin
"""

import os
import sys
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
import numpy as np

from unet_model import UNet
import warnings
warnings.filterwarnings('ignore')#warnings模块用于忽略警告信息
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):#做一个torch到tensor的转换，tensor是四维的，所以后续需要降维
        data_in, data_in_name = sample['Net_input'], sample['Net_input_name']

        # swap color axis because
        # numpy image: H * W * C
        # torch image: C * H * W
        #image = image.transpose((2, 0, 1))#.transpose 可以一次性调换3个索引的位置 
        #landmarks = landmarks.transpose((2, 0, 1))
        
        #return {'image': image, 'landmarks': torch.from_numpy(landmarks)}
        return {'Net_input': torch.from_numpy(data_in),
               'Net_input_name': data_in_name}
class ReconsDataset(torch.utils.data.Dataset):#就是读图形成sample，这部分就是准备数据集。把一个.npy文件中第一层取出来作为金标准，其他作为input，然后形成一个sample作为网络的input
     
    def __init__(self, test_data_path,transform,image_size):
        self.test_data_path = test_data_path
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):#训练集长度，即后续循环的总个数
        # dirs = os.listdir(self.test_data_path)  # open the files
        return len(self.test_data_path)            # because one of the file is for groundtruth

    def __getitem__(self, idx):
        IN =  io.imread(self.test_data_path)/65535
        sample = {'Net_input': IN, 'Net_input_name': self.test_data_path}
         
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == "__main__":
    # 调用显卡 “0”
    cuda = torch.device('cuda:0')
    batch_size = 1
    test_data_path0 = sys.argv[2]
    filepath = sys.argv[3]
    SRRFDATASET = ReconsDataset(test_data_path = test_data_path0,
                                transform = ToTensor(),
                                image_size = 256)
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop
    model = UNet(n_channels=1, n_classes=1)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    for num in range(1,2):
        model.load_state_dict(torch.load(sys.argv[1]))
        model.eval()
        for batch_idx, items in enumerate(test_dataloader):
            
            image = items['Net_input']
            image_name = items['Net_input_name']
            # print(image_name[0])
            image = image.unsqueeze(dim = 3)
            
            image = np.swapaxes(image, 1,3)
            image = np.swapaxes(image, 2,3)
            image = image.float()
            image = image.cuda(cuda)  
            
            pred = model(image).squeeze()
            pred = pred.detach().cpu().numpy()
            #pred = (pred[0,:]*299+pred[1,:]*587+pred[2,:]*114+500)/1000
            #print(pred.shape)
            # print(pred.max())
            # print(pred.min())
            # pred[pred<0] = 0
            pred[pred<0] = 0
            pred = pred/pred.max()*65535
            
            if  os.path.isdir(os.path.split(filepath)[0]):
                io.imsave(filepath, pred.astype(np.uint16))
            else:
                os.mkdir(os.path.split(filepath)[0])
                io.imsave(filepath, pred.astype(np.uint16))
            
