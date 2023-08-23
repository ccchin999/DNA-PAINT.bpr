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
class ReconsDataset(torch.utils.data.Dataset):#就是读图形成sampe，这部分就是准备数据集。把一个.npy文件中第一层取出来作为金标准，其他作为input，然后形成一个sample作为网络的input
     
     def __init__(self, test_data_path, wf_data_path,transform,image_size):
        self.test_data_path = test_data_path
        self.transform = transform
        self.image_size = image_size
        self.wf_data_path = wf_data_path
        
     def __len__(self):#训练集长度，即后续循环的总个数
        return len(self.test_data_path)            # because one of the file is for groundtruth

     def __getitem__(self, idx):
         IN= np.zeros((self.image_size, self.image_size, 4))
         IN[:,:,0] = io.imread(os.path.join(self.wf_data_path))/65535 #-WideField.tif
         IN[:,:,1] = io.imread(os.path.join(self.test_data_path))/65535 #-3000R.tif
         sample = {'Net_input': IN, 'Net_input_name': self.test_data_path}
         if self.transform:
              sample = self.transform(sample)
         return sample

if __name__ == "__main__":
    # 调用显卡 “0”
    cuda = torch.device('cuda:0')
    batch_size = 1
    
    test_data_path0 = sys.argv[2]
    wf_data_path0 = sys.argv[4]
    SRRFDATASET = ReconsDataset(test_data_path = test_data_path0,
                                wf_data_path = wf_data_path0,
                                transform = ToTensor(),
                                image_size = 256)
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop
    model = UNet(n_channels=4, n_classes=1)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    for epoch_num1 in range(1,2):
        # print(epoch_num1)
        model.load_state_dict(torch.load(sys.argv[1]))
        model.eval()
        filepath = sys.argv[3]
        for batch_idx, items in enumerate(test_dataloader):
            
            image = items['Net_input']
            image_name = items['Net_input_name']
            # print(image_name[0])
            # image = image.unsqueeze(dim = 3)
            
            image = np.swapaxes(image, 1,3)
            image = np.swapaxes(image, 2,3)
            image = image.float()
            image = image.cuda(cuda)  
            
            pred = model(image).squeeze()
            pred = pred.detach().cpu().numpy()
            # pred = (pred[0,:,:]*299+pred[1,:,:]*587+pred[2,:,:]*114+500)/1000
            # print(pred.min())
            # print(pred.max())
            pred = pred+abs(pred.min())
            pred = pred/pred.max()*65535
            if  os.path.isdir(os.path.split(filepath)[0]):
                io.imsave(filepath, pred.astype(np.uint16))
            else:
                os.mkdir(os.path.split(filepath)[0])
                io.imsave(filepath, pred.astype(np.uint16))
            
