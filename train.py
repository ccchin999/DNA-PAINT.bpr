  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:36 2019

@author: lhjin
"""

train_data_path0='/home/chin/Experiments/DNA-PAINT/dataset'
model_load="/home/user/DNA-PAINT/3000R.pkl"
model_save="/home/user/personal/model/path"
epoch_num=2000

from xlwt import *
import numpy as np
import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform

from unet_model import UNet


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):#做一个torch到tensor的转换，tensor是四维的，所以后续需要降维
        data_in, data_out = sample['Net_input'], sample['Net_gt']

        # swap color axis because
        # numpy image: H * W * C
        # torch image: C * H * W
        #image = image.transpose((2, 0, 1))#.transpose 可以一次性调换3个索引的位置 
        #landmarks = landmarks.transpose((2, 0, 1))
        
        #return {'image': image, 'landmarks': torch.from_numpy(landmarks)}
        return {'Net_input': torch.from_numpy(data_in),
               'Net_gt': torch.from_numpy(data_out)}

class ReconsDataset(torch.utils.data.Dataset):#就是读图形成sampe，这部分就是准备数据集。把一个.npy文件中第一层取出来作为金标准，其他作为input，然后形成一个sample作为网络的input
     
     def __init__(self, train_data_path,transform,image_size):
        self.train_data_path = train_data_path
        self.transform = transform
        self.image_size = image_size
        self.dirs_data = os.listdir(self.train_data_path)
        
     def __len__(self):#训练集长度，即后续循环的总个数
        dirs = os.listdir(self.train_data_path)  # open the files
        return len(dirs)            

     def __getitem__(self, idx):#idx相当于i用于循环。。。 此循环中sample个数一共1135个  
         IN =  io.imread(os.path.join(self.train_data_path, self.dirs_data[idx], "sparsely-localized.tif"))
         IN = IN/65535
         GT =  io.imread(os.path.join(self.train_data_path, self.dirs_data[idx], "ground-truth.tif"))
         GT = GT/65535
         sample = {'Net_input': IN, 'Net_gt': GT}
         
         if self.transform:
              sample = self.transform(sample)
         return sample

def get_learning_rate(epoch):
    limits = [3, 8, 12]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1#assert是为了测试程序的偶然性错误，其实就相当于一个if语句，如果成立则继续运行，否则终止
    for lim, lr in zip(limits, lrs):# for a,b in c  这样的形式表示的是把c的元素付给a b，不过得是c有两列元素。
        if epoch < lim:
            return lr * learning_rate
            
        else:
            return lrs[-1] * learning_rate


if __name__ == "__main__":

# 调用显卡 “0”
    cuda = torch.device('cuda:0')
    #初始化学习率
    learning_rate = 0.001
    # momentum = 0.99
    # weight_decay = 0.0001
    batch_size = 1
    #形成自定义的输入数据集
    net_input_size=1
    SRRFDATASET = ReconsDataset(train_data_path = train_data_path0,
                                transform = ToTensor(),
                                image_size = 256)
    train_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop
    model = UNet(n_channels=net_input_size, n_classes=1)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))#.parameters()shi 用来返回net网络中的参数，sum后就是网络中参数的总数目
    model.cuda(cuda)#在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，而是需要在程序中显示指定。调用model.cuda()，可以将模型加载到GPU上去。这种方法不被提倡，而建议使用model.to(device)的方式，这样可以显示指定需要使用的计算资源，特别是有多个GPU的情况下。
    model.load_state_dict(torch.load(model_load))
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))# torch.optim.Adam 一个实现各种优化算法的包。model.parameters()用于查看网络参数，多见于优化器的初始化

    for epoch in range(1,(epoch_num+1)):#epoch：对所有的训练数据进行一次完整的训练，称为 一轮训练
        lr = get_learning_rate(epoch)#确定此轮训练的学习率
        for p in optimizer.param_groups:#动态修改学习率
            p['lr'] = lr
            print("learning rate = {}".format(p['lr']))#.format（）括号内的值填入前面大括号中
            
        for batch_idx, items in enumerate(train_dataloader):
            
            image = items['Net_input']
            gt = items['Net_gt']
            image = image.unsqueeze(dim = 3)
            gt = gt.unsqueeze(dim = 3)
            # print(image.size())
            #如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train(),而在测试时添加model.eval() 
            model.train()
            #因为image 和 gt经过模型训练后会变成四维的（输入进去训练的时候就是tensor四维的）所以进行降维
            image = np.swapaxes(image, 1,3)#改变图像数组的排列
            image = np.swapaxes(image, 2,3)
            image = image.float()#转换为浮点型
            image = image.cuda(cuda)    #放到GPU
            
            gt = np.swapaxes(gt, 1,3)#改变图像数组的排列
            gt = np.swapaxes(gt, 2,3)
            gt = gt.squeeze()
            gt = gt.float()
            gt = gt.cuda(cuda)
            
            pred = model(image).squeeze()#进行前向传播
            # print(pred.shape)
            # print(gt.shape)
            loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()#计算loss
            #这三布通常一起出现，==
            optimizer.zero_grad()#将模型中梯度设为0
            loss.backward()#loss来定义损失函数是要确定优化目标是什么，然后以目标为头，进行链式法则和反向传播
            optimizer.step()#用来更新参数

            print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))
        if epoch%10==0:#判断epoch，如果能被十整除就保存这一轮的模型，
            epoch1=str(epoch)+str('.pkl') 
            model_save_path = os.path.join(model_save,epoch1)#拼接存储模型的路径及文件名后缀
            torch.save(model.state_dict(),model_save_path)#torch.save用于在某路径下保存参数，包括明星参数，优化器参数等。model.state_dict()用于查看网络参数，多见于模型的保存（和保存一起出现就是保存模型的参数）
