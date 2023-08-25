#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20211202
对于网络进行固定细胞的测试，97个测试样品。记录每个输入张数下每epoch对于97个测试样品的表现，并统计97个样品之间的p值。
@author: cjc
"""
from xlwt import *
import numpy as np
import os
import math
import sys
from skimage import io, transform
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
def psnr(img1, img2):#峰值信噪比
    img1 = np.array(img1)
    img2 = np.array(img2)
    img1 = (img1/np.amax(img1))*255
    img2 = (img2/np.amax(img2))*255
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    # img1 = img1/img1.max()
    # img2 = img2/img2.max()
    # return compare_psnr(img1,img2,data_range=1)

def nrmse(img1, img2,type='sd'):#归一化均方根误差,灰度值相似度
    img1 = np.array(img1)
    img2 = np.array(img2)
    img1 = (img1/np.amax(img1))*255
    img2 = (img2/np.amax(img2))*255
    #最大值归一化
    # img1 = img1/img1.max()
    # img2 = img2/img2.max()

    #计算nrmse
    mse = np.mean( (img1 - img2) ** 2 )
    rmse = math.sqrt(mse)
    if type == "sd":
        nrmse = rmse/np.std(img1)
    if type == "mean":
        nrmse = rmse/np.mean(img1)
    if type == "maxmin":
        nrmse = rmse/(np.max(img1) - np.min(img1))
    if type == "iq":
        nrmse = rmse/ (np.quantile(img1, 0.75) - np.quantile(img1, 0.25))
    if type not in ["mean", "sd", "maxmin", "iq"]:
        print("Wrong type!")
    return nrmse
    # img1 = img1/img1.max()
    # img2 = img2/img2.max()

    # nrmse=compare_nrmse(img1,img2,normalization='euclidean')
    # return nrmse

# def ssim(img1, img2, data_range):#结构相似度
#     # if img2.min() < 0:
#     #   img2 += abs(img2.min())
#     if data_range is None:
#         score = compare_ssim(img1, img2)
#     else:
#         score = compare_ssim(img1, img2, data_range = data_range)
#     return score

def ssim(img1, img2, data_range):#结构相似度
    #if img2.min() < 0:
    #   img2 += abs(img2.min())
    # img2 = (img2/img2.max()) * img1.max()
    #img1 = (img1/img1.max()) * 255
    # img2= img2+abs(img2.min())
    img1 = np.array(img1)
    img2 = np.array(img2)
    img2 = (img2/img2.max()) * img1.max()
    if data_range is None:
        score = compare_ssim(img1, img2)
    else:
        score = compare_ssim(img1, img2, data_range = data_range)
    return score




if __name__ == "__main__":
    print("PSNR,RMSE,SSIM")
    gt=io.imread(sys.argv[1])
    im=io.imread(sys.argv[2])
    p=psnr(gt,im)
    r=nrmse(gt,im,'sd')
    s=ssim(gt,im,np.max(gt))
    print("%f,%f,%f" % (p,r,s))
