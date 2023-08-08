  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:36 2019

@author: lhjin
"""
import numpy as np
import os
from skimage import io




def myDataRange(data_path0):
    
    dirs_all  = os.listdir(data_path0) 
    dirs_each = os.listdir(data_path0+dirs_all[0]) 
    all_intensity= np.zeros((len(dirs_all),len(dirs_each)))
    for i in range (len(dirs_all)):
        print(i)
        dirs_each = os.listdir(data_path0+dirs_all[i]) 
        for j in range(len(dirs_each)):
            I = io.imread(data_path0+dirs_all[i]+'/'+dirs_each[j])
            all_intensity[i,j] = I.flatten().max()
    max_intensity= np.zeros((len(dirs_each)))
    for i in range(len(dirs_each)):
        max_intensity[i] = all_intensity[:,i].max()
    return max_intensity

if __name__ == "__main__":


    data_path0='/media/star/EXPDATA/ToLHJ/all/'
    max_intensity = myDataRange(data_path0)
    print(max_intensity)