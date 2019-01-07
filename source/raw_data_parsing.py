#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 10:03:27 2018

@author: zili
"""
##########################################################################################
###Don't brutally run this file. This is only for showinmg how data parsing works      ###
###Unzip raw data (csv and foil_figure) in data/raw_data and modify directory instead  ###
###"Real raw data" are .txt directly downloaded from UIUC Airfoil Coordinates Dataset  ###
###CSV files: generated cf/cd results from Xflr5, with disorganized structure and heads###
###Foil_figure: generated images from corrdinates, and filled-in and then binarized    ###
###This program reads, parses, matches and stores organized data in 1 .mat file        ###
###data_x: n by 16384 0/1 matrix, n is #samples, 16384 is 128*128 square flatterned    ###
###data_y: n by 1 matrix, storing all samples' cf/cd ratios calculated from Xflr5      ###
###Normlization_factor: 1 float number normalizing data_y. More details in report      ###
##########################################################################################
import numpy as np
import pandas
import glob
import os
import skimage.io
import scipy.io

#read labels
route= glob.glob(r'./test_data/labels/*.csv')
route_label_list = [os.path.basename(x)[:-4] for x in route]
label = []
for route_label in route_label_list:
    a = pandas.read_csv('./test_data/labels/'+route_label+'.csv',delimiter=',')
    data = a.values
    
    start, end= route_label.split('_')
    
    start = int(start)
    end = int(end)
    
    
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            if data[i,j] != ' ' and j%3 == 1:
                filename = start + j//3
                angle = float(data[i,j-1])
                value = float(data[i,j])
                print('Sample: %d | Angle: %.2lf | Value: %lf'%(filename,angle,value))
                if angle % 1 == 0 and -100 < value < 500:
                    # value < 1000 to rule out bad points
                    if angle == 0:
                        label.append([str(filename) + '_' + '0',value])
                    elif angle < 0:
                        label.append([str(filename) + '_n' + str(-int(angle)),value])
                    else:
                        label.append([str(filename) + '_p' + str(int(angle)),value])
    
print('...............................................................................')  
label = np.array(label)    
labelName = [x for x in label[:,0]]
labelDic = {label[i,0]:float(label[i,1]) for i in range(label.shape[0])}

#read image
route_image = glob.glob(r'./test_data/images/figure/*/*.png')
imageName = [os.path.basename(x)[:-9] for x in route_image]

#match labels and images
commonName = set(labelName) & set(imageName)
pairName = [x for x in commonName]

#build dataset

"""
input - a (n, 128^2) image matrix
output - a normalized (n,1) Cl/Cd Ratio matrix, normalized by ymax

"""
data_x = []
data_y = []
for name in pairName:
    data_y.append(labelDic[name])
    folderName, _ = name.split('_')
    img = skimage.io.imread('./test_data/images/figure/'+ folderName+'/'+ name + '_data.png')
    im = img/65535
    data_x.extend(im.reshape(1,-1))
data_y = np.array([data_y]).reshape(-1,1)
ymax = np.max(data_y)
data_y = data_y/ymax
data_x = np.array(data_x)
scipy.io.savemat('all_data',{'data_x':data_x, 'data_y':data_y, 'Normalization_Factor':[ymax]})















