# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:02:23 2019

@author: RicardoEspinosa
"""

import cv2
import numpy as np
import os

training_x1 =[]
training_x2 =[]
training_x3 =[]
training_x4=[]
training_y =[]
testing_x1 = []
testing_x2 = []
testing_x3 = []
testing_y = []


Path = "images_06092020/"
PathDBG = "bandgaps.csv"
csv = np.genfromtxt(PathDBG)
row = []
for i in range(0,12500):
    #
    img1 = cv2.imread(Path+str(i)+'_yx.jpg', cv2.IMREAD_COLOR)
    #img2 = cv2.imread(Path+str(i)+'_yz.jpg', cv2.IMREAD_COLOR)
    #img3 = cv2.imread(Path+str(i)+'_zx.jpg', cv2.IMREAD_COLOR)
    #img4 = cv2.imread(Path+str(i)+'_yx.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.resize(img1, None, fx=0.10, fy=0.10)
    #img2 = cv2.resize(img2,None,fx=0.10,fy=0.10)
    #img3 = cv2.resize(img3,None,fx=0.10,fy=0.10)
    #img4 = cv2.resize(img4,None,fx=0.50,fy=0.50)
    if i == 150:
      break
      
    if(i < 100):
        training_x1.append(img1)
        #training_x2.append(img2)
        #training_x3.append(img3)
        #training_x4.append(img4)
        training_y.append(csv[i])
    else:
        testing_x1.append(img1)
        #testing_x2.append(img2)
        #testing_x3.append(img3)
        #testing_x4.append(img4)
        testing_y.append(csv[i])
        
training_x1 = np.array(training_x1)
#training_x2 = np.array(training_x2)
#training_x3 = np.array(training_x3)

training_y = np.array(training_y)
testing_x1 = np.array(testing_x1)
#testing_x2 = np.array(testing_x2)
#testing_x3 = np.array(testing_x3)

testing_y = np.array(testing_y)   

print(training_x1.shape)
"""
training_x1 = training_x1.reshape(10000,196,268,1)
training_x2 = training_x2.reshape(10000,196,268,1)
training_x3 = training_x3.reshape(10000,196,268,1)

testing_x1 = testing_x1.reshape(2500,196,268,1)
testing_x2 = testing_x2.reshape(2500,196,268,1)
testing_x3 = testing_x3.reshape(2500,196,268,1)
"""
            
'''
        img = cv2.imread('images2/'+file, cv2.IMREAD_COLOR)
        row.append(img)
        print(file)
        if(len(row)==3):
            if(i < 8750):
                training_x.append(row)
                training_y.append(csv[i])
            else:
                testing_x.append(row)
                testing_y.append(csv[i])
            i = i+1
            row=[]
          
        
training_x = np.array(training_x)
training_y = np.array(training_y)
testing_x = np.array(testing_x)
testing_y = np.array(testing_y)
'''