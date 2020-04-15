# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:16:17 2018

@author: Ujjawal Panchal
"""

import pandas as pd
import numpy as np

#Allocating Data
dataset = pd.read_csv('MergedCombinedNA0dFormatCleaned.csv')
X = dataset.iloc[:,68:]
Y = dataset.iloc[:,66+1].values
Y_gamma = (sum(Y)/len(Y))*0.02
X_new = X[1:]
Y_old = Y[:-1]
Y_new = Y[1:]
#X = pd.concat([X_new,X.iloc[:-1,:]], axis = 1)
Class = np.array(Y_new)
for i in range(0,len(Y_new)):
    if(Y_new[i]-Y_old[i] > Y_gamma):
        Class[i] = 1
    elif(Y_old[i]-Y_new[i] > Y_gamma):
        Class[i] = -1
    else:
        Class[i] = 0
