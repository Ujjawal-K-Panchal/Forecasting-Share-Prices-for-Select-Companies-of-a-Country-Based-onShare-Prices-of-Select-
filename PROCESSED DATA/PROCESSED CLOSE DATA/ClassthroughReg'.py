# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 08:47:37 2018

@author: uchih
"""

#importing and preprocessing the data
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
dataset = pd.read_csv('MergedCombinedNA0dFormatCleaned.csv')
X = dataset.iloc[:,:].values
Y = dataset.iloc[:,30+1].values#Company for which analysis is to be done +1
X = X[:,68:]
Y_gamma = (sum(Y)/len(Y))*0.005

Y = Y[7:]
X = X[:-7]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0, test_size = 0.2, shuffle = False)

X_train = np.matrix(X_train, dtype = float)
Y_train = np.matrix(Y_train, dtype = float)
Y_train = Y_train.T

from sklearn.ensemble import RandomForestClassifier
reg = RandomForestRegressor(n_estimators = 15, random_state = 0) # 15 by experimentation.
reg.fit(X_train, Y_train)

#predicting
Y_pred = reg.predict(X_test) # X_test is already transformed

#Now Test to Class
Class = np.array(Y_test)
Class_pred= np.array(Y_pred)
for i in range(0,len(Y_test)-1):
    Class[i] = Y_test[i+1] - Y_test[i]
    if(Class[i] > Y_gamma):
        Class[i] = 1
    elif(-1*Class[i] > Y_gamma):
        Class[i] = -1
    else:
        Class[i] = 0
Class = Class[:-1]

for i in range(0,len(Y_test)-1):
    Class_pred[i] = Y_pred[i+1] - Y_pred[i]
    if(Class_pred[i] > Y_gamma):
        Class_pred[i] = 1
    elif(-1*Class_pred[i] > Y_gamma ):
        Class_pred[i] = -1
    else:
        Class_pred[i] = 0
    
Class_pred = Class_pred[:-1]

#eval
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn import metrics
cm = metrics.confusion_matrix(Class, Class_pred)
print ('Confusion Matrix')
print(cm)
print('Accuracy : ',accuracy_score(Class_pred,Class)*100,'%')

print('Precision : ', precision_score(Class,Class_pred, average = 'macro')*100)
print('Recall : ', recall_score(Class,Class_pred, average = 'macro')*100)
