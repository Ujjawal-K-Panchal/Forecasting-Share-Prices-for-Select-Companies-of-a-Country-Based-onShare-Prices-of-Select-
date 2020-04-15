# -*- coding: utf-8 -*-
"""
Created on Wed Jun 2dif 15:26:29 2018
@author: Ujjawal.K.Panchal
"""
#Changes made : DIF variable. Days.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('MergedCombinedNA0dFormatCleaned.csv')

X = dataset.iloc[:,68:].values
Y = dataset.iloc[:,2+1].values

#Days in future: DIF
Y_gamma =  (sum(Y)/len(Y)) * 0.05

Y_theta = np.array(Y) # Will contain
# Days in future : (30)
dif = 7
for i in range(0,4015-dif):
    Y_theta[i] = Y[i+dif]-Y[i]

Y_theta = Y_theta[:-dif]

Y_theta_Class = np.array(Y_theta)
for i in range(0,len(Y_theta)):
    if(Y_theta[i] > Y_gamma ):
        Y_theta_Class[i] = 1
    elif(-1*Y_theta[i] > Y_gamma):
         Y_theta_Class[i] = -1
    else:
        Y_theta_Class[i] = 0
Class = np.array(Y_theta_Class)
X = X[:-dif]
#splitting the dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Class, test_size = 0.2)

#Classification Modelling
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 15, max_features = 5, max_depth = 10)
rfc.fit(X_train,Y_train)

#training error
Y_pred = rfc.predict(X_train)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print('Confusion Matrix')
cm = confusion_matrix(Y_train, Y_pred)
print(pd.crosstab(Y_train,Y_pred))
print('Accuracy : ',rfc.score(X_train,Y_train) )
print('Precision : ', precision_score(Y_train, Y_pred, average = 'macro'))
print('Recall : ', recall_score(Y_train, Y_pred, average = 'macro'))
print('F1 Score : ', f1_score(Y_train, Y_pred , average = 'macro'))
print(rfc.predict([dataset.iloc[0,68:]]))

#prediction
Y_pred = rfc.predict(X_test)

#Evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print('Confusion Matrix')
cm = confusion_matrix(Y_test, Y_pred)
print(pd.crosstab(Y_test,Y_pred))
print('Accuracy : ',rfc.score(X_test,Y_test) )
print('Precision : ', precision_score(Y_test, Y_pred, average = 'macro'))
print('Recall : ', recall_score(Y_test, Y_pred, average = 'macro'))
print('F1 Score : ', f1_score(Y_test, Y_pred , average = 'macro'))
print(rfc.predict([dataset.iloc[0,68:]]))