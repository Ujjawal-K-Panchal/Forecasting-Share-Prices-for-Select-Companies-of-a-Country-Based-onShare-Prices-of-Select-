# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:14:58 2018

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
Y = dataset.iloc[:,1+1].values#Company for which analysis is to be done +1
X = X[:,68:]
#X = ohe.fit_transform(X).toarray()

#splitting the data intro training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0, test_size = 0.2)
X_train = np.matrix(X_train, dtype = float)
Y_train = np.matrix(Y_train, dtype = float)
Y_train = Y_train.T

#Feature Scaling. Because not defaultly done by SVR class.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)

#modelling
from sklearn.svm import SVR
reg = SVR(kernel = 'rbf') 
reg.fit(X_train, Y_train)

#predicting
Y_pred = sc_y.inverse_transform(reg.predict(X_test)) # X_test is already transformed

#Evaluating
from sklearn import metrics 
print(metrics.mean_absolute_error(Y_test,Y_pred))
print(metrics.mean_squared_error(Y_test,Y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

#plotting first 100 test points (red) vs model prediction (blue)
length = range(0,100)
plt.scatter(length, Y_test[0:100] , color = 'red')
plt.plot(length , Y_pred[0:100] , color = 'blue')
print(sum(Y_test)/len(Y_test))
