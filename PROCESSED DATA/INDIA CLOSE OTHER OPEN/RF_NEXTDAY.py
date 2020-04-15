# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:48:32 2018

@author: Ujjawal Panchal
Next Day Random Forest with unshuffled and shuffled testing.
"""

#importing and preprocessing the data
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
dataset = pd.read_csv('MergedCombinedNA0d.csv')
#X = dataset.iloc[:,2:].values
Y = dataset.iloc[:,10+1].values#Company for which analysis is to be done +1
X = pd.concat([dataset.iloc[:,2:13],dataset.iloc[:,(13)+1:]],axis = 1)
#X = ohe.fit_transform(X).toarray()

#Now for predicting results for the next day, 1st row of Y and last row of X are deleted.
#X = X[:-7,:]
#Y = Y[7:]
'''
original:-
1   -   1
### -> #
### -> #
### -> #
### -> #

now:-
        #(deleted)
### ->  #
### ->  #
### ->  #
###(deleted)
'''
#splitting the data intro training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0, test_size = 0.2)
X_train = np.matrix(X_train, dtype = float)
Y_train = np.matrix(Y_train, dtype = float)
Y_train = Y_train.T

from sklearn.model_selection import train_test_split
Xut_train,Xut_test,Yut_train,Yut_test = train_test_split(X,Y,random_state = 0, test_size = 0.2, shuffle = False)
Xut_train = np.matrix(Xut_train, dtype = float)
Yut_train = np.matrix(Yut_train, dtype = float)
Yut_train = Yut_train.T


#Feature Scaling. Because not defaultly done by SVR class.
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)
'''
#modelling
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 15, random_state = 0) # 15 by experimentation.
reg.fit(X_train, Y_train)

#predicting
Y_pred = reg.predict(X_test) # X_test is already transformed
#Yut_pred = reg.predict(Xut_test)
#Evaluating
print('Shuffled Prediction Evaluation :-')
from sklearn import metrics 
print(metrics.mean_absolute_error(Y_test,Y_pred))
print(metrics.mean_squared_error(Y_test,Y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
print(sum(Y_test)/len(Y_test))

#CUSTOM PREDICTION : -
ans = reg.predict()
#plotting first 100 test points (red) vs model prediction (blue)
#length = range(0,100)
#plt.scatter(length, Y_test[0:100] , color = 'red')
#plt.plot(length , Y_pred[0:100] , color = 'blue')
#

#Unshuffled Testing Evaluation
'''
print('Unshuffled Testing Prediction Evaluation :-')
from sklearn import metrics 
print(metrics.mean_absolute_error(Yut_test,Yut_pred))
print(metrics.mean_squared_error(Yut_test,Yut_pred))
print(np.sqrt(metrics.mean_squared_error(Yut_test,Yut_pred)))
'''
#plotting first 100 test points (red) vs model prediction (blue)
print('Unshuffled Testing Prediction Visualization :-')
length = range(0,803)#802
plt.scatter(length, Yut_test , color = 'red')
plt.plot(length , Yut_pred , color = 'blue')
print(sum(Yut_test)/len(Yut_test))


