# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:56:12 2018

@author: uchihamadara
"""
#importing and preprocessing the data
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
dataset = pd.read_csv('MergedCombinedNA0dFormatCleaned.csv')
data = dataset.drop('India 66', axis = 1)
X = dataset.iloc[:,:].values
Y = dataset.iloc[:,67].values
X = X[:,68:]
#X = ohe.fit_transform(X).toarray()

#splitting the data intro training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0, test_size = 0.2)
X_train = np.matrix(X_train, dtype = float)
Y_train = np.matrix(Y_train, dtype = float)
Y_train = Y_train.T

#Building an all in model using the statsmodels for analysis.
import statsmodels.formula.api as sm
reg = sm.OLS(endog = Y_train, exog = X_train).fit() 
Y_pred = reg.predict(X_test)

#Model evaluation
reg.summary()

#Evaluating
from sklearn import metrics 
print("Mean Absolute Error : ",metrics.mean_absolute_error(Y_test,Y_pred))
print("Mean Squared Error : ",metrics.mean_squared_error(Y_test,Y_pred))
print("Root Mean Squared Error : ",np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
Y_pred = lr.predict(X_test)

from sklearn import metrics 
print("Mean Absolute Error : ",metrics.mean_absolute_error(Y_test,Y_pred))
print("Mean Squared Error : ",metrics.mean_squared_error(Y_test,Y_pred))
print("Root Mean Squared Error : ",np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
'''