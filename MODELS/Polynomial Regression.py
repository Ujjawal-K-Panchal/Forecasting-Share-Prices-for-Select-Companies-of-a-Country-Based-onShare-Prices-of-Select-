# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:49:22 2018

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
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 10, test_size = 0.2)
X_train = np.matrix(X_train, dtype = float)
Y_train = np.matrix(Y_train, dtype = float)
Y_train = Y_train.T
#Building an all in model for analysis.
#'''
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
#X_poly = poly.fit_transform(X_train)
from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile = 87) # Result of experimentation.
select.fit(poly.fit_transform(X_train),Y_train)
X_train_selected = select.transform(poly.fit_transform(X_train))
reg = sm.OLS(endog = Y_train, exog = X_train_selected).fit()
X_test_selected = select.transform(poly.fit_transform(X_test))
Y_pred = reg.predict(X_test_selected)
Y_pred = np.matrix(Y_pred, dtype = float).T
#Model evaluation
reg.summary()
#'''
'''
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
select = SelectPercentile(percentile = 50)
select.fit(X_train, Y_train)
X_train_selected = select.transform(X_train)
lr.fit(X_train_selected, Y_train)
X_test_selected = elect.transform(X_test)
Y_pred = lr.predict(X_test)
'''
#Evaluating
from sklearn import metrics 
print(metrics.mean_absolute_error(Y_test,Y_pred))
print(metrics.mean_squared_error(Y_test,Y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
length = range(0,100)
plt.scatter(length, Y_test[0:100] , color = 'red')
plt.plot(length , Y_pred[0:100] , color = 'blue')
print(sum(Y_test)/len(Y_test))
