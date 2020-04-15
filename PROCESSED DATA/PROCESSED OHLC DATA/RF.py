# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:14:58 2018

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
dataset = pd.read_csv('MergedCombinedNA0dFormatCleaned.csv')
X = dataset.iloc[0:-1,68:].values
Y = dataset.iloc[1:,20+1].values#Company for which analysis is to be done +1

#X = ohe.fit_transform(X).toarray()

#splitting the data intro training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0, test_size = 0.2)
X_train = np.matrix(X_train, dtype = float)
Y_train = np.matrix(Y_train, dtype = float)
Y_train = Y_train.T

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
reg = RandomForestRegressor(n_estimators = 100, max_depth = None  , random_state = 0)
reg.fit(X_train, Y_train)

#predicting
Y_pred = reg.predict(X_test) # X_test is already transformed
Y_train_pred = reg.predict(X_train)
#Evaluating Test Error
from sklearn import metrics
print('Evaluating Test Error')
print(metrics.mean_absolute_error(Y_test,Y_pred))
print(metrics.mean_squared_error(Y_test,Y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
print('R^2 Value : ', metrics.r2_score(Y_test,Y_pred))
r_sq = metrics.r2_score(Y_test,Y_pred)
adj_r2 = 1 - (1-r_sq)*(len(Y_test)-1)/(len(Y_test)-X.shape[1]-1)
print('Adj R^2 Value : ', adj_r2)
#Evaluating Train Error
print('Evaluating Train Error')
print(metrics.mean_absolute_error(Y_train, Y_train_pred))
print(metrics.mean_squared_error(Y_train,Y_train_pred))
print(np.sqrt(metrics.mean_squared_error(Y_train, Y_train_pred)))
print('R^2 Value : ',metrics.r2_score(Y_train,Y_train_pred))
r_sq = metrics.r2_score(Y_train,Y_train_pred)
adj_r2 = 1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1)
print('Adj R^2 Value : ', adj_r2)

#plotting first 100 test points (red) vs model prediction (blue)
'''
length = range(0,100)
plt.scatter(length, Y_test[0:100] , color = 'red')
plt.plot(length , Y_pred[0:100] , color = 'blue')
print(sum(Y_test)/len(Y_test))
'''
'''
#from https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd
#Feature Importance
feature_list = dataset.columns
feature_list = feature_list[68:]
importances = list(reg.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 9)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
'''