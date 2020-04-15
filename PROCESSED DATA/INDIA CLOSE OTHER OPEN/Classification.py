# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:38:21 2018

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
Y = dataset.iloc[:,53+1].values#Company for which analysis is to be done +1
X = X[:,68:]

#classifiable computation
YC = np.zeros((4015,1))+20000

#for 7th day in future

for i in range(6,4015):
    YC[i-6] = Y[i] - Y[i-6]

Class = np.array(Y)
YC = YC[0:4010]
for i in range(0,len(YC)):
    if(YC[i] > 0.0 ):
        Class[i] = 1
    #elif(YC[i] < 0.0):
     #  Class[i] = -1
    else :
        Class[i] = 0

Class = Class[:-5]
#Y = Class
X = X[:-5]
'''
#for 1 day in future
for i in range(1,4015):
    YC[i-1] = Y[i] - Y[i-1]

YC = YC[:-1]
Class = np.array(Y)
for i in range(0,len(YC)):
    if(YC[i] > 0.0 ):
        Class[i] = 1
    #elif(YC[i] < 0.0):
     #  Class[i] = -1
    else :
        Class[i] = 0
Class = Class[:-1]
X = X[:-1]
'''
#saving classifiable dataset into two files endog and exog
#np.savetxt('endog.csv',X,delimiter = ',')
#np.savetxt('exog.csv',Y,delimiter = ',')

#splitting the data intro training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Class,random_state = 0, test_size = 0.2)
X_train = np.matrix(X_train, dtype = float)
Y_train = np.matrix(Y_train, dtype = float)
Y_train = Y_train.T

#Modelling Logistic Regression:-
#from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()
#logreg.fit(X_train,Y_train)

#prediction
#Y_pred = logreg.predict(X_test)

#evaluation

#score = logreg.score(X_test, Y_test)
#print(score)
#from sklearn import metrics
#cm = metrics.confusion_matrix(Y_test, Y_pred)
#print(cm)

#Modelling SVMs
#Standard Scaling for SVMs
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#Y_train = sc_y.fit_transform(Y_train)
'''
#from sklearn import svm
#SVM = svm.SVC()
#SVM.fit(X_train,Y_train)

#Y_pred = SVM.predict(X_test)
#from sklearn import metrics
#cm = metrics.confusion_matrix(Y_test, Y_pred)
#print(cm)
#print(SVM.score(X_test,Y_test))

#Random Forest Classification

#Accuracy : 76-81 %
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
poly  = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
rfc = RandomForestClassifier(n_estimators = 50, max_features = 95)
rfc.fit(X_train_poly,Y_train)
Y_pred = rfc.predict(poly.fit_transform(X_test))
from sklearn import metrics
cm = metrics.confusion_matrix(Y_test, Y_pred)
print ('Confusion Matrix')
print(cm)
print('Accuracy : ',rfc.score(poly.fit_transform(X_test),Y_test)*100,'%')

#Multi Layered Perceptron
'''
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier(hidden_layer_sizes = (900,3))
mlpc.fit(X_train_poly,Y_train)
Y_pred = mlpc.predict(poly.fit_transform(X_test))
print(mlpc.score(poly.fit_transform(X_test),Y_test)*100,'%')
'''