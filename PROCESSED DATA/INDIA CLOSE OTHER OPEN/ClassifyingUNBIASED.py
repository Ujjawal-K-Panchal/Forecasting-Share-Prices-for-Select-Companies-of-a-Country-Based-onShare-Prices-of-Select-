# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:05:20 2018

@author: uchih
"""

#importing and preprocessing the data
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
dataset = pd.read_csv('UNBIASED.csv')
X = dataset.iloc[:,:].values
Y = dataset.iloc[:,269].values#Company for which analysis is to be done +1
X = X[:,:-1]

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 0)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

kb = SelectKBest(score_func = chi2 , k = 180)
select = kb.fit(np.abs(X_train),Y_train)

X_train = select.transform(X_train)
X_test = select.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, max_features = 16)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)

from sklearn.metrics import  precision_score, accuracy_score, recall_score
print('Confusion Matrix')
print(pd.crosstab(Y_test,Y_pred, margins = True))
print('Accuracy : ',accuracy_score(Y_test,Y_pred)*100, ' %' )
print('Precision : ', precision_score(Y_test,Y_pred, average = 'weighted'))