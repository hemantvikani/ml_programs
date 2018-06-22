# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 10:59:30 2018

@author: heman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('affairs.csv')
features=dataset.iloc[:,0:8].values
labels=dataset.iloc[:,8].values


from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[6])
features=onehotencoder.fit_transform(features).toarray()
features=features[:,1:]
onehotencoder=OneHotEncoder(categorical_features=[11])
features=onehotencoder.fit_transform(features).toarray()
features = features[:,1:]

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(features_train,labels_train)

labels_pred= classifier.predict(features_test)
print dataset["affair"].value_counts(normalize=True)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_pred)

Pred = classifier.predict(np.array([1,0,0,0,0,0,0,1,0,0,3,25,3,1,4,16]).reshape(1,-1))


# Building the optimal model
features=dataset.iloc[:,0:8].values
labels=dataset.iloc[:,8].values

import statsmodels.formula.api as sm
features = np.append(arr = np.ones((6366, 1)).astype(int), values = features, axis = 1)
features_opt = features[:, [0,1,2,3,4,5,6,7,8]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:, [0,1,2,3,5,6,7,8]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:, [0,1,2,3,5,6,7]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

print regressor_OLS.params[0]
print regressor_OLS.params[1]
print regressor_OLS.params[2]
print regressor_OLS.params[3]
print regressor_OLS.params[4]
print regressor_OLS.params[5]
print regressor_OLS.params[6]

