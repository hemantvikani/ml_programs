# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 11:29:18 2018

@author: heman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 12:06:19 2018

@author: Kunal
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iq_size.csv')
features = dataset.iloc[:,1:].values
labels = dataset.iloc[:, 0:1].values

# Encoding categorical data
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()'''

# Avoiding the Dummy Variable Trap
#features = features[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(features_train, labels_train)

# Predicting the Test set results
Pred = regressor.predict(np.array([90,70,150]).reshape(1,-1))

# Getting Score for the Multi Linear Reg model
Score = regressor.score(features_train, labels_train)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
features = np.append(arr = np.ones((38, 1)).astype(int), values = features, axis = 1)
features_opt = features[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
features_opt = features[:, [0, 1,2]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
features_opt = features[:, [ 1,2]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
features_opt = features[:, [1]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()





