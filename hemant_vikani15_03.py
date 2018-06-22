# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:45:41 2018

@author: heman
"""

import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Bahubali2_vs_Dangal.csv')
features = dataset.iloc[:,0:1].values
labels_B = dataset.iloc[:,1:2].values

labels_D = dataset.iloc[:,2:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train_B,labels_test_B= train_test_split(features, labels_B, test_size = 0.2, random_state = 0)
features_train, features_test, labels_train_D,labels_test_D= train_test_split(features, labels_D, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor1 = LinearRegression()

regressor.fit(features_train, labels_train_B)
regressor1.fit(features_train, labels_train_D)

# Predicting the Test set results
labels_pred_bahubali = regressor.predict(10)
labels_pred_dangal = regressor1.predict(10)

print labels_pred_bahubali
print labels_pred_dangal

# Model Score
Score = regressor.score(features_test,labels_test_B)
Score1 = regressor.score(features_test,labels_test_D)

# Visualising the Training set results
