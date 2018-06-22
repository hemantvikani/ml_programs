# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:24:49 2018

@author: heman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:13:18 2018

@author: Kunal
"""

# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Foodtruck.csv')
features = dataset.iloc[:,0:-1].values
labels = dataset.iloc[:,1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = regressor.predict(features_test)
jaipur_outlet= regressor.predict(3.073)

# Model Score
Score = regressor.score(features_test,labels_test)

# Visualising the Training set results
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Training set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()

# Visualising the Test set results
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Test set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()