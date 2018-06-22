

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 11:22:47 2018

@author: heman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("bluegills.csv")
features=dataset.iloc[:,0:1].values
labels=dataset.iloc[:,1:].values




from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(features,labels)

print "Predicting result with Linear Regression",lin_reg.predict(5)[0]
score1=lin_reg.score(features,labels)

# Visualising the Linear Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, lin_reg.predict(features), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('age')
plt.ylabel('length')
plt.show()


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# create the polynomial features using above class
poln_object = PolynomialFeatures(degree = 6)
features_poln = poln_object.fit_transform(features)


#once you have the poln_matrix read, input it to linear regressor
lin_reg_2 = LinearRegression()
lin_reg_2.fit(features_poln, labels)


print ("Predicting result with Polynomial Regression",)
#need to convert 6.5 into polynomial features
print (lin_reg_2.predict(poln_object.fit_transform(5)))
score2=lin_reg_2.score(features_poln,labels)


# Visualising the Polynomial Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, lin_reg_2.predict(poln_object.fit_transform(features)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('age')
plt.ylabel('length')
plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
features_grid = np.arange(min(features), max(features), 0.1)
features_grid = features_grid.reshape((-1, 1))
plt.scatter(features, labels, color = 'red')
plt.plot(features_grid, lin_reg_2.predict(poln_object.fit_transform(features_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
