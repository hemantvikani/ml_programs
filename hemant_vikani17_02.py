# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 12:25:11 2018

@author: heman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_fwf("Auto_mpg.txt",header=None)

dataset.columns=["mpg","cylinder","displacement","horsepower","weight","acceleration","model year","origin","car name"]

dataset["horsepower"]=dataset['horsepower'].replace(["?"],dataset["horsepower"].value_counts().index[0] )
f_mpg=dataset.groupby(["mpg"])
print f_mpg["car name"].value_counts().index[-1]


dataset["horsepower"]=pd.to_numeric(dataset["horsepower"])
features=dataset.iloc[:,1:8].values
labels=dataset.iloc[:,0].values
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)








from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(features_train,labels_train)

score=regressor.score(features_test,labels_test)

Pred = regressor.predict(np.array([6,215,100,2630,22.2,80,3]).reshape(1,-1))





from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(features_train,labels_train)

score=regressor.score(features_test,labels_test)
Pred1 = regressor.predict(np.array([6,215,100,2630,22.2,80,3]).reshape(1,-1))


