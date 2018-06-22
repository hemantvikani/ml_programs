# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 11:00:30 2018

@author: heman
"""

import numpy as np
import pandas as pd


#importing the dataset

dataset=pd.read_csv("mushrooms.csv")
features=dataset.iloc[:,[5,21,22]].values
labels=dataset.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
features[:,0]=labelencoder.fit_transform(features[:,0])
features[:,1]=labelencoder.fit_transform(features[:,1])
features[:,2]=labelencoder.fit_transform(features[:,2])


onehotencoder=OneHotEncoder(categorical_features=[0])
features=onehotencoder.fit_transform(features).toarray()

onehotencoder=OneHotEncoder(categorical_features=[-2])
features=onehotencoder.fit_transform(features).toarray()

onehotencoder=OneHotEncoder(categorical_features=[-1])
features=onehotencoder.fit_transform(features).toarray()

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

#fitting K-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(features_train,labels_train)


labels_pred=classifier.predict(features_test)
score=classifier.score(features_test,labels_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_pred)
score1=classifier.score(features_test,labels_test)

