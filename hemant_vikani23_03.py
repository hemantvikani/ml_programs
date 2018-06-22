

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 23:03:14 2018

@author: mangalam
"""

import pandas as pd
import numpy as np

df=pd.read_csv("tree_addhealth.csv")
for i in df:
    df[i]= df[i].fillna(df[i].mode()[0])
labels=df.iloc[:,7]
df=df.drop("TREG1",axis=1)
features=df.iloc[:,0:15]

from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0) 
classifier.fit(features_train,labels_train)
fit1=classifier.predict(features_test)
Score=classifier.score(features_test,labels_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,fit1)


labels1=df.iloc[:,-4]
features1=df.iloc[:,[0,15]]
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0) 
classifier.fit(features1,labels1)
fit2=classifier.predict(features1)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels1,fit2)
Score1=classifier.score(features1,labels1)
