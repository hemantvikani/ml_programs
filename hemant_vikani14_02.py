# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:53:11 2018

@author: heman
"""

import pandas as pd
data=pd.read_csv("Red_Wine.csv")


for i in data:
    data[i]=data[i].fillna(data[i].mode()[0])
features=data.iloc[:,0:-1].values
labels=data.iloc[:,-1].values





from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()

features[:,0]=labelencoder.fit_transform(features[:,0])

onehotencoder=OneHotEncoder(categorical_features=[0])
features=onehotencoder.fit_transform(features).toarray()
labels=labelencoder.fit_transform(labels)

view_features = pd.DataFrame(features)
view_labels = pd.DataFrame(labels)






from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)
