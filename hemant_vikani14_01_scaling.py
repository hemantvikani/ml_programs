# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:05:32 2018

@author: heman
"""
#standard scaling
import pandas as pd

df = pd.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
    )
df.columns=['Class label','Alcohol','Malic acid']
features=df.iloc[:,1:]
labels=df.iloc[:,0]
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)

#minmaxscaling
import pandas as pd

df = pd.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
    )
df.columns=['Class label','Alcohol','Malic acid']
features=df.iloc[:,1:]
labels=df.iloc[:,0]
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
features_train_minmax = min_max_scaler.fit_transform(features_train)
features_test_minmax = min_max_scaler.fit_transform(features_test)