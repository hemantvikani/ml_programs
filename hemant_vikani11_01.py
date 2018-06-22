# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:55:52 2018

@author: heman
"""

import pandas as pd
df=pd.read_csv("training_titanic.csv")
print df["Survived"].value_counts()
print df["Survived"].value_counts(normalize = True)
df_sex=df.groupby(["Sex"])
print df_sex["Survived"].value_counts()
print df_sex["Survived"].value_counts(normalize = True)

df=df.fillna(df.mean())
df.insert(12,"child","0",allow_duplicates=True)

df["child"][df["Age"]<18] = 1
df


