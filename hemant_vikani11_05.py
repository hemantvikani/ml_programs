# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:02:08 2018

@author: heman
"""

import numpy as np

import pandas as pd

df=pd.read_csv("Automobile.csv")

df['price'] = df['price'].fillna(df['price'].mean())
x=np.array(df["price"])

print x.min()
print x.max()
print x.mean()
print x.std()
