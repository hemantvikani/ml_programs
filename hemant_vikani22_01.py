# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:17:17 2018

@author: heman
"""
import pandas as pd

data=pd.read_csv('election_data.csv')
transactions=[]
features=data.iloc[:,[4,6]]
labels=data.iloc[:,12]




    
    
#pie chart
values=[]
colors=[]
explode=[]
labels=[]
plt.pie()
plt.title()
plt.show()


