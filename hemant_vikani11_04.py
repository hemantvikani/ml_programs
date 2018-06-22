# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:28:37 2018

@author: heman
"""

import numpy as np
user_input=raw_input().split(" ")
list1=[]
for i in user_input:
    list1.append(int(i))
print list1

x = np.array(list1)
print x.reshape(3,3)
