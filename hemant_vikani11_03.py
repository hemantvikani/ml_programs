# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:05:25 2018

@author: heman
"""
from collections import Counter
import numpy as np
#import matplotlib.pyplot as plt
x = np.random.random_integers(5,15,40)
print x

#without using numpy
b = Counter(x)
print b
print b.most_common(1)


#with numpy
counts = np.bincount(x)
print counts
print np.argmax(counts)

#another code for without using numpy
d={}
for i in x:
    if i not in d:
        d[i]=1
    else:
        d[i]+=1

maximum = max(d.values())
        
for i,j in d.items():
        if j == maximum:
            print i
            break
    
    
    