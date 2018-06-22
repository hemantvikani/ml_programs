# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:09:11 2018

@author: heman
"""
dict1={}
for i in ["a","b","c"]:
    dict1[i]=input()
    
s=0
for i in dict1.values():
    if i in [13,14,17,18,19]:
        s=s+0
    else:
        s=s+i
print s
            