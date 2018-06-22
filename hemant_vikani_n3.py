# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:35:47 2018

@author: heman
"""

user_input=raw_input("enter the string:")
List=list(user_input)

dict1 = {}

for key in List:
    #print (key,user_input.count(key))
    dict1[key] = user_input.count(key)
    
print dict1
  



    