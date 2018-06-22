# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:19:40 2018

@author: heman
"""

user_input=input("enter the numbers")
list1=list(user_input)


list1.sort()
print list1




list2=list1[1:-1]
print list2

sum1=0
for item in list2:
    sum1+=int(item)
    avg=sum1/len(list2)
print sum1
print avg
    
    
    