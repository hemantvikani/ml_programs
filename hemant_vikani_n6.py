# -*- coding: utf-8 -*-
"""
Created on Mon May 14 23:55:50 2018

@author: heman
"""
#my approach but not working
mylist=input("enter the array:")
list1=list(mylist)


s=0


x=len(list1)

    
k=0
while k<x:
    if list1[k]==13 or (list1[k-1]==13):
        k+=1
        continue
    
    else:
        s=s+int(list1[k])
    k=k+1;
    
print s
        
        
        
        