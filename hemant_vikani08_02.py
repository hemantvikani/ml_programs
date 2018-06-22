# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:01:16 2018

@author: heman
"""
list1=[]
while True:
    user_input=raw_input("enter item,price")
    if not user_input:
        break
    tup1=user_input.split(" ")
    price=tup1[-1]
    prod=' '.join(tup1[0:-1])
    
    list1.append((prod,int(price)))
    
    

d={}
for i in list1:
     m=list1.count(i)
      
     d[i[0]] = i[1]*m
     

for k,v in d.items():
    print k,v    
  
