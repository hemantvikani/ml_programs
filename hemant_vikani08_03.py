# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:43:23 2018

@author: heman
"""



user_input=raw_input("enter item,price").split(" ")
 

    
flag=0    
   
for i in user_input:
    if (int(i)>0):
        if (i[0:-1]==i[-1:0]):
            flag=1
            
    else:
        flag=0
        break
        
if flag==1:
    print True
else:
    print False

    
    
    