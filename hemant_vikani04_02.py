# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:01:47 2018

@author: heman
"""


user_input=input("enter the three parameter")
List1=list(user_input)

def brick(List1):
   
    if List1[2]%5>=List1[0]:
        print False
       
    elif (1*List1[0]+5*List1[1]>=List1[2]):
        print True
    
brick(List1)