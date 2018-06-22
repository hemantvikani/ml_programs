# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:57:16 2018

@author: heman
"""
import re
n=input("enter no. of inputs")
i=0
while i<n:
    
    usr_inp=raw_input("enter email address" )
    response=re.compile(r'[a-z0-9-_]+@[a-z0-9]+\.[a-z]{2,4}$')
    #response=re.compile(r'[\w.-]+@[\w.-]+.com')
    response1=response.match(usr_inp)
    if response1:
        print usr_inp
    else:
        print "wrong"
    i+=1;
        
   
      
