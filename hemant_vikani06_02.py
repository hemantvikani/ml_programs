# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:48:59 2018

@author: heman
"""
str1=raw_input("enter the string")
def reverse(str1):
    
        i=0
        length=len(str1)
        while (length>i):
            print str1[length-1],
            length-=1;
        
        
reverse(str1)
