# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:30:14 2018

@author: heman
"""

str1=raw_input("enter the string")



def translate(str1):
    str2=""
    
    for i in str1:
        if i not in ["a","e","i","o","u"," "]:
            str2=str2+ i+"o"+i
        else:
            str2=str2+i

    print str2
translate(str1)
