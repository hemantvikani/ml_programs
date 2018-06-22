# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:13:10 2018

@author: heman
"""

user_input=raw_input("enter the number:")
digit=0
letter=0
for i in user_input:
    if i.isdigit():
        digit+=1
    elif i.isalpha():
        letter+=1
print digit
print letter
 