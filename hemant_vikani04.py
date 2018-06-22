# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:33:28 2018

@author: heman
"""

s=raw_input("enter the string: ").strip()

index=s.find(" ")

print s[index+1:].strip(),s[:index].strip()

