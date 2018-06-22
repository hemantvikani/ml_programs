# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:52:51 2018

@author: heman
"""
import re

while True:
    usr_inp=raw_input("enter the string")
    if not usr_inp:
        break
    else:
        regex=re.compile(r"[+-]?\d*\.\d+$")
    response=regex.match(usr_inp)
        
    if response:
        print "true"
    else:
        print "false"
        
    
