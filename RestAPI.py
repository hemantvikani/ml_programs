# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:58:38 2018

@author: heman
"""

str1=raw_input("enter the string")
str2=str1.lower().replace(" ","")
#print str2


list1=list(str2)
list1.sort()
new_list=(set(list1))

        
#print new_list
length=len(new_list)
if length==26:
    print "PANGRAM"
else:
    print "NOT PANGRAM"
