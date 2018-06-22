# -*- coding: utf-8 -*-
"""
Created on Sun May 13 21:22:58 2018

@author: heman
"""

list1=["he","is","a","good","boy"]
for x in list1:
    print x
    
    
list2=[1,2,3,4,5,6,7,8,9,10]
for x in list2[::2]:
    print x
    str1="A string is a sequence of zero or more characters"
    str1.split(' ',5)
    words=str1.split()
    ','.join(words)
    
    print str1[3:-1]
    _2="harry potter"
    print _2
    os="mac os"
    ver ="6s"
    mob="iphone"
print 'I have an {2}  uses {1}  and model {0}'.format(ver,mob,os)