# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:57:13 2018

@author: heman
"""



                
user_input=input("enter the number")
'''i=0

while (i<=user_input-1):
    j=0
    while (j<=i):
        print ("*"),
        j=j+1;
    print "\n"
    
    i=i+1;


n=0
while (user_input>=n):
    b=0
    
    while (b<user_input-1):
        print ("*"),
        b=b+1;
    print "\n"
    
    user_input-=1; ''''

for i in range(0,user_input):
    print "* "*i

for i in range(user_input,0,-1):
    print "* "*i
    