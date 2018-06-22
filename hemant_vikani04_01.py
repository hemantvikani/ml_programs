# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:06:21 2018

@author: heman
"""
def add(list1):
    return sum(list1)
def mul(list1):
    return reduce(lambda x,y : x*y,list1)
def largest(list1):
    return max(list1)
def smallest(list1):
    return min(list1)
def sort(list1):
    list1.sort()
    return list1
def remove_duplicates(list1):
    new_list=list(set(list1))
    return new_list
    

def print1():
    list1=list(input("enter the list"))
    print "sum=",add(list1)
    print "multiply=",mul(list1)
    print "largest=",largest(list1)
    print "smallest=",smallest(list1)
    print "sorted=",sort(list1)
    print "removing duplicates=",remove_duplicates(list1)
    

print1()
