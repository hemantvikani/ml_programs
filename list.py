# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:44:47 2018

@author: heman
"""

x = [1,2,3]
print x
x.append(5)
print x
x.insert(0,100)
print x
x.pop
print x
x.pop(0)
print x
x=[2,4,5,6]
print x
x.pop()
print x
x.remove(3)
print x
list1=[2345,34,56,678,-34,56,78]
print list1 
 list1.sort()
 print list1
 del list1[-2]
 print list1
 x.append(list1)
 print x
 print x[-1]
 x.extend(list1)
 print x
 print x[-1]
 stack=[2,3,4]
 print stack
 stack.append(6)
 stack.pop()
 print stack
 a='fedora','debian','ubuntu'
 print a
 print a[1]
 t=1234,3456,'abc'
 print t
 print divmod(15,2)
 x,y=divmod(15,2)
 print x
 print y
 x=None
 print x
 print x is None
 a= [4,2,1,3,6]
 for i in a:
     if i<3:
         print i
     else:
         print "no"
print "done"
 
