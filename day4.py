# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:30:10 2018

@author: heman
"""

a=[1,2]
b=[1,2]

a is b
a==b

c=a
a is c

def add_two_numbers(a,b):
    return a+b
add_two_numbers(2,4)

def add(a,b):
    print a+b

add(2,3)
print add(2,3)

def explain_python():
    """print a meassage explaining what python is"""
    print('python is not a snake!')
    print('python is a programming language')

explain_python.func_doc

def test(a,b=-99):
    if a>b:
        return a
    else:
        return b

test(13,23)
test(13)


#filter,map,reduce


range(0,10)
def f(x) : return x%3 ==0 or x%5==0
f(3)

filter(f,range(2,25))

def cube(x) : return x*x*x

map(cube,range(1,11))


def add(x,y) : return x+y
reduce(add,range(1,11))

map(lambda x: x*x*x,range(1,11))

a=[1,2,3]
[x**2 for x in a]

[x+1 for x in[x**2 for x in a]]

r=['It','is','raining','in','and','dogs']
map(lambda r:len(r),r)


