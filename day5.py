# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:37:10 2018

@author: heman
"""

fp = open("test.txt")
type(fp)

fp.read()
print fp.read()

fp.close()
fp=open("test.txt")
fp.readLine()

fp.seek(0,0)

fp.readLine()

fp.readLines()

fp.seek(0,0)

for line in fp:
    print line
    
fp.close()

fp=open("test.txt","w")
fp.write("Machine Learning class")
fp.close()


import zlib

s="python is used extensively in industry.Bsdu"

len(s)

t=zlib.compress(s)

zlib.decompress(t)


import urllib2

f=urllib2.urlopen("http://www.forsk.in/")

f.read(1000)

import os
os.getcwd() 


from PIL import ImageFilter

img_filename=Image.open("sample1.jpg")
img_filename.show()

from PIL import ImageFilter

img_filename.filter (ImageFilter.BLUR).show()

img_filename.mode()   
