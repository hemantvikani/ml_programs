# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:34:38 2018

@author: heman
"""

from PIL import Image
img_filename=Image.open("sample.jpg").convert("L").rotate(270)
w,h = img_filename.size
hw,hh = w/2,h/2
dim = (hw-80,hh-102,hw+80,hh+102)
img_filename = img_filename.crop(dim)
img_filename.thumbnail((75,75))
img_filename.show()
img_filename.save("my_img.jpg")