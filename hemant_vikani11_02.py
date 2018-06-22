# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:34:07 2018

@author: heman
"""


import requests

data={
      "phone number":"904859748",
      "name":"hemant",
      "brannch":"IT"}
send_url="https://api.mlab.com/api/1/databases/country/collections/student?apiKey=9rlXOS2waMlGWqqr91uGZVyUCfXTjFHY"
send_req=requests.post(send_url,json=data)
print send_req.text