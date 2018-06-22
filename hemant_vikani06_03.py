# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:02:56 2018

@author: heman
"""
import requests
#import json
def send():
    
    d = {}
    for i in ["Phone_Number","Name","Branch","College_Name"]:
        d[i] = raw_input("Enter "+i+": ")
      
  
    r = requests.post("http://13.127.155.43/api_v0.1/sending", json=d)
    print r.text
    
send()

def recieve():
    recieve_url="http://13.127.155.43/api_v0.1/receiving"
    recieve_url+="?Phone_Number=123"
    recieve_req=requests.get(recieve_url)
    print recieve_req.text
recieve()
