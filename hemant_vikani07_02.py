# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:05:10 2018

@author: heman
"""
import oauth2
import time
import urllib2
import json
url1 = "https://api.twitter.com/1.1/search/tweets.json"  # FIXED AUTHENCATION PARAMETERS

params = {
        "oauth_version": "1.0",
        "oauth_nonce": oauth2.generate_nonce(),
        "oauth_timestamp": int(time.time())
    }

consumer = oauth2.Consumer(key="EibhvPsBnsC8UTIqs6VPqUnAF", secret="c1yGUQZlFeDWZ9LJTB3mtrzLNaLNbep6NXM22k7FNHr8Ygpb1x")

token = oauth2.Token(key="3250551589-f3esRzHMjRhOXTAFLD5khKO7jWMg6NOZn96AaxC",
                     secret="e12yZvPj41D867f2QXKeZKQM4HUS1EViQvjTdDHlX8shy")

params["oauth_consumer_key"] = consumer.key   # VARIABLE AUTHENCATION PARAMETERS

params["oauth_token"] = token.key
params["q"]="jaipur"

req = oauth2.Request(method="GET", url=url1, parameters=params)
signature_method = oauth2.SignatureMethod_HMAC_SHA1() 
req.sign_request(signature_method, consumer, token)
url = req.to_url()
response = urllib2.Request(url)
data = json.load(urllib2.urlopen(response))

filename = params["q"]      
f = open(filename + "_File.txt", "w")  # SAVING DATA TO FILE
json.dump(data["statuses"], f)
f.close()