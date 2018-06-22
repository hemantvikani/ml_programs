# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:20:32 2018

@author: heman
"""



 #Description:Car License plate detection of India
 #Author: Forsk Technologies
 #Version:1.1
 #Revision Date:27/08/2016


from havenondemand.hodclient import *
import json
import pandas as pd

client = HODClient("9f31a3e7-25e7-4bc6-8c22-9963faceba5f", "v1") #apikey

params = {'url': 'https://www.havenondemand.com/sample-content/videos/gb-plates.mp4', 'source_location': 'GB'} ##if using url
#params = {'file': 'E:/abcd.mp4', 'source_location': 'GB'} ## or if using a local file
response_async = client.post_request(params, 'recognizelicenseplates', async=True)
jobID = response_async['jobID']
#print jobID

## Wait some time afterwards for async call to process...
response = client.get_job_result(jobID)
print response

#dump data in a json file
with open('data.json', 'w') as outfile:
    json.dump(response, outfile)