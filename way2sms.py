#to send sms to a number


import requests
phone = raw_input("Enter receiver's number: ")
msg = raw_input("Enter the message to send: ")
url = " https://smsapi.engineeringtgr.com/send/?Mobile=9529160701&Password=D6927E&Key=hemanuLHhZ7w8rRXlMJ1pS&Message="+msg+"&To="+phone+""

resp = requests.get(url)   # GET request to REST API

print resp.text