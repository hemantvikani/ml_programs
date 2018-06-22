# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:34:10 2018

@author: heman
"""

import facebook as fb


# Facebook Graphic Explorer API user Access Token
access_token = "EAACEdEose0cBANzsJyOMhsSZCIme2BbbA5O1fG9Y7Q5PH09qD1ZA3OzYpq4kncYwaS2Wr61QxlaaKXEBJZAi7kRi0NJy0YE5Y54q4I0M42lXAdkgyRJmhXRcVpBpMFNPoqEdqZAmaWOHZBS9lo8Es2PhenZBZBZA7laNk7IVi8v6Bz0myXA4hmG7GS8EtERSShPinHoaUxbctwZDZD"

# Message to post as status on Facebook
#status = "<Your status message>"

# Authenticating
#graph = fb.GraphAPI(access_token)


# Posting Status on your wall
#post_id = graph.put_wall_post(  )



graph = fb.GraphAPI(access_token)
photo = open("IMG_20170730_171413.jpg","rb")
graph.put_photo(message="hello",image=photo.read())
photo.close()