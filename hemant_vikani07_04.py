# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:17:19 2018

@author: heman
"""

from twython import Twython
APP_KEY="AZGsj38t94F4IwImkoPFP61OO"
APP_SECRET="CJjVPJ1wPoZVOMAS9tQaIaexyvU8j2PuUv4CyunBKnWkJGQBcM"
OAUTH_TOKEN="3250551589-zMhhfBaP24RiNL37R1WMe7MJc5SpRm7M1pMvPxr"
OAUTH_TOKEN_SECRET="QT5XKVpFNPn4Jb7tpWZ1JfWDZDKv7EyzxwdFJ3W27Q3eX"

twitter = Twython(APP_KEY, APP_SECRET,
                  OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
twitter.update_status(status='See how easy using Twython is!')
