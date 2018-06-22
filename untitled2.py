# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:52:48 2018

@author: heman
"""


import sqlalchemy
from pandas import DataFrame

# connect to  MySQL server along with Database name
# as mysql://username:password@host:port/database
engine = sqlalchemy.create_engine('mysql://root:@localhost/data science')
query = "select * from student" #query Database
resoverall = engine.execute(query) #execute Query
df = DataFrame(resoverall.fetchall()) #putting the result into Dataframe
df.columns = resoverall.keys() #Setting the Column names as it was in database.