# -*- coding: utf-8 -*-
"""
Created on Wed May 23 23:01:18 2018

@author: heman
"""


from pandas import DataFrame
import mysql.connector

# connect to  MySQL server along with Database name
conn = mysql.connector.connect(user='root', password='',
                              host='localhost',
                              database='job')

# Creating cursor Object from connection object
cursor = conn.cursor()

query = ("SELECT * FROM job_satisfaction;")  # query Database
cursor.execute(query)  # execute Query
df = DataFrame(cursor.fetchall())  # putting the result into Dataframe
df.columns = cursor.column_names # Setting the Column names as it was in database.