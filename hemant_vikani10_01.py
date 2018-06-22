# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:19:36 2018

@author: heman
"""

import urllib2
wiki="https://www.icc-cricket.com/rankings/mens/team-rankings/odi"
page=urllib2.urlopen(wiki)
from bs4 import BeautifulSoup
soup=BeautifulSoup(page)
all_tables=soup.find_all("table")
right_table=soup.find('table',class_="table")
#generate list
A=[]
B=[]
C=[]
D=[]
E=[]


for row in right_table.find_all("tr"):
    cells=row.find_all("td")
   # states=row.find_all("th")
    if len(cells)==5:
        A.append(cells[0].find(text=True))
        B.append(cells[1].text.strip())
        C.append(cells[2].find(text=True))
        D.append(cells[3].find(text=True))
        E.append(cells[4].find(text=True))
       
import pandas as pd
df=pd.DataFrame(A,columns=['pos'])
df['team']=B
df['matches']=C
df['point']=D
df['rating']=E

print df
