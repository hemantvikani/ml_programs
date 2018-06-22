# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:30:45 2018

@author: heman
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:42:12 2018

@author: Karan
"""

import urllib2

city=["gurgaon","sonipat","noida"]

import pandas as pd
df=pd.DataFrame()
df1=pd.DataFrame()

for i in city:
    wiki="https://www.wunderground.com/hourly/in/"+i
    page=urllib2.urlopen(wiki)
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(page)
    all_tables=soup.find_all("table")
    right_table=soup.find('table')
    #generate list
    A=[]
    B=[]
    C=[]
    D=[]
    E=[]
    F=[]
    G=[]
    H=[]
    I=[]
    J=[]
    K=[]


    
    for row in right_table.find_all("tr"):
        cells=row.find_all("td")
       # states=row.find_all("th")
        if cells!=[]:
            A.append(cells[0].text.strip())
            B.append(cells[1].text.strip())
            C.append(cells[2].text.strip())
            D.append(cells[3].text.strip())
            E.append(cells[4].text.strip())
            F.append(cells[5].text.strip())
            G.append(cells[6].text.strip())
            H.append(cells[7].text.strip())
            I.append(cells[8].text.strip())
            J.append(cells[9].text.strip())   
            K.append(cells[10].text.strip()) 
    df['time of '+i]=A
    df['condition of '+i]=B
    df['temp. of '+i]=C
    df['feels like of '+i]=D
    df['precip of '+i]=E
    df['amount of '+i]=F
    df['cloud cover of '+i]=G
    df['dew point of '+i]=H
    df['humidity of '+i]=I
    df['wind of '+i]=J
    df['pressure of '+i]=K
    
    


#for delhi
    wiki="https://www.wunderground.com/hourly/in/delhi"
    page=urllib2.urlopen(wiki)
   
    soup=BeautifulSoup(page)
    all_tables=soup.find_all("table")
    right_table=soup.find('table')
    #generate list
    A=[]
    B=[]
    C=[]
    D=[]
    E=[]
    F=[]
    G=[]
    H=[]
    I=[]
    J=[]
    K=[]


    
    for row in right_table.find_all("tr"):
        cells=row.find_all("td")
       # states=row.find_all("th")
        if cells!=[]:
            A.append(cells[0].text.strip())
            B.append(cells[1].text.strip())
            C.append(cells[2].text.strip())
            D.append(cells[3].text.strip())
            E.append(cells[4].text.strip())
            F.append(cells[5].text.strip())
            G.append(cells[6].text.strip())
            H.append(cells[7].text.strip())
            I.append(cells[8].text.strip())
            J.append(cells[9].text.strip())   
            K.append(cells[10].text.strip()) 
    df1['time of delhi']=A
    df1['condition of delhi']=B
    df1['temp. of delhi']=C
    df1['feels like of delhi']=D
    df1['precip of delhi']=E
    df1['amount of delhi']=F
    df1['cloud cover of delhi']=G
    df1['dew point of delhi']=H
    df1['humidity of delhi ']=I
    df1['wind of delhi']=J
    df1['pressure of delhi']=K
    







#predictions

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(0, df.shape[1]):
    df.iloc[:, i] = le.fit_transform(df.iloc[:, i])
    
for i in range(0, df1.shape[1]):
    df1.iloc[:, i] = le.fit_transform(df1.iloc[:, i])

features=df.iloc[:,0:33]
labels=df1.iloc[:,0:11]


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# create the polynomial features using above class
poln_object = PolynomialFeatures(degree = 4)
features_poln = poln_object.fit_transform(features)


#once you have the poln_matrix read, input it to linear regressor
lin_reg_2 = LinearRegression()
lin_reg_2.fit(features_poln, labels)

import numpy as np
print ("Predicting result with Polynomial Regression",)
#need to convert 6.5 into polynomial feature
score2=lin_reg_2.score(features_poln,labels)
Pred1 = lin_reg_2.predict(np.array([12,1,8,1,0,0,2,3,7,1,4,12,0,8,2,0,0,9,3,6,3,4,12,0,8,1,0,0,1,3,7,2,4]).reshape(1,-1))





