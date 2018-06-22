# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:22:00 2018

@author: heman
"""

import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('deliveryfleet.csv')
features=dataset.iloc[:,1:3].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters =i,init = 'k-means++',random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
    
    
kmeans =KMeans(n_clusters = 2 ,init = 'k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(features)

#visualising the clusters
plt.scatter(features[y_kmeans == 0,0],features[y_kmeans==0,1], s=100,c= 'red',label='Cluster 1')
plt.scatter(features[y_kmeans == 1,0],features[y_kmeans==1,1], s=100,c= 'blue',label='Cluster 2')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300,c= 'yellow',label='Centroids')
plt.title('Cluster of drivers')
plt.xlabel('distance')
plt.ylabel('speed')
plt.legend()
plt.show()


#4 clusters
    
kmeans =KMeans(n_clusters = 4 ,init = 'k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(features)

#visualising the clusters
plt.scatter(features[y_kmeans == 0,0],features[y_kmeans==0,1], s=100,c= 'red',label='c1')
plt.scatter(features[y_kmeans == 1,0],features[y_kmeans==1,1], s=100,c= 'blue',label='c2')
plt.scatter(features[y_kmeans == 2,0],features[y_kmeans==2,1], s=100,c= 'black',label='c3')
plt.scatter(features[y_kmeans == 3,0],features[y_kmeans==3,1], s=100,c= 'green',label='c4')


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300,c= 'yellow',label='Centroids')
plt.title('Cluster of drivers')
plt.xlabel('distance')
plt.ylabel('speed')
plt.legend()
plt.show()