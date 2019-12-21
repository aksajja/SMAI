#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import random
# plt.rcParams['figure.figsize'] = (16, 9)
mu=1;
sigma=0.1
# Creating a sample dataset with 4 clusters
X1 = np.random.normal(mu, sigma, (10,2)) 
X2= np.random.normal(mu*0.2, sigma, (10,2)) 


# In[6]:


plt.scatter(X1[:,0],X1[:,1],marker='o',c='r')

plt.scatter(X2[:,0],X2[:,1],marker='*',c='g')
plt.title("Data")
plt.show()


# In[8]:


from sklearn.cluster import KMeans
# Initializing KMeans
X=[X1,X2]
X=np.row_stack(X)
np.random.shuffle(X)
kmeans = KMeans(n_clusters=5)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
print(labels)
# Getting the cluster centers
C = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1],c=labels)
plt.title("After Clustering K=5")
plt.show()


# In[9]:


Error=[]
list_k = list(range(1,10))

for k in list_k:
    km=KMeans(n_clusters=k)
    km.fit(X)
    Error.append(km.inertia_)
#print(Error)    
plt.plot(list_k,Error,'-o')
plt.grid()
plt.xlabel("Number of Clusters k")
plt.ylabel("Sum of squared distance")
plt.title("Elbow Curve")


# In[10]:


from sklearn.metrics import silhouette_score
sil=[]
kmax=10
klist=[]
for k in range(2,kmax+1):
    klist.append(k)
    km=KMeans(n_clusters=k)
    km.fit(X)
    labels1=km.predict(X)
    sil.append(silhouette_score(X,labels1,metric='euclidean'))
print(sil)    
plt.plot(klist,sil,'-o')
plt.grid()
plt.xlabel("Number of Clusters k")
plt.ylabel("Silhoutte score")
plt.title("Silhoutte curve")


# In[ ]:




