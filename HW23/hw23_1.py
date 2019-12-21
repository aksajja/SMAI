#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import random
Data=[[0,-1],[0,1],[2,-1],[2,1],[1,-1],[1,1],[1,-0.5],[1,0.5],[0.5,0],[1.5,0]]
mue1=[[0.5,0]]
mue2=[[1.5,0]]


# In[2]:


import matplotlib.pyplot as plt
Data1=np.asarray(Data)
print(Data1.shape)

plt.scatter(Data1[0:4:,0] ,Data1[0:4:,1] ,c='k')
plt.scatter(Data1[4:6,0] ,Data1[4:6:,1] ,c='b')
plt.scatter(Data1[6:8:,0] ,Data1[6:8:,1] ,c='g')
plt.scatter(Data1[8:10:,0] ,Data1[8:10:,1] ,c='r',marker='x')
plt.xlim([-1,3])
plt.show()


# In[3]:


K1=[]
K2=[]
for it in range(5):
    for i in range(np.size(Data,0)):
        #print(Data[i],mue1)
        Ed1=euclidean_distances(mue1, [Data[i]])
        Ed2=euclidean_distances(mue2, [Data[i]]) 
        if Ed1<Ed2:
            K1.append(Data[i])
        elif Ed1>Ed2:
            K2.append(Data[i])
        else:
            if(it%2==1):
                K1.append(Data[i])
            elif(it%2==0):
                K2.append(Data[i])
    mue1=[np.mean(K1,axis=0)]
    mue2=[np.mean(K2,axis=0)]
    print ("At iteration",it)
    print(K1)
    print("==========")
    print(K2)
    K1=[]
    K2=[]
    
            
        

    
    
        
    
    
    
    


# In[4]:


from sklearn import datasets , cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
dataset, labels=datasets.make_circles(n_samples=1000
                              ,factor=0.5,noise=0.05)
#print(labels)
plt.scatter(dataset[:,0],dataset[:,1] , c=labels)
plt.show()


# In[5]:


KM= cluster.KMeans(n_clusters=2,max_iter=300,init='k-means++',n_init=10).fit_predict(dataset)
#print(KM)
plt.scatter(dataset[:,0],dataset[:,1],c=KM)
plt.show()


# In[ ]:




