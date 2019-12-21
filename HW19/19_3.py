#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

######## Data Generation ############
np.random.seed(0)
x,y = np.random.multivariate_normal([0,0],[[1,0],[0,1]],100).T
data1 = np.append(x.reshape(100,1),y.reshape(100,1),axis=1)
labels1 = (np.ones(100))*-1
x,y = np.random.multivariate_normal([0.5,0.5],[[1,-1],[-1,2]],100).T
data2 = np.append(x.reshape(100,1),y.reshape(100,1),axis=1)
labels2 = np.ones(100)
data = np.append(data1,data2, axis=0)
labels = np.append(labels1, labels2, axis=0)
labels = labels.reshape(200,1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
plt.plot(data1[:,:1],data1[:,1:],'x')
plt.plot(data2[:,:1],data2[:,1:],'x')
plt.axis('equal'); plt.show()


# In[2]:


############# Converting Data from numpy to tensors

import torch
import torch.nn as nn


X = torch.tensor(X_train, dtype=torch.float) 
y = torch.tensor(y_train, dtype=torch.float) 
xPredicted = torch.tensor(X_test, dtype=torch.float)


# In[7]:


import matplotlib.pyplot as plt
from torch import nn, optim

input_size = 2
hidden_size = 2
output_size = 1

model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.Tanh(),
                      nn.Linear(hidden_size, output_size),
                      nn.Sigmoid())
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   #weight decay will be used for L2/L1 penality
epochs = 1000
learning_rate = []
for e in range(epochs):
    running_loss = 0
    for i in range(X_train.shape[0]):
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(X[i])
        loss = criterion(output, y[i])
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()    
    learning_rate.append(running_loss/X_train.shape[0])
    
plt.plot(list(range(epochs)), learning_rate)    
plt.show()
    
y_pred = []
with torch.no_grad():
    for i in range(X_test.shape[0]):
        logps = model(xPredicted[i])
        if(logps>0.001):
            y_pred.append(1)
        else:
            y_pred.append(-1)

acc = accuracy_score(y_test, y_pred)
print("Accuracy of model with Pytorch library: ", acc)


# In[15]:


import matplotlib.pyplot as plt
from torch import nn, optim

input_size = 2
hidden_size = 2
output_size = 1

model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.Tanh(),
                      nn.Linear(hidden_size, output_size),
                      nn.Sigmoid())
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   #weight decay will be used for L2/L1 penality
epochs = 10
learning_rate = []
for e in range(epochs):
    running_loss = 0
    for i in range(X_train.shape[0]):
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(X[i])
        loss = criterion(output, y[i])
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        #running_loss += loss.item()
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))
    learning_rate.append(regularization_loss/X_train.shape[0])
    
plt.plot(list(range(epochs)), learning_rate)    
plt.show()
    
y_pred = []
with torch.no_grad():
    for i in range(X_test.shape[0]):
        logps = model(xPredicted[i])
        if(logps>0.5):
            y_pred.append(1)
        else:
            y_pred.append(-1)

acc = accuracy_score(y_test, y_pred)
print("Accuracy of model with Pytorch library: ", acc)

