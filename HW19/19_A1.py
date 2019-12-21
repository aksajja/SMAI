import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

######## Data Generation ############
np.random.seed(0)
x,y = np.random.multivariate_normal([0,0],[[1,0],[0,1]],1000).T
data1 = np.append(x.reshape(1000,1),y.reshape(1000,1),axis=1)
labels1 = (np.ones(1000))*-1
x,y = np.random.multivariate_normal([0.5,0.5],[[1,-1],[-1,2]],1000).T
data2 = np.append(x.reshape(1000,1),y.reshape(1000,1),axis=1)
labels2 = np.ones(1000)
data = np.append(data1,data2, axis=0)
labels = np.append(labels1, labels2, axis=0)
labels = labels.reshape(2000,1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
plt.plot(data1[:,:1],data1[:,1:],'x')
plt.plot(data2[:,:1],data2[:,1:],'x')
plt.axis('equal'); plt.show()

############# Converting Data from numpy to tensors

import torch
import torch.nn as nn


X = torch.tensor(X_train, dtype=torch.float) 
y = torch.tensor(y_train, dtype=torch.float) 
xPredicted = torch.tensor(X_test, dtype=torch.float)

############# Converting Data from numpy to tensors

import torch
import torch.nn as nn


X = torch.tensor(X_train, dtype=torch.float) 
y = torch.tensor(y_train, dtype=torch.float) 
xPredicted = torch.tensor(X_test, dtype=torch.float)

## Model with weights initialized as 0.0
def init_weights_zero(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.01)

model.apply(init_weights_zero)

print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 100
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
    print("Epoch {} - Training loss: {}".format(e, running_loss/X_train.shape[0]))
    learning_rate.append(running_loss/X_train.shape[0])
    
epochs = list(range(epochs))
plt.plot(epochs, learning_rate)
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

## Model with weights initialized as 1.0
def init_weights_one(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.01)

model.apply(init_weights_one)

print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 100
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
    
epochs = list(range(epochs))
plt.plot(epochs, learning_rate)
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