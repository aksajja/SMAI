#!/usr/bin/env python
# coding: utf-8

# In[77]:


# Code here
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init as weight_init
import matplotlib.pyplot as plt
import pdb
import numpy as np


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('../MNIST', download=True, train=True, transform=transform)
testset = datasets.MNIST('../MNIST', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');


# input_size = 784
# hidden_sizes = 
# output_size = 10
# 
# encoder = nn.Sequential(
#             nn.Linear(28*28, 500),
#             nn.ReLU(),
#             nn.Linear(500, 10),
#         )
# 
# model = nn.Sequential(nn.Linear(10, hidden_sizes[0]),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[1], output_size),
#                       nn.LogSoftmax(dim=1))
# 
# print(encoder, model)

# In[84]:


input_size = 784
hidden_sizes = [500, 500]
output_size = 30

encoder = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(),
            nn.Linear(500, 30),
        )

model = nn.Sequential(nn.Linear(30, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print(encoder, model)


# In[85]:


from torch import nn, optim

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = encoder(images)
        output = model(output)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))


# In[86]:


c=0
with torch.no_grad():
    for img_test, labels in testloader:
        # Flatten MNIST images into a 784 long vector
        img_test = img_test.view(img_test.shape[0], -1)
        en_output = encoder(img_test)
        logps = model(en_output)
        ps = torch.exp(logps)
        probab = list(ps.numpy())
        pred_output = np.where(probab[0]==max(probab[0]))[0][0]
        if(pred_output == labels):
            c+=1

acc = (c/10000)*100
print("Accuracy of model is: "+str(acc)+"%")


# In[87]:


model_1 = nn.Sequential(nn.Linear(784, 1000),
                      nn.ReLU(),
                      nn.Linear(1000, 1000),
                      nn.ReLU(),
                      nn.Linear(1000, 10),
                      nn.LogSoftmax(dim=1))

print( model_1)


# In[90]:


from torch import nn, optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(model_1.parameters(), lr=0.003, momentum=0.9)
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model_1(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))


# In[92]:


c=0
with torch.no_grad():
    for img_test, labels in testloader:
        # Flatten MNIST images into a 784 long vector
        img_test = img_test.view(img_test.shape[0], -1)
        logps = model_1(img_test)
        ps = torch.exp(logps)
        probab = list(ps.numpy())
        pred_output = np.where(probab[0]==max(probab[0]))[0][0]
        if(pred_output == labels):
            c+=1
            
acc_1 = (c/10000)*100    
print("Accuracy of model is: "+str((c/10000)*100)+"%")


# In[93]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

objects = ('Scenario-1', 'Scenario-2')
y_pos = np.arange(len(objects))
performance = [acc, acc_1]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Encoder+MLP and Normal MLP')

plt.show()

