#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


# In[19]:


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


# In[37]:


input_size = 784
hidden_sizes = [850, 1000]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print(model)


# In[38]:


learning_rate = [1, 0.1, 0.01, 0.001, 0.0001]
criterion = nn.NLLLoss()
for item in learning_rate:
    cal_loss = []
    epoch = list(range(1,11))
    print(item)
    optimizer = optim.SGD(model.parameters(), lr=item, momentum=0.9)
    epochs = 10
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            #This is where the model learns by backpropagating
            loss.backward()

            #And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        cal_loss.append(running_loss/len(trainloader))
    plt.plot(epoch, cal_loss)
plt.show()

