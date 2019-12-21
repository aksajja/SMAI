#!/usr/bin/env python
# coding: utf-8

# ## Simple Model -- Undercomplete AE

# In[1]:


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


#parameters
batch_size = 128

preprocess = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

#Loading the train set file
dataset = datasets.MNIST(root='./data',
                            transform=preprocess,  
                            download=True)

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# In[31]:


hidden_sizes=[8, 10, 12, 16]
input_size = 28*28
out_sizes=[16,32,64,128]
num_classes = 2
num_epochs = 50
learning_rate = 0.01
momentum_rate = 0.9


# ### Autoencoder Class

# In[32]:


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )
    
    def forward(self,x):
        h = self.encoder(x)
        xr = self.decoder(h)
        return xr,h

#Misc functions
def loss_plot(losses):
    max_epochs = len(losses)
    times = list(range(1, max_epochs+1))
    plt.figure(figsize=(30, 7))
    plt.xlabel("epochs")
    plt.ylabel("cross-entropy loss")
    return plt.plot(times, losses)


# In[33]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using CUDA ', use_cuda)

hidden_size =256
net = AE()
net = net.to(device)

#Mean square loss function
criterion = nn.MSELoss()

#Parameters
learning_rate = 1e-2
weight_decay = 1e-5

#Optimizer and Scheduler
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.001, patience=5, verbose = True)

##Adam
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
##Adagrad
# torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)


# In[34]:




#Training
TotalLoss = {}
for index , hs in enumerate(hidden_sizes):
    total_loss, cntr = 0, 0
    epochLoss = []
    for epoch in range(num_epochs):
        for i,(images,_) in enumerate(loader):

            images = images.view(-1, 28*28)
            images = images.to(device)

            # Initialize gradients to 0
            optimizer.zero_grad()

            # Forward pass (this calls the "forward" function within Net)
            hidden_size = hs
            outputs, _ = net(images)

            # Find the loss
            loss = criterion(outputs, images)

            # Find the gradients of all weights using the loss
            loss.backward()

            # Update the weights using the optimizer and scheduler
            optimizer.step()

            total_loss += loss.item()
            cntr += 1

        scheduler.step(total_loss/cntr)
        print ('Epoch [%d/%d], Loss: %.4f' 
                       %(epoch+1, num_epochs, total_loss/cntr))
        epochLoss.append(total_loss/cntr)
    TotalLoss[hs] = epochLoss
    


# In[41]:


#a= loss_plot(epochLoss)
epochs = list(range(1,51))
for item in hidden_sizes:
    plt.plot(epochs, TotalLoss[item],  label="hidden_units: "+str(item))
plt.title('RMS Prop', fontsize=20)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('loss', fontsize=16)
plt.legend(loc='lower right')
plt.show()


# In[34]:


len(dataset)


# ### Reconstruction

# In[42]:


#Feature Extraction
ndata = len(dataset)
hSize = 2

test_dataset = datasets.MNIST(root='./data',
                            transform=preprocess,  
                            download=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

iMat = torch.zeros((ndata,28*28))
rMat = torch.zeros((ndata,28*28))
featMat = torch.zeros((ndata,hSize))
labelMat = torch.zeros((ndata))
cntr=0

with torch.no_grad():
    for i,(images,labels) in enumerate(loader):

        images = images.view(-1, 28*28)
        images = images.to(device)
        
        rImg, hFeats = net(images)
        
        iMat[cntr:cntr+batch_size,:] = images
        rMat[cntr:cntr+batch_size,:] = (rImg+0.1307)*0.3081
        
        featMat[cntr:cntr+batch_size,:] = hFeats
        labelMat[cntr:cntr+batch_size] = labels
        
        cntr+=batch_size
        
        if cntr>=ndata:
            break


# In[43]:


#Reconstruction
plt.figure()
plt.axis('off')
plt.imshow(rMat[1,:].view(28,28),cmap='gray')


# In[44]:


print(iMat.shape , rMat.shape)
err = torch.sqrt(torch.mean(iMat - rMat)**2)
print(err)
#print("Error for",np.sqrt(torch.mean(torch.stack(iMat - rMat)**2)))


# # 
