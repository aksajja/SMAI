import numpy as np
import torch


# Load MNIST data
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("../../assignment1/sample_train.csv")
test_data, test_labels = read_data("../../assignment1/sample_test.csv")


dtype = torch.float
device = torch.device("cpu")

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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 200
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
    if(e%10==0):    
        print("Epoch {} - Training loss: {}".format(e, running_loss/X_train.shape[0]))
        
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