'''
    10C2 LR classifiers from MNIST.
'''

import numpy as np
import sklearn.linear_model as LM

# Build LR classifier for given pair of classes
def pairwise_LR(X, y):
    lr = LM.LogisticRegression(solver='liblinear').fit(X,y)
    return lr

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

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")


lr_map = {}
# 10C2
for i in range(9):
    for j in range(i+1,10):
        indices_by_label = []
        indices_by_label = np.append(train_data[np.nonzero(train_labels == i)], train_data[np.nonzero(train_labels == j)], axis=0)
        indices_by_label = np.asarray(indices_by_label)
        y = np.append(np.full((600,),i), np.full((600,),j))
        lr_map[(i,j)] = pairwise_LR(indices_by_label, y)

lr_pred = []
for key, iter_lr in lr_map.items():
    lr_pred.append(iter_lr.predict(test_data))

lr_pred = np.asarray(lr_pred)
final_pred = []
for i in range(1000):
    unique_elements, counts_elements = np.unique(lr_pred[:,i], return_counts=True)
    final_pred.append(unique_elements[np.nonzero(counts_elements == np.max(counts_elements))][0])
final_pred = np.asarray(final_pred)

confusion_mat = []
for i in range(10):
    confusion_mat.append([])
    actual_labels = np.nonzero(test_labels == i)[0]
    for j in range(10):
        count = 0
        for index in actual_labels:
            if final_pred[index]==j:
                count+=1
        confusion_mat[i].append(count)

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, final_pred) 

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()