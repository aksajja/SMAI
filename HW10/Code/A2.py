import numpy as np
import csv
import matplotlib.pyplot as plt
import math

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 13
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ x for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("wine.data")
dim = train_data.shape[1]
print("Mean : ", np.mean(train_data,axis=0))
# for data_class in range(1,4):
#     each_class_data = train_data[np.nonzero(train_labels == class_label)]
#     cov_mat = np.cov(each_class_data.T)
cov_matrix = np.cov(train_data.T)
print("Cov : ", cov_matrix.shape)
eigenvectors, eigenvalues, _ = np.linalg.svd(cov_matrix)

eigenvalues = eigenvalues.reshape(eigenvalues.shape[0],1)
norm_constant = math.sqrt(np.sum((eigenvalues.T).dot(eigenvalues)))

norm_eigenvalues = [i/norm_constant for i in eigenvalues]
X_axis = np.linspace(1,13,13)
plt.plot(X_axis, eigenvalues, color = 'black')
plt.scatter(X_axis, eigenvalues.reshape(1,dim)[0], color = 'red')
plt.show()

'''
Only one PC should be used as can be seen from the eigen values. The difference in eigen values is the reason. 
    First : 9.92017895e+04 
    Second : 1.72535266e+02 
    Third : 9.43811370e+00 
'''

pca_2d = eigenvectors[:2]

projected_pts = pca_2d.dot(train_data.T)

print(projected_pts[:][0][:59].shape, projected_pts[:][0][59:130].shape, projected_pts[:][0][130:].shape)
# Class 1
plt.scatter(projected_pts[:][0][:59], projected_pts[:][1][:59], label="Class 1")
# Class 2
plt.scatter(projected_pts[:][0][59:130], projected_pts[:][1][59:130],c='red', label="Class 2")
# Class 3
plt.scatter(projected_pts[:][0][130:], projected_pts[:][1][130:],c='green', label="Class 3")
plt.legend(loc='upper right')
plt.show()

'''
Class 1 is distinctly separated.
There is a certain overlap between classes 2 & 3.
'''