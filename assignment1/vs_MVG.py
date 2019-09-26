from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

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

# Print accuracy on the test set using MLE
indices_by_label = []
print(train_data[np.nonzero(train_labels == 0)].shape)
indices_by_label.append(train_data[np.nonzero(train_labels == 0)])
indices_by_label.append(train_data[np.nonzero(train_labels == 1)])
# indices_2 = train_data[np.nonzero(train_labels == 2)]
# indices_3 = train_data[np.nonzero(train_labels == 3)]
# indices_4 = train_data[np.nonzero(train_labels == 4)]
# indices_5 = train_data[np.nonzero(train_labels == 5)]
# indices_6 = train_data[np.nonzero(train_labels == 6)]
# indices_7 = train_data[np.nonzero(train_labels == 7)]
# indices_8 = train_data[np.nonzero(train_labels == 8)]
# indices_9 = train_data[np.nonzero(train_labels == 9)]

number_of_labels = 2
paritioned_indices = np.empty([0,0,784])
for i in range(number_of_labels):
    paritioned_indices = np.append(paritioned_indices, indices_by_label[i], axis=0)

print(paritioned_indices.shape)
# for row in indices_1:
#    print(np.where(np.all(train_data==row,axis=1)))

# def pdf_singular_case(x,m,V,k,D):
#     #  ((2π)k|D|)−1/2exp(−Q(x)/2)
#     Qx = (x-m)*np.linalg.inv(V)*(x-m)
#     pd = pow(pow((2*np.pi),k)*np.linalg.det(D),-(1/2))* pow(np.e,-Qx/2)
    
#     return pd

# temp_data[i] = train_data
# mean = np.mean(temp_data[i], axis=0)
# mean = mean.reshape(1,784)
# temp_data[i] = temp_data[i] - mean
# cov_mat = ((temp_data[i].T).dot(temp_data[i]))

# eigenvectors, eigenvalues, V = np.linalg.svd(cov_mat, full_matrices=False)
# cov_mat_rank = np.linalg.matrix_rank(cov_mat)
# print("Rank of covariance matrix : ", cov_mat_rank)

# zero_vec = np.zeros([1,784])
# count = 0
# full_rank_cov_mat = np.empty([0,784])
# non_zero_eigenvalues = np.empty([0])
# # Reducing the p eigenvectors of dim p, to k eigen.Vecs of dim p.
# for i in range(784):
#     # We take 1 as eigen-values very close to 0 need to be ignored. The rank is also satisfied when we take this condition.
#     if eigenvalues[i]>1:
#         non_zero_eigenvalues = np.append(non_zero_eigenvalues,eigenvalues[i])
#         full_rank_cov_mat = np.append(full_rank_cov_mat,eigenvectors[i].reshape(1,784),axis=0)

# # print(non_zero_eigenvalues.shape)
# print("Number of eigenvectors : ", full_rank_cov_mat.shape[0], " ; Dimension: ", full_rank_cov_mat.shape[1])
# mean = mean.reshape(784,)

# # V = E (D) E.T
# D = np.diag(non_zero_eigenvalues)
# V = full_rank_cov_mat.T.dot(D.dot(full_rank_cov_mat))

# for _test_row in test_data:
#     pd = pdf_singular_case(_test_row,mean,V,cov_mat_rank,D)

# # Fitting Multivariate Gaussian
# # from scipy.stats import multivariate_normal
# # fitted_curve = multivariate_normal.pdf(train_data, mean=mean, cov=cov_mat)
