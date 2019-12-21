import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import sklearn.svm as svm

def plot_graphs(modelA,X,y):
    colormap = np.array(['r', 'k'])
    print(X.shape,y.shape)
    # Plot the original data
    plt.scatter(X[:600,0], X[:600,1], c='red', s=20)
    print(X[:50])
    # plt.scatter(X[600:,0], X[600:,1], c='blue', s=20)
    
    ## Calc the hyperplane (decision boundary)
    # ModelA
    ymin, ymax = plt.ylim()
    print(modelA)
    w = modelA.coef_[0]
    a_modelA = -w[0]
    xx = np.linspace(ymin, ymax)
    yy_modelA = a_modelA * xx - (modelA.intercept_[0])
    
    # Plot the line
    plt.plot(yy_modelA,xx, 'k-')
    print_margin(a_modelA, ymin, ymax, modelA)
    plt.show()
    

def print_margin(a_modelA, ymin, ymax, modelA):
    w_modelA = modelA.coef_[0]
    p1 = np.asarray([a_modelA*ymin-(modelA.intercept_[0]) ,ymin])
    p2 = np.asarray([a_modelA*ymax-(modelA.intercept_[0]) ,ymax])
    p3 = np.asarray([1,1])
    d1 = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    p3 = [-1,-1]
    d2 = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    modelA_name = str(modelA).split('(')[0]
    print("Margin from ", modelA_name, "line is : ",d1, d2)


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

train_data, train_labels = read_data("../../HW13/Code/sample_train.csv")
test_data, test_labels = read_data("../../HW13/Code/sample_test.csv")


X = np.append(train_data[np.nonzero(train_labels == 1)], train_data[np.nonzero(train_labels == 2)], axis=0)
X = np.asarray(X)
y = np.append(np.full((600,),1), np.full((600,),2))
y = y.reshape(-1,1)
c_range = [0.01, 0.1,1,10,100]

temp_data = X
mean = np.mean(temp_data, axis=0)
mean = mean.reshape(1,784)
temp_data = temp_data - mean
cov_mat = ((temp_data.T).dot(temp_data))
eigenvectors, eigenvalues, V = np.linalg.svd(cov_mat, full_matrices=False)
cov_mat_rank = np.linalg.matrix_rank(cov_mat)
print("Rank of covariance matrix : ", cov_mat_rank)
iter_X = X.dot(eigenvectors[:2].T)

for c in c_range:

    linear_model = svm.SVC(kernel='linear',C=c)
    fitted_model = linear_model.fit(iter_X,y)
    plot_graphs(fitted_model,iter_X,y)

plt.show()