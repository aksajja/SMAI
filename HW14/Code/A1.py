import sklearn.svm as svm
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def plot_graphs(modelA):
    colormap = np.array(['r', 'k'])
    # Plot the original data
    plt.scatter(X[:4], y[:4], c=colormap[0], s=40)
    plt.scatter(X[4:], y[4:], c=colormap[1], s=40)
    
    ## Calc the hyperplane (decision boundary)
    # ModelA
    ymin, ymax = plt.ylim()
    print(modelA)
    print(modelA.coef_)
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

classA = [7,9,5,3]
classB = [-2,0,-8,1]

X = np.append(classA,classB)
y = [1,1,1,1,-1,-1,-1,-1]
X = X.reshape(-1,1)

linear_model = svm.LinearSVC()
fitted_model = linear_model.fit(X,y)
plot_graphs(fitted_model)