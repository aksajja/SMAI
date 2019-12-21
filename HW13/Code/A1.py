'''
    10C2 LR classifiers from MNIST.
'''

import numpy as np
from numpy.linalg import norm
import sklearn.linear_model as LM
import matplotlib.pyplot as plt

# Build LR classifier for given pair of classes
def log_reg(X, y):
    lr = LM.LogisticRegression(solver='liblinear').fit(X,y)
    return lr

def perceptron(X, y):
    perc = LM.Perceptron(tol=1e-3, random_state=0, eta0=0.1).fit(X,y)
    return perc

def plot_graphs(modelA, modelB, gamma=False):
    colormap = np.array(['r', 'k'])
    # Plot the original data
    plt.scatter(X[:2,0], X[:2,1], c=colormap[0], s=40)
    plt.scatter(X[2:,0], X[2:,1], c=colormap[1], s=40)
    
    ## Calc the hyperplane (decision boundary)
    # ModelA
    ymin, ymax = plt.ylim()
    w = modelA.coef_[0]
    a_modelA = -w[0] / w[1]
    xx = np.linspace(ymin, ymax)
    yy_modelA = a_modelA * xx - (modelA.intercept_[0]) / w[1]
    
    # ModelB
    ymin, ymax = plt.ylim()
    w = modelB.coef_[0]
    a_modelB = -w[0] / w[1]
    xx = np.linspace(ymin, ymax)
    yy_modelB = a_modelB * xx - (modelB.intercept_[0]) / w[1]
    # Plot the line
    plt.plot(yy_modelA,xx, 'k-')
    plt.plot(yy_modelB,xx, 'k-', color='red')
    print_margin(a_modelA, a_modelB, ymin, ymax, modelA, modelB, gamma)
    plt.show()

def compare_lr_gamma(X,gamma,y):
    X_gamma=X*gamma
    trained_lr = log_reg(X, y)
    trained_lr_gamma = log_reg(X_gamma, y)

    plot_graphs(trained_lr, trained_lr_gamma, gamma)

def compare_perc_gamma(X,gamma,y):
    X_gamma=X*gamma
    trained_perc = perceptron(X, y)
    trained_perc_gamma = perceptron(X_gamma, y)

    plot_graphs(trained_perc, trained_perc_gamma, gamma)

def compare_lr_perc(X,y):
    trained_lr = log_reg(X, y)
    trained_perc = perceptron(X, y)

    plot_graphs(trained_lr, trained_perc)

def print_margin(a_modelA, a_modelB, ymin, ymax, modelA, modelB, gamma=False):
    w_modelA = modelA.coef_[0]
    p1 = np.asarray([a_modelA*ymin-(modelA.intercept_[0])/w_modelA[1] ,ymin])
    p2 = np.asarray([a_modelA*ymax-(modelA.intercept_[0])/w_modelA[1] ,ymax])
    p3 = np.asarray([1,1])
    d1 = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    p3 = [-1,-1]
    d2 = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    modelA_name = str(modelA).split('(')[0]
    print("Margin from ", modelA_name, "line is : ",d1, d2)

    w_modelB = modelB.coef_[0]
    p1 = np.asarray([a_modelB*ymin-(modelB.intercept_[0])/w_modelB[1] ,ymin])
    p2 = np.asarray([a_modelB*ymax-(modelB.intercept_[0])/w_modelB[1] ,ymax])
    p3 = np.asarray([1,1])
    d1 = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    p3 = [-1,-1]
    d2 = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    modelB_name = str(modelB).split('(')[0]
    print(gamma)
    print("Margin from ", modelB_name, "with gamma "+str(gamma) if gamma else "","line is : ",d1,d2)

# X = [[1,1],[2,2],[4,4],[5,5]]
# y = [0,0,1,1]
X = [[1,1],[2,2],[-1,-1],[-2,-2]]
X = np.asarray(X)
y = [1,1,-1,-1]
gamma = 0.1

compare_lr_perc(X,y)
compare_perc_gamma(X,gamma,y)
compare_lr_gamma(X,gamma,y)