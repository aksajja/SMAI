# Ex-OR problem

import sklearn.svm as svm
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()

def plot_graph(modelA,X,y,subplot_value,C,_kernel_type):
    colormap = np.array(['r', 'k'])
    # Plot the original data
    
    ## Calc the hyperplane (decision boundary)
    # ModelA
    ymin, ymax = plt.ylim()
    xx = []
    x_coordinates = np.linspace(-0.5,0.5,100)
    y_coordinates = np.linspace(-0.5,0.5,100)
    for i in x_coordinates:
        for j in y_coordinates:
            xx.append([i,j])
    xx = np.asarray(xx)
    yy_modelA = modelA.predict(xx)
    unique_elements, counts_elements = np.unique(yy_modelA, return_counts=True)
    # Plot the line
    plots1 = fig.add_subplot(subplot_value)
    plots1.scatter(X[:2,0], X[:2,1], c=colormap[0], s=20)
    plots1.scatter(X[2:,0], X[2:,1], c=colormap[1], s=20)
    title = "C="+str(C)+" Kernel_type="+_kernel_type
    plots1.title.set_text(title)
    plots1.legend()
    plots1.scatter(xx[:,0], xx[:,1], c=yy_modelA, s=20)
    # plots1.plot(yy_modelA,xx, 'k-')
#     plt.show()

classA = [[1,1],[-1,-1]]
classB = [[1,-1],[-1,1]]
X = np.append(classA,classB,axis=0)
y = [1,1,-1,-1]

C = [0.1,1,100]
kernels = ['rbf','poly','sigmoid']
subplot_value = 331
for c in C:
    for _kernel_type in kernels:
        solver = svm.SVC(kernel=_kernel_type,C=c,degree=2, coef0=0.1)  # Degree>2 doesn't seem to work for polynomial kernel. coef0=0.1/1 work but not 100.
        solver.fit(X,y)
        plot_graph(solver,X,y,subplot_value,c,_kernel_type)
        subplot_value+=1

plt.show()